"""
Story-continuation generation across all seed_1 stories checkpoints.

For each method/model-config, prompt with the first ``prompt_len`` tokens of 100
held-out test stories per aux category and autoregressively sample a
continuation (temperature) until EOS or the context limit, batched.

Methods (all checkpoints under results/stories/seed_1/<run>/):
  - baseline       : 1 config  (full model; retained="all")
  - filtering      : 5 configs (one checkpoint per retain set)
  - GRAM   (moe)   : 1 checkpoint, 5 expert-mask configs  [core],[core,a],...,[core,d]
  - FT-LoRA (lora) : 1 checkpoint, 5 expert-mask configs
  - demix          : 1 checkpoint, 5 single-expert configs [core],[a],[b],[c],[d]

The same 100 stories per aux category are used across every method (selected
once, with a fixed seed).

Outputs (CSV):
  generations.csv   : method, retained, prompt_topic, prompt, continuation
  ground_truth.csv  : topic, story          (the 4 x 100 reference stories)

Run on one GPU:
  export OMP_NUM_THREADS=16 && torchrun --standalone --nproc_per_node=1 \
    -m src.run.experiment.stories.generate.generate
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from src.model.config import ModelConfig, RoutedModelConfig
from src.model.base import BaseTransformer
from src.model.moe import MoETransformer
from src.model.lora import LoRATransformer
from src.model.demix import DemixTransformer
from src.run.experiment.config import GetStoriesConfig
from src.run.util.tools import get_exp_mask

EOS_FALLBACK = 1  # stories [EOS] id (confirmed: delimits test bins)
EXP_DIR = Path(__file__).resolve().parents[4]
STORIES_DATA = EXP_DIR / "src" / "data" / "stories"
SEED1_ROOT = EXP_DIR / "results" / "stories" / "seed_1"


def labels_to_str(labels) -> str:
    labels = set(labels)
    parts = []
    if "core" in labels:
        parts.append("core")
        labels.discard("core")
    parts.extend(sorted(labels))
    return "_".join(parts)


# --------------------------------------------------------------------------- #
# discovery / data                                                            #
# --------------------------------------------------------------------------- #

def discover_run_dir() -> Path:
    """Latest seed_1 ts_dir with a baseline checkpoint + all routed/filtering ckpts."""
    cands = [
        d for d in SEED1_ROOT.iterdir()
        if d.is_dir() and (d / "baseline" / "checkpoint.pth").exists()
    ]
    if not cands:
        raise FileNotFoundError(f"No baseline checkpoint under {SEED1_ROOT}")
    return max(cands, key=lambda d: d.name)


def get_labels() -> tuple[list[str], list[str]]:
    meta = json.load(open(STORIES_DATA / "metadata.json"))
    all_labels = sorted(meta["all"]["labels"])
    aux_labels = all_labels[:4]
    labels = ["core"] + aux_labels  # expert/label order used at train time
    return labels, aux_labels


def split_stories(topic: str, eos: int) -> list[list[int]]:
    """Read a topic's test bin and split the token stream into stories on EOS."""
    arr = np.asarray(
        np.memmap(STORIES_DATA / f"{topic}_test.bin", dtype=np.uint16, mode="r")
    ).astype(np.int64)
    stories: list[list[int]] = []
    cur: list[int] = []
    for tok in arr.tolist():
        if tok == eos:
            if cur:
                stories.append(cur)
            cur = []
        else:
            cur.append(tok)
    if cur:
        stories.append(cur)
    return stories


def select_prompts(
    aux_labels: list[str], eos: int, n: int, prompt_len: int, seed: int,
) -> dict[str, list[list[int]]]:
    """Pick the same `n` stories (>= prompt_len tokens) per topic, fixed seed.

    Returns {topic: [story_token_ids, ...]} (full stories; prompt = first
    prompt_len tokens).
    """
    rng = random.Random(seed)
    out: dict[str, list[list[int]]] = {}
    for topic in aux_labels:
        stories = [s for s in split_stories(topic, eos) if len(s) >= prompt_len]
        if len(stories) < n:
            raise ValueError(
                f"{topic}: only {len(stories)} stories >= {prompt_len} tokens (need {n})"
            )
        idx = rng.sample(range(len(stories)), n)
        out[topic] = [stories[i] for i in idx]
    return out


# --------------------------------------------------------------------------- #
# model loading                                                               #
# --------------------------------------------------------------------------- #

def _load_state(ckpt_path: Path) -> dict:
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    return ck["model"]


def _base_model_config(eos: int, vocab_size: int) -> ModelConfig:
    cfg = GetStoriesConfig().model
    cfg.eos_token_id = eos
    cfg.vocab_size = vocab_size
    cfg.tokenizer = None
    return cfg


def load_base(ckpt_path: Path, eos: int, device) -> BaseTransformer:
    sd = _load_state(ckpt_path)
    vocab = sd["embed.weight"].shape[0]
    model = BaseTransformer(_base_model_config(eos, vocab))
    model.load_state_dict(sd)
    return model.to(device, dtype=torch.bfloat16).eval()


def load_routed(stage_dir: Path, labels: list[str], eos: int, device):
    """Build the right routed class from stage.json + checkpoint."""
    stage = json.load(open(stage_dir / "stage.json"))["stage"]
    m = stage["model"]
    arch = m["arch"]
    sd = _load_state(stage_dir / "checkpoint.pth")
    vocab = sd["embed.weight"].shape[0]
    base = _base_model_config(eos, vocab)
    rcfg = RoutedModelConfig.from_base(
        base, arch=arch,
        core_param_prc=m["core_param_prc"], aux_param_prc=m["aux_param_prc"],
    )
    cls = {"moe": MoETransformer, "lora": LoRATransformer, "demix": DemixTransformer}[arch]
    model = cls(rcfg, labels=labels)
    model.load_state_dict(sd)
    return model.to(device, dtype=torch.bfloat16).eval(), arch


# --------------------------------------------------------------------------- #
# generation                                                                  #
# --------------------------------------------------------------------------- #

@torch.inference_mode()
def generate_batch(
    model, prompts: list[list[int]], routed: bool, expert_labels, labels,
    device, eos: int, max_new: int, temperature: float, gen: torch.Generator,
    micro: int = 100,
) -> list[list[int]]:
    """Batched temperature sampling. prompts all share length=prompt_len.
    Returns the generated continuation token ids per prompt (EOS excluded)."""
    fwd = bck = None
    if routed:
        fwd = get_exp_mask(labels, expert_labels, device=device)
        bck = get_exp_mask(labels, expert_labels, device=device)

    out: list[list[int]] = []
    for start in range(0, len(prompts), micro):
        chunk = prompts[start:start + micro]
        x = torch.tensor(chunk, dtype=torch.long, device=device)  # (B, P)
        B = x.shape[0]
        finished = [False] * B
        cont: list[list[int]] = [[] for _ in range(B)]
        for _ in range(max_new):
            if routed:
                logits = model(tokens=x, targets=None, fwd_mask=fwd, bck_mask=bck)[0]
            else:
                logits = model(x)[0]
            nl = logits[:, -1, :].float() / temperature
            probs = torch.softmax(nl, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1, generator=gen).squeeze(1)
            x = torch.cat([x, nxt.unsqueeze(1)], dim=1)
            for i, tok in enumerate(nxt.tolist()):
                if finished[i]:
                    continue
                if tok == eos:
                    finished[i] = True
                else:
                    cont[i].append(tok)
            if all(finished):
                break
        out.extend(cont)
    return out


# --------------------------------------------------------------------------- #
# main                                                                        #
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", default=None, help="seed_1 ts_dir (default: auto-discover)")
    p.add_argument("--n_stories", type=int, default=100)
    p.add_argument("--prompt_len", type=int, default=30)
    p.add_argument("--max_new", type=int, default=None, help="default: ctx_len - prompt_len")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--select_seed", type=int, default=1234, help="fixes the 100 stories/topic")
    p.add_argument("--gen_seed", type=int, default=0, help="sampling RNG seed")
    p.add_argument("--out_dir", default=None, help="default: analysis/generations")
    args = p.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")

    run_dir = Path(args.run_dir) if args.run_dir else discover_run_dir()
    out_dir = Path(args.out_dir) if args.out_dir else (EXP_DIR / "analysis" / "generations")
    out_dir.mkdir(parents=True, exist_ok=True)

    labels, aux_labels = get_labels()

    meta = json.load(open(STORIES_DATA / "metadata.json"))
    tok = AutoTokenizer.from_pretrained(meta["all"]["tokenizer"])
    eos = tok.eos_token_id if tok.eos_token_id is not None else EOS_FALLBACK

    ctx_len = GetStoriesConfig().model.ctx_len
    max_new = args.max_new if args.max_new is not None else (ctx_len - args.prompt_len)

    print(f"[gen] run_dir={run_dir.name} eos={eos} labels={labels}", flush=True)
    print(f"[gen] n={args.n_stories} prompt_len={args.prompt_len} max_new={max_new} "
          f"temp={args.temperature}", flush=True)

    # ---- select the shared 100 stories per topic ----
    prompts_by_topic = select_prompts(
        aux_labels, eos, args.n_stories, args.prompt_len, args.select_seed,
    )

    # ---- ground-truth CSV ----
    gt_path = out_dir / "ground_truth.csv"
    with open(gt_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["topic", "story"])
        for topic in aux_labels:
            for story in prompts_by_topic[topic]:
                w.writerow([topic, tok.decode(story, skip_special_tokens=True)])
    print(f"[gen] wrote {gt_path}", flush=True)

    # prompts as first prompt_len tokens (decoded once for the CSV)
    prompt_ids_by_topic = {
        t: [s[:args.prompt_len] for s in stories]
        for t, stories in prompts_by_topic.items()
    }
    prompt_text_by_topic = {
        t: [tok.decode(p, skip_special_tokens=True) for p in plist]
        for t, plist in prompt_ids_by_topic.items()
    }

    # ---- build method/config plan ----
    RETAIN_SETS = [["core"]] + [["core", a] for a in aux_labels]   # GRAM / FT-LoRA / filtering
    DEMIX_SETS = [["core"]] + [[a] for a in aux_labels]            # demix single-expert

    gen_path = out_dir / "generations.csv"
    f = open(gen_path, "w", newline="")
    writer = csv.writer(f)
    writer.writerow(["method", "retained", "prompt_topic", "prompt", "continuation"])

    g = torch.Generator(device=device).manual_seed(args.gen_seed)

    def run_method(method: str, model, routed: bool, configs: list):
        """configs: list of (retained_str, expert_labels_or_None)."""
        for retained, experts in configs:
            for topic in aux_labels:
                cont = generate_batch(
                    model, prompt_ids_by_topic[topic], routed, experts, labels,
                    device, eos, max_new, args.temperature, g,
                )
                for prompt_text, c_ids in zip(prompt_text_by_topic[topic], cont):
                    writer.writerow([
                        method, retained, topic, prompt_text,
                        tok.decode(c_ids, skip_special_tokens=True),
                    ])
            f.flush()
            print(f"[gen] {method} | retained={retained} done", flush=True)

    # baseline (1 config)
    base = load_base(run_dir / "baseline" / "checkpoint.pth", eos, device)
    run_method("baseline", base, routed=False, configs=[("all", None)])
    del base; torch.cuda.empty_cache()

    # filtering (one checkpoint per retain set)
    filt_root = run_dir / "filtering"
    for rs in RETAIN_SETS:
        key = labels_to_str(rs)
        ckpt = filt_root / key / "checkpoint.pth"
        if not ckpt.exists():
            print(f"[gen] WARN: missing filtering ckpt {ckpt}", flush=True)
            continue
        fm = load_base(ckpt, eos, device)
        run_method("filtering", fm, routed=False, configs=[(key, None)])
        del fm; torch.cuda.empty_cache()

    # routed methods: GRAM (moe)=routed_01, FT-LoRA (lora)=routed_02, demix=routed_03
    routed_plan = {
        "routed_01": ("GRAM", RETAIN_SETS),
        "routed_02": ("FT-LoRA", RETAIN_SETS),
        "routed_03": ("demix", DEMIX_SETS),
    }
    for sub, (method, sets) in routed_plan.items():
        stage_dir = run_dir / sub
        if not (stage_dir / "checkpoint.pth").exists():
            print(f"[gen] WARN: missing {stage_dir}/checkpoint.pth", flush=True)
            continue
        model, arch = load_routed(stage_dir, labels, eos, device)
        configs = [(labels_to_str(s), list(s)) for s in sets]
        run_method(method, model, routed=True, configs=configs)
        del model; torch.cuda.empty_cache()

    f.close()
    print(f"[gen] wrote {gen_path}", flush=True)


if __name__ == "__main__":
    main()
