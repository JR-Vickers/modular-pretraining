"""
Generate MaxEnt continuations on the SAME prompts as generate.py and merge them
into the big generations CSV.

MaxEnt is an unlearning method applied to a base model, so (like Filtering) it
produces one BaseTransformer checkpoint per retain set, found under
  results/stories_maxent_seed1/seed_1/maxent_0N/<retain_key>/checkpoint.pth

We reuse generate.py's prompt-selection helpers verbatim (same select_seed,
prompt_len, tokenizer/EOS) so the prompts are byte-identical to the existing
rows, then append rows with method="MaxEnt", retained=<retain_key>. The merge is
idempotent: any pre-existing rows for this method are dropped before appending.

Run on one GPU:
  export OMP_NUM_THREADS=16 && python -m src.run.experiment.stories.generate.generate_maxent
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from src.run.experiment.config import GetStoriesConfig
from src.run.experiment.stories.generate.generate import (
    EXP_DIR, EOS_FALLBACK, STORIES_DATA,
    get_labels, select_prompts, load_base, generate_batch,
)

MAXENT_ROOT = EXP_DIR / "results" / "stories_maxent_seed1" / "seed_1"
GEN_CSV = EXP_DIR / "analysis" / "generations" / "generations.csv"
HEADER = ["method", "retained", "prompt_topic", "prompt", "continuation"]


def discover_maxent_ckpts(root: Path):
    """[(retain_key, checkpoint_path), ...] from maxent_0N/<retain_key>/checkpoint.pth."""
    out = []
    for stage_dir in sorted(root.glob("maxent_*")):
        for key_dir in sorted(p for p in stage_dir.iterdir() if p.is_dir()):
            ck = key_dir / "checkpoint.pth"
            if ck.exists():
                out.append((key_dir.name, ck))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--maxent_root", default=str(MAXENT_ROOT))
    p.add_argument("--csv", default=str(GEN_CSV))
    p.add_argument("--method_name", default="MaxEnt")
    p.add_argument("--n_stories", type=int, default=100)
    p.add_argument("--prompt_len", type=int, default=30)
    p.add_argument("--max_new", type=int, default=None)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--select_seed", type=int, default=1234)
    p.add_argument("--gen_seed", type=int, default=0)
    args = p.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")

    labels, aux_labels = get_labels()
    meta = json.load(open(STORIES_DATA / "metadata.json"))
    tok = AutoTokenizer.from_pretrained(meta["all"]["tokenizer"])
    eos = tok.eos_token_id if tok.eos_token_id is not None else EOS_FALLBACK
    ctx_len = GetStoriesConfig().model.ctx_len
    max_new = args.max_new if args.max_new is not None else (ctx_len - args.prompt_len)

    # identical prompt selection to generate.py -> prompts match existing rows
    prompts_by_topic = select_prompts(
        aux_labels, eos, args.n_stories, args.prompt_len, args.select_seed
    )
    prompt_ids = {t: [s[:args.prompt_len] for s in v] for t, v in prompts_by_topic.items()}
    prompt_text = {t: [tok.decode(pp, skip_special_tokens=True) for pp in v]
                   for t, v in prompt_ids.items()}

    ckpts = discover_maxent_ckpts(Path(args.maxent_root))
    assert ckpts, f"no maxent checkpoints under {args.maxent_root}"
    print(f"[maxent-gen] eos={eos} max_new={max_new} temp={args.temperature}", flush=True)
    print(f"[maxent-gen] {len(ckpts)} checkpoints: {[k for k, _ in ckpts]}", flush=True)

    g = torch.Generator(device=device).manual_seed(args.gen_seed)
    new_rows = []
    for retained, ck in ckpts:
        model = load_base(ck, eos, device)
        for topic in aux_labels:
            cont = generate_batch(
                model, prompt_ids[topic], False, None, labels,
                device, eos, max_new, args.temperature, g,
            )
            for ptext, cids in zip(prompt_text[topic], cont):
                new_rows.append([args.method_name, retained, topic, ptext,
                                 tok.decode(cids, skip_special_tokens=True)])
        del model
        torch.cuda.empty_cache()
        print(f"[maxent-gen] {args.method_name} | retained={retained} done", flush=True)

    # idempotent merge: keep all non-<method> rows, append fresh ones
    csv_path = Path(args.csv)
    existing = []
    if csv_path.exists():
        with open(csv_path) as f:
            rd = csv.reader(f)
            hdr = next(rd)
            assert hdr == HEADER, f"unexpected header {hdr}"
            existing = [row for row in rd if row and row[0] != args.method_name]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(existing)
        w.writerows(new_rows)
    print(f"[maxent-gen] wrote {csv_path}: +{len(new_rows)} {args.method_name} rows "
          f"(total {len(existing) + len(new_rows)})", flush=True)


if __name__ == "__main__":
    main()
