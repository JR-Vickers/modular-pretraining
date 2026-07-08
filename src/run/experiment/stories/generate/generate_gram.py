"""
Re-generate GRAM (moe) continuations from a specific routed checkpoint on the
SAME prompts as generate.py, and swap them into the big generations CSV.

Used to replace the original GRAM rows with a new-hyperparameter GRAM run. The
checkpoint is a routed MoETransformer with 5 expert-mask configs (the retain
sets [core], [core,a], ..., [core,d]). Prompt selection reuses generate.py's
helpers verbatim so prompts stay byte-identical to the existing rows. The merge
is idempotent: pre-existing rows for this method are dropped before appending.

Run on one GPU:
  python -m src.run.experiment.stories.generate.generate_gram --gram_dir <path/to/routed>
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
    get_labels, select_prompts, load_routed, generate_batch, labels_to_str,
)

# Latest new-hparam GRAM run for seed 1 (robust_prc=0.5, aux_route_prc=0.3).
DEFAULT_GRAM_DIR = EXP_DIR / "results" / "stories" / "seed_1" / "20260616035947118035" / "routed"
GEN_CSV = EXP_DIR / "analysis" / "generations" / "generations.csv"
HEADER = ["method", "retained", "prompt_topic", "prompt", "continuation"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gram_dir", default=str(DEFAULT_GRAM_DIR),
                   help="routed stage dir with stage.json + checkpoint.pth")
    p.add_argument("--csv", default=str(GEN_CSV))
    p.add_argument("--method_name", default="GRAM")
    p.add_argument("--n_stories", type=int, default=100)
    p.add_argument("--prompt_len", type=int, default=30)
    p.add_argument("--max_new", type=int, default=None)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--select_seed", type=int, default=1234)
    p.add_argument("--gen_seed", type=int, default=0)
    args = p.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")

    gram_dir = Path(args.gram_dir)
    stage = json.load(open(gram_dir / "stage.json"))["stage"]
    print(f"[gram-gen] checkpoint={gram_dir}", flush=True)
    print(f"[gram-gen] arch={stage['model']['arch']} robust_prc={stage.get('robust_prc')} "
          f"aux_route_prc={stage.get('aux_route_prc')}", flush=True)

    labels, aux_labels = get_labels()
    meta = json.load(open(STORIES_DATA / "metadata.json"))
    tok = AutoTokenizer.from_pretrained(meta["all"]["tokenizer"])
    eos = tok.eos_token_id if tok.eos_token_id is not None else EOS_FALLBACK
    ctx_len = GetStoriesConfig().model.ctx_len
    max_new = args.max_new if args.max_new is not None else (ctx_len - args.prompt_len)

    prompts_by_topic = select_prompts(
        aux_labels, eos, args.n_stories, args.prompt_len, args.select_seed
    )
    prompt_ids = {t: [s[:args.prompt_len] for s in v] for t, v in prompts_by_topic.items()}
    prompt_text = {t: [tok.decode(pp, skip_special_tokens=True) for pp in v]
                   for t, v in prompt_ids.items()}

    model, arch = load_routed(gram_dir, labels, eos, device)
    assert arch == "moe", f"expected moe (GRAM), got {arch}"

    retain_sets = [["core"]] + [["core", a] for a in sorted(aux_labels)]
    g = torch.Generator(device=device).manual_seed(args.gen_seed)
    new_rows = []
    for experts in retain_sets:
        retained = labels_to_str(experts)
        for topic in aux_labels:
            cont = generate_batch(
                model, prompt_ids[topic], True, list(experts), labels,
                device, eos, max_new, args.temperature, g,
            )
            for ptext, cids in zip(prompt_text[topic], cont):
                new_rows.append([args.method_name, retained, topic, ptext,
                                 tok.decode(cids, skip_special_tokens=True)])
        print(f"[gram-gen] {args.method_name} | retained={retained} done", flush=True)

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
    print(f"[gram-gen] wrote {csv_path}: swapped in {len(new_rows)} {args.method_name} rows "
          f"(total {len(existing) + len(new_rows)})", flush=True)


if __name__ == "__main__":
    main()
