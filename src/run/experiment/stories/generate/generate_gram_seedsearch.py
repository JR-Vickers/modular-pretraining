"""
Seed-search GRAM continuations for the appendix prompts.

The new-hyperparameter GRAM is sampled at temperature 1.0, so a single draw can
miss the desired behaviour. For each appendix prompt and each GRAM config
(capability retained = core+topic, capability removed = core), we draw K
independent samples in one batch and dump them to JSON, so we can pick a draw
as good as the previous (old-GRAM) continuation.

Run on one GPU:
  python -m src.run.experiment.stories.generate.generate_gram_seedsearch
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from src.run.experiment.config import GetStoriesConfig
from src.run.experiment.stories.generate.generate import (
    EXP_DIR, EOS_FALLBACK, STORIES_DATA,
    get_labels, select_prompts, load_routed, generate_batch,
)

DEFAULT_GRAM_DIR = EXP_DIR / "results" / "stories" / "seed_1" / "20260616035947118035" / "routed"
OUT = EXP_DIR / "analysis" / "generations" / "gram_seedsearch.json"

# (topic, prompt prefix) for the two examples currently in the paper appendix.
APPENDIX = [
    ("alien-encounters", "before dawn, samuel woke up to a strange noise"),
    ("a-deadline-or-time-limit", "sparkles filled the air at the holiday celebration"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gram_dir", default=str(DEFAULT_GRAM_DIR))
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--k", type=int, default=48, help="samples per (prompt, config)")
    p.add_argument("--prompt_len", type=int, default=30)
    p.add_argument("--n_stories", type=int, default=100)
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

    # same prompt selection as generate.py -> resolve the exact appendix prompts
    prompts_by_topic = select_prompts(
        aux_labels, eos, args.n_stories, args.prompt_len, args.select_seed
    )

    def find_prompt_ids(topic, prefix):
        for story in prompts_by_topic[topic]:
            ids = story[:args.prompt_len]
            if tok.decode(ids, skip_special_tokens=True).startswith(prefix):
                return ids
        raise AssertionError(f"prompt not found: {topic} / {prefix!r}")

    model, arch = load_routed(Path(args.gram_dir), labels, eos, device)
    assert arch == "moe"
    print(f"[gram-seedsearch] gram_dir={args.gram_dir} k={args.k} max_new={max_new}", flush=True)

    g = torch.Generator(device=device).manual_seed(args.gen_seed)
    result = {}
    for topic, prefix in APPENDIX:
        ids = find_prompt_ids(topic, prefix)
        prompt_text = tok.decode(ids, skip_special_tokens=True)
        entry = {"prompt": prompt_text, "topic": topic, "retained": [], "removed": []}
        for cfgname, experts in [("retained", ["core", topic]), ("removed", ["core"])]:
            batch = [ids] * args.k  # K identical prompts -> K independent samples
            cont = generate_batch(
                model, batch, True, experts, labels,
                device, eos, max_new, args.temperature, g,
            )
            entry[cfgname] = [tok.decode(c, skip_special_tokens=True) for c in cont]
            print(f"[gram-seedsearch] {topic} / {cfgname}: {len(cont)} samples", flush=True)
        result[topic] = entry

    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"[gram-seedsearch] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
