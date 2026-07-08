"""Rank each <domain>_curated.txt sentence by how much domain knowledge the
core-only *filtered* 800M model lost relative to the *baseline* all-data 800M model.

For every line we compute the per-token gold (ground-truth) log-probability under
each model, take the cumulative difference (base - filt) over the scored tokens, and
normalize by sentence length (number of scored tokens). Sentences where the filtered
model is most deficient vs. baseline score highest (= most domain-specific knowledge).

    score = mean_t [ logp_base(t) - logp_filt(t) ]            (larger -> ranks higher)

Writes, per domain:
  analysis/logp/<domain>_curated_sorted.txt  -- sentences, best-first (just the lines)
  analysis/logp/<domain>_curated_sorted.tsv  -- score \t base_mean \t filt_mean \t n_tok \t sentence

Run on GPU:  srun ... python analysis/logp/score_curated.py
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve()
AGI = HERE.parents[2]                      # .../AGI-1699-New-Dataset-Runs
sys.path.insert(0, str(AGI))
RES = AGI / "results/scaling/realistic"
TOKENIZER = "EleutherAI/gpt-neo-125M"

BASE_DIR  = RES / "base/800M/seed_1/20260422084509038162"
BASE_CKPT = BASE_DIR / "baseline/checkpoint.pth"
FILT_CKPT = RES / "filtering/800M/seed_1/20260514032047887159/filtering/core/checkpoint.pth"

DOMAINS = ["biology", "cyber", "nuclear"]


def model_fields(m, extra=()):
    keep = {"arch", "ctx_len", "vocab_size", "num_layers", "num_heads", "num_key_value",
            "attn_bias", "eos_token_id", "embed_dim", "mlp_dim", *extra}
    return {k: v for k, v in m.items() if k in keep}


def _load(model, ckpt):
    import torch
    t0 = time.time()
    sd = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    model.load_state_dict(sd["model"]); del sd
    model.eval()
    print(f"    loaded {Path(ckpt).relative_to(RES)} in {time.time()-t0:.1f}s", flush=True)


def build_base(ckpt):
    import torch
    from src.model.base import BaseTransformer
    from src.model.config import ModelConfig
    cfg = json.loads((BASE_DIR / "config.json").read_text())["model"]
    m = BaseTransformer(ModelConfig(**model_fields(cfg))).to("cuda", torch.bfloat16)
    _load(m, ckpt); return m


def gold_logp_sum(model, ids):
    """Return (sum of gold-token logp, n_scored_tokens) for one sentence."""
    import torch
    x = torch.tensor([ids], device="cuda")
    with torch.no_grad():
        logits = model(tokens=x)[0]
        lp = torch.log_softmax(logits[0, :-1].float(), dim=-1)
        g = torch.tensor(ids[1:], device="cuda")
        gold = lp[torch.arange(lp.shape[0], device="cuda"), g]
        return float(gold.sum().cpu()), int(gold.shape[0])


def main():
    import torch
    from transformers import AutoTokenizer
    assert torch.cuda.is_available(), "no CUDA (run under srun/sbatch)"

    ctx = json.loads((BASE_DIR / "config.json").read_text())["model"]["ctx_len"]
    tok = AutoTokenizer.from_pretrained(TOKENIZER, use_fast=True)

    # load + tokenize curated sentences (mirror run.py: encode, clip to ctx, pad len<2)
    data = {}
    for dom in DOMAINS:
        sents = [ln.rstrip("\n") for ln in (AGI / "analysis/logp" / f"{dom}_curated.txt")
                 .read_text().splitlines() if ln.strip()]
        ids = [tok.encode(s)[:ctx] for s in sents]
        ids = [x if len(x) >= 2 else x + x[:1] for x in ids]
        data[dom] = {"sents": sents, "ids": ids}
        print(f"  {dom}: {len(sents)} sentences", flush=True)

    def score_with(ckpt, key):
        m = build_base(ckpt)
        for dom in DOMAINS:
            sums, ns = [], []
            for ids in data[dom]["ids"]:
                s, n = gold_logp_sum(m, ids)
                sums.append(s); ns.append(n)
            data[dom][key] = np.array(sums); data[dom]["n"] = np.array(ns)
        del m; torch.cuda.empty_cache()

    print("== baseline (all-data) ==", flush=True); score_with(BASE_CKPT, "base")
    print("== filtered (core-only) ==", flush=True); score_with(FILT_CKPT, "filt")

    for dom in DOMAINS:
        d = data[dom]
        n = d["n"]
        base_mean = d["base"] / n
        filt_mean = d["filt"] / n
        score = base_mean - filt_mean                      # mean per-token deficit (base - filt)
        order = np.argsort(-score)                         # larger difference first
        outtxt = AGI / "analysis/logp" / f"{dom}_curated_sorted.txt"
        outtsv = AGI / "analysis/logp" / f"{dom}_curated_sorted.tsv"
        with open(outtxt, "w") as ft, open(outtsv, "w") as fv:
            for i in order:
                ft.write(d["sents"][i] + "\n")
                fv.write(f"{score[i]:.6f}\t{base_mean[i]:.6f}\t{filt_mean[i]:.6f}\t{n[i]}\t{d['sents'][i]}\n")
        print(f"{dom}: wrote {len(order)} -> {outtxt.name} (score {score[order[0]]:.3f}..{score[order[-1]]:.3f})",
              flush=True)


if __name__ == "__main__":
    main()
