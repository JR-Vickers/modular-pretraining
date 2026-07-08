"""Per-token (and per-word) gold-logp differences for the <domain>_curated_sorted
sentences (all 3 domains), used to drive MCQ generation: the words where the
core-only filtered model is most deficient vs. baseline are the ones to quiz on.

For each line of <domain>_curated_sorted.txt we emit:
  original   : the sentence
  logp_diff  : length-normalized (base_mean - filt_mean)   [same convention as _sorted]
  base_mean, filt_mean, n_tok
  words      : [[word, summed (base-filt) diff over its tokens], ...] in sentence order
  toks       : [[token_str, (base-filt) diff], ...] in sentence order

-> analysis/logp/<domain>_pertok.json   (list, same order as the sorted txt)

Run on GPU:  srun ... python analysis/logp/pertoken.py
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve()
AGI = HERE.parents[2]
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


def gold_logp(model, ids):
    """Per-token gold logp for tokens[1:] (len = len(ids)-1)."""
    import torch
    x = torch.tensor([ids], device="cuda")
    with torch.no_grad():
        logits = model(tokens=x)[0]
        lp = torch.log_softmax(logits[0, :-1].float(), dim=-1)
        g = torch.tensor(ids[1:], device="cuda")
        return lp[torch.arange(lp.shape[0], device="cuda"), g].cpu().numpy()


def main():
    import torch
    from transformers import AutoTokenizer
    assert torch.cuda.is_available(), "no CUDA (run under srun/sbatch)"

    ctx = json.loads((BASE_DIR / "config.json").read_text())["model"]["ctx_len"]
    tok = AutoTokenizer.from_pretrained(TOKENIZER, use_fast=True)

    # tokenize every domain up front (with char offsets)
    D = {}
    for dom in DOMAINS:
        src = AGI / "analysis/logp" / f"{dom}_curated_sorted.txt"
        sents = [ln.rstrip("\n") for ln in src.read_text().splitlines() if ln.strip()]
        enc = [tok(s, return_offsets_mapping=True) for s in sents]
        ids_list  = [e["input_ids"][:ctx] for e in enc]
        offs_list = [e["offset_mapping"][:ctx] for e in enc]
        for i, x in enumerate(ids_list):
            if len(x) < 2:
                ids_list[i] = x + x[:1]; offs_list[i] = offs_list[i] + offs_list[i][:1]
        D[dom] = {"sents": sents, "ids": ids_list, "offs": offs_list}
        print(f"{dom}: {len(sents)} sentences", flush=True)

    print("== baseline (all-data) ==", flush=True)
    m = build_base(BASE_CKPT)
    for dom in DOMAINS:
        D[dom]["base"] = [gold_logp(m, ids) for ids in D[dom]["ids"]]
    del m; torch.cuda.empty_cache()
    print("== filtered (core-only) ==", flush=True)
    m = build_base(FILT_CKPT)
    for dom in DOMAINS:
        D[dom]["filt"] = [gold_logp(m, ids) for ids in D[dom]["ids"]]
    del m; torch.cuda.empty_cache()

    for dom in DOMAINS:
        d = D[dom]
        out = []
        for si, s in enumerate(d["sents"]):
            ids, offs = d["ids"][si], d["offs"][si]
            diff = d["base"][si] - d["filt"][si]          # aligned to ids[1:]
            n = len(diff)
            base_mean = float(d["base"][si].sum() / n)
            filt_mean = float(d["filt"][si].sum() / n)
            toks = [[tok.decode([ids[i]]), round(float(diff[i-1]), 3)] for i in range(1, len(ids))]
            words_spans, pos = [], 0
            for w in s.split():
                j = s.find(w, pos)
                words_spans.append((w, j, j + len(w))); pos = j + len(w)
            wdiff = [0.0] * len(words_spans)
            for i in range(1, len(ids)):
                c0 = offs[i][0]
                for wi, (w, a, b) in enumerate(words_spans):
                    if a <= c0 < b:
                        wdiff[wi] += float(diff[i-1]); break
            words = [[w, round(dv, 3)] for (w, a, b), dv in zip(words_spans, wdiff)]
            out.append({
                "original": s,
                "logp_diff": round(base_mean - filt_mean, 4),
                "base_mean": round(base_mean, 4),
                "filt_mean": round(filt_mean, 4),
                "n_tok": n,
                "words": words,
                "toks": toks,
            })
        outp = AGI / "analysis/logp" / f"{dom}_pertok.json"
        outp.write_text(json.dumps(out, ensure_ascii=False, indent=1))
        print(f"{dom}: wrote {len(out)} -> {outp.name}", flush=True)


if __name__ == "__main__":
    main()
