"""Per-token differential-knowledge viz, one file for both scales.

    python run.py --scale 5B        # GPU; writes results/metric/5B/{data.json,viz.html}
    python run.py --scale 800M      # GPU; writes results/metric/800M/{data.json,viz.html}
    python run.py --scale 800M --viz-only   # CPU; rebuild HTML from data.json

For each topic dataset (analysis/logp/<topic>_definitions.txt) we compute per-token gold
log-probability for the base model, an anchor "filtered" model (the model that forgot the
topic's domain), and each method in {GRAM, FT-LoRA, (MaxEnt)} under two deployments: core-only
and core+aux (aux = the topic's matching expert/checkpoint).

Coloring (single HTML, Topic dropdown x method tabs): each token's deficit vs base,
(base-method), divided by sigma = the 90th-percentile of the within-sentence filtering deficits
(base-filt), clipped to [0,1], floored at 1 nat; white = matches base, red = predicted as poorly
as a typical token the filtered model forgot.

Scale differences (handled by SCALES below):
  5B  : methods GRAM/FT-LoRA; topics cyber,nuclear; anchor = core+bio filter (forgot cyber/nuclear).
        (biology excluded: the only 5B filter retains bio, so there is no contrast.)
  800M: methods GRAM/FT-LoRA/MaxEnt; topics biology,cyber,nuclear; anchor = core-only filter.
"""
from __future__ import annotations
import argparse, glob, json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve()
AGI = HERE.parents[5]
sys.path.insert(0, str(AGI))
RES = AGI / "results/scaling/realistic"
TOKENIZER = "EleutherAI/gpt-neo-125M"

_ap = argparse.ArgumentParser()
_ap.add_argument("--scale", choices=["5B", "800M"], default="800M")
_ap.add_argument("--viz-only", action="store_true")
ARGS = _ap.parse_args()
SCALE = ARGS.scale

# topic display -> (dataset file under analysis/, aux expert label, short aux name)
TOPIC_INFO = {
    "biology": ("logp/biology_definitions.txt", "papers-biology", "bio"),
    "cyber":   ("logp/cyber_definitions.txt",   "papers-cyber",   "cyber"),
    "nuclear": ("logp/nuclear_definitions.txt", "papers-nuclear", "nuclear"),
}

SCALES = {
    "5B": {
        "topics": ["cyber", "nuclear"],
        "methods": [("GRAM", "moe"), ("FT-LoRA", "lora")],
        "base_dir":  RES / "base/5B/seed_1/run_1",
        "base_ckpt": RES / "base/5B/seed_1/run_1/baseline/checkpoint.pth",
        "filt_ckpt": RES / "filtering/5B/seed_1/run_1/filtering/core_papers-biology/checkpoint.pth",
        "grmoe_dir": RES / "grmoe/5B/seed_1/run_1",
        "grmoe_ckpt": RES / "grmoe/5B/seed_1/run_1/routed/checkpoint.pth",
        "lora_dir":  RES / "lora/5B/seed_1/run_1",
        "lora_ckpt": RES / "lora/5B/seed_1/run_1/routed/checkpoint.pth",
        "maxent_dir": None,
        "show_filtering": False,
    },
    "800M": {
        "topics": ["biology", "cyber", "nuclear"],
        "methods": [("GRAM", "moe"), ("FT-LoRA", "lora"), ("MaxEnt", "maxent")],
        "base_dir":  RES / "base/800M/seed_1/20260422084509038162",
        "base_ckpt": RES / "base/800M/seed_1/20260422084509038162/baseline/checkpoint.pth",
        "filt_ckpt": RES / "filtering/800M/seed_1/20260514032047887159/filtering/core/checkpoint.pth",
        "grmoe_dir": RES / "grmoe/800M/seed_1/20260422115512415711",
        "grmoe_ckpt": RES / "grmoe/800M/seed_1/20260422115512415711/routed/checkpoint.pth",
        "lora_dir":  RES / "lora/800M/seed_1/20260422121913832756",
        "lora_ckpt": RES / "lora/800M/seed_1/20260422121913832756/routed/checkpoint.pth",
        "maxent_dir": RES / "maxent/800M/seed_1/20260616005741246286",
        "show_filtering": True,
        "filt_root": RES / "filtering/800M/seed_1",
    },
}
CFG = SCALES[SCALE]
OUT = AGI / "results" / "metric" / SCALE
OUT.mkdir(parents=True, exist_ok=True)
DATA_JSON = OUT / "data.json"
HTML = OUT / "viz.html"


# --------------------------------------------------------------------------- #
# model building / scoring
# --------------------------------------------------------------------------- #
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
    cfg = json.loads((CFG["base_dir"] / "config.json").read_text())["model"]
    m = BaseTransformer(ModelConfig(**model_fields(cfg))).to("cuda", torch.bfloat16)
    _load(m, ckpt); return m

def build_routed(kind, ckpt, cfg_dir, labels):
    import torch
    from src.model.moe import MoETransformer
    from src.model.lora import LoRATransformer
    from src.model.config import RoutedModelConfig
    raw = json.loads((cfg_dir / "config.json").read_text())["stages"][0]["model"]
    cfg = RoutedModelConfig(**model_fields(raw, extra=("core_param_prc", "aux_param_prc")))
    cls = MoETransformer if kind == "moe" else LoRATransformer
    m = cls(cfg, labels=labels).to("cuda", torch.bfloat16); _load(m, ckpt); return m

def gold_logp(model, ids, mask=None):
    import torch
    x = torch.tensor([ids], device="cuda")
    with torch.no_grad():
        logits = (model(tokens=x)[0] if mask is None
                  else model(tokens=x, targets=None, fwd_mask=mask, bck_mask=mask)[0])
        lp = torch.log_softmax(logits[0, :-1].float(), dim=-1)
        g = torch.tensor(ids[1:], device="cuda")
        return lp[torch.arange(lp.shape[0], device="cuda"), g].cpu().numpy()

def maxent_ckpt(labels):
    """Resolve the 800M maxent checkpoint retaining exactly `labels`."""
    for p in glob.glob(str(CFG["maxent_dir"] / "maxent_*/*/checkpoint.pth")):
        if sorted(Path(p).parent.name.split("_")) == sorted(labels):
            return Path(p)
    raise FileNotFoundError(f"no maxent ckpt for {labels} under {CFG['maxent_dir']}")


def filt_aux_ckpt(aux):
    """Resolve the 800M core+aux filtering checkpoint for `aux` (scattered across timestamps)."""
    hits = sorted(glob.glob(str(CFG["filt_root"] / f"*/filtering/core_{aux}/checkpoint.pth")))
    if not hits:
        raise FileNotFoundError(f"no core+{aux} filtering ckpt under {CFG['filt_root']}")
    return Path(hits[-1])


def compute():
    import torch
    from transformers import AutoTokenizer
    from src.run.util.tools import get_exp_mask
    assert torch.cuda.is_available(), "no CUDA (run under srun/sbatch)"

    labels = json.loads((CFG["grmoe_dir"] / "config.json").read_text())["run"]["labels"]
    ctx = json.loads((CFG["base_dir"] / "config.json").read_text())["model"]["ctx_len"]
    tok = AutoTokenizer.from_pretrained(TOKENIZER, use_fast=True)
    topics = CFG["topics"]
    print(f"scale={SCALE} topics={topics} methods={[m for m,_ in CFG['methods']]} labels={labels}", flush=True)

    T = {}
    for tp in topics:
        rel, aux, short = TOPIC_INFO[tp]
        sents = [ln.rstrip("\n") for ln in (AGI / "analysis" / rel).read_text().splitlines() if ln.strip()]
        ids = [tok.encode(s)[:ctx] for s in sents]
        ids = [x if len(x) >= 2 else x + x[:1] for x in ids]
        T[tp] = {"ids": ids, "strs": [[tok.decode([t]) for t in x] for x in ids], "aux": aux, "short": short}
        print(f"  {tp}: {len(ids)} sentences (aux={aux})", flush=True)

    res = {tp: {} for tp in topics}          # res[topic][colkey] = list(per sentence) of per-token logp
    def run_all(model, key, mask=None, only=None):
        for tp in topics:
            if only and tp != only: continue
            res[tp][key] = [gold_logp(model, ids, mask) for ids in T[tp]["ids"]]

    print("== base ==", flush=True)
    m = build_base(CFG["base_ckpt"]); run_all(m, "base"); del m; torch.cuda.empty_cache()
    print("== filt (anchor) ==", flush=True)
    m = build_base(CFG["filt_ckpt"]); run_all(m, "filt"); del m; torch.cuda.empty_cache()

    for disp, kind in CFG["methods"]:
        if kind in ("moe", "lora"):
            print(f"== {disp} (routed, masked) ==", flush=True)
            ck = CFG["grmoe_ckpt"] if kind == "moe" else CFG["lora_ckpt"]
            cd = CFG["grmoe_dir"] if kind == "moe" else CFG["lora_dir"]
            model = build_routed(kind, ck, cd, labels)
            for tp in topics:
                run_all(model, f"{disp}|core", get_exp_mask(labels, ["core"], "cuda"), only=tp)
                run_all(model, f"{disp}|aux", get_exp_mask(labels, ["core", T[tp]["aux"]], "cuda"), only=tp)
            del model; torch.cuda.empty_cache()
        elif kind == "maxent":
            print(f"== {disp} (per-retain checkpoints) ==", flush=True)
            mc = build_base(maxent_ckpt(["core"])); run_all(mc, f"{disp}|core"); del mc; torch.cuda.empty_cache()
            for tp in topics:
                ma = build_base(maxent_ckpt(["core", T[tp]["aux"]]))
                run_all(ma, f"{disp}|aux", only=tp); del ma; torch.cuda.empty_cache()

    if CFG.get("show_filtering"):
        print("== Filtering (core = anchor alias; per-topic core+aux) ==", flush=True)
        for tp in topics:
            res[tp]["Filtering|core"] = res[tp]["filt"]   # core-only filter == the anchor model
        for tp in topics:
            mf = build_base(filt_aux_ckpt(T[tp]["aux"]))
            run_all(mf, "Filtering|aux", only=tp); del mf; torch.cuda.empty_cache()

    method_cols = [f"{d}|{v}" for d, _ in CFG["methods"] for v in ("core", "aux")]
    filt_cols = ["Filtering|core", "Filtering|aux"] if CFG.get("show_filtering") else []
    cols = ["base", "filt"] + method_cols + filt_cols
    tab_methods = [d for d, _ in CFG["methods"]] + (["Filtering"] if CFG.get("show_filtering") else [])
    out = {"scale": SCALE, "topics": topics, "methods": [d for d, _ in CFG["methods"]],
           "tab_methods": tab_methods,
           "auxname": {tp: T[tp]["short"] for tp in topics}, "cols": cols, "data": {}}
    for tp in topics:
        sents = []
        for si, strs in enumerate(T[tp]["strs"]):
            n = len(strs) - 1
            rows = [[strs[0]] + [None] * len(cols)]
            for j in range(n):
                rows.append([strs[j + 1]] + [round(float(res[tp][c][si][j]), 3) for c in cols])
            sents.append(rows)
        out["data"][tp] = sents
    DATA_JSON.write_text(json.dumps(out))
    print(f"wrote {DATA_JSON.name} ({DATA_JSON.stat().st_size/1e6:.2f} MB)", flush=True)


# --------------------------------------------------------------------------- #
# viz
# --------------------------------------------------------------------------- #
def _reorder(data):
    """Rank each topic's sentences most-compelling-first: content tokens that core-only
    gets wrong (large base-method) and core+aux recovers, averaged over methods."""
    import re
    cols = data["cols"]; ci = {c: cols.index(c) + 1 for c in cols}
    B, F = ci["base"], ci["filt"]; methods = data["methods"]
    def content(t): return len(re.sub(r"[^A-Za-z]", "", t.strip().replace("�", ""))) >= 3
    def score(se):
        gaps = sorted(max(t[B] - t[F], 0) for t in se if t[B] is not None and t[F] is not None)
        sig = max(gaps[int(0.9 * (len(gaps) - 1))], 1.0) if gaps else 1.0
        tot = 0.0
        for t in se[1:]:
            if t[B] is None or not content(t[0]): continue
            dcs = [max(0, min(1, (t[B] - t[ci[f'{m}|core']]) / sig)) for m in methods]
            das = [max(0, min(1, (t[B] - t[ci[f'{m}|aux']]) / sig)) for m in methods]
            dc, da = sum(dcs) / len(dcs), sum(das) / len(das)
            if dc > 0.5 and dc - da > 0.3: tot += (dc - da) * dc
        return tot
    for tp in data["topics"]:
        data["data"][tp].sort(key=lambda se: -score(se))


HTML_TEMPLATE = """<!doctype html><html><head><meta charset="utf-8">
<title>__SCALE__ per-token logp</title>
<style>
 body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:24px;color:#222;}
 h1{font-size:18px;} .sub{color:#666;font-size:13px;max-width:1050px;margin-bottom:10px;}
 .ctrl{margin:8px 0;font-size:13px;} select{font-size:13px;padding:3px 6px;margin-right:14px;}
 .tab{padding:6px 11px;border:1px solid #ccc;background:#f4f4f4;cursor:pointer;border-radius:6px;margin:0 5px 5px 0;font-size:12px;display:inline-block;}
 .tab.active{background:#222;color:#fff;border-color:#222;}
 .sent{margin:5px 0;padding:5px 8px;border-bottom:1px solid #eee;line-height:2.0;}
 .idx{color:#bbb;font-size:11px;margin-right:8px;} .tk{white-space:pre;border-radius:3px;padding:1px 0;}
 .first{background:#e9e9e9;color:#999;}
</style></head><body>
<h1>Gradient-routing __SCALE__ &mdash; per-token log-probability</h1>
<div class="sub">Choose a <b>Topic</b> and a <b>Method</b> tab; the <b>core+&lt;aux&gt;</b> tabs use that topic's matching
expert. Each token is colored by its deficit vs base, (base&minus;method), divided by <i>&sigma;</i> = the 90th
percentile of the within-sentence <b>filtering deficits</b> (base&minus;filtered), clipped to [0,1]:
white = matches base (general word, or knowledge retained), <b style="color:#d74b3c">red</b> = predicted as poorly as a
typical token the filtered model forgot. First token grey.</div>
<div class="ctrl">Topic: <select id="topic" onchange="rebuild()"></select> <span id="tabs"></span></div>
<div id="root">loading&hellip;</div>
<script>
const B64="__B64__";
const W=[255,255,255],R=[215,75,60],L=(a,b,f)=>Math.round(a+(b-a)*f);
const esc=q=>q.replace(/�/g,'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
let D=null, topic, mi=0, TABS=[];
function colr(c){ return 'rgb('+L(W[0],R[0],c)+','+L(W[1],R[1],c)+','+L(W[2],R[2],c)+')'; }
function p90(a){ if(!a.length)return 0; const s=a.slice().sort((x,y)=>x-y); return s[Math.floor(0.9*(s.length-1))]; }
function buildTabs(){ TABS=[]; (D.tab_methods||D.methods).forEach(m=>{TABS.push([m,'core']);TABS.push([m,'aux']);}); }
function tabCol(t){ return t[1]=='core'? t[0]+'|core' : t[0]+'|aux'; }
function tabLabel(t){ return t[1]=='core'? t[0]+' core' : t[0]+' core+'+D.auxname[topic]; }
function draw(){
  const ci=D.cols.indexOf(tabCol(TABS[mi]))+1;
  const bci=D.cols.indexOf('base')+1, fci=D.cols.indexOf('filt')+1;
  let h='';
  D.data[topic].forEach((se,i)=>{
    const gaps=[]; for(const t of se){const b=t[bci],f=t[fci]; if(b!=null&&f!=null)gaps.push(Math.max(b-f,0));}
    const sig=Math.max(p90(gaps),1.0);
    h+='<div class="sent"><span class="idx">'+(i+1)+'</span>';
    for(const t of se){
      const v=t[ci], b=t[bci];
      if(v==null||b==null){h+='<span class="tk first">'+esc(t[0])+'</span>';continue;}
      let c=(b-v)/sig; if(c<0)c=0; if(c>1)c=1;
      h+='<span class="tk" title="'+tabLabel(TABS[mi])+' logp='+v.toFixed(2)+' (base '+b.toFixed(2)+', deficit '+(b-v).toFixed(2)+', sigma '+sig.toFixed(2)+')" style="background:'+colr(c)+'">'+esc(t[0])+'</span>';
    }
    h+='</div>';
  });
  document.getElementById('root').innerHTML=h;
}
function renderTabs(){
  document.getElementById('tabs').innerHTML=TABS.map((t,i)=>'<span class="tab" data-i="'+i+'" onclick="setM('+i+')">'+tabLabel(t)+'</span>').join('');
  document.querySelectorAll('#tabs .tab').forEach(e=>e.classList.toggle('active',+e.dataset.i==mi));
}
function setM(i){mi=i;renderTabs();draw();}
function rebuild(){topic=document.getElementById('topic').value;renderTabs();draw();}
async function init(){
  try{
    const bin=Uint8Array.from(atob(B64),c=>c.charCodeAt(0));
    const buf=await new Response(new Blob([bin]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
    D=JSON.parse(new TextDecoder().decode(buf));
  }catch(e){document.getElementById('root').innerHTML='<p style="color:#c00">Failed to load data: '+e+'</p>';return;}
  topic=D.topics[0]; buildTabs();
  document.getElementById('topic').innerHTML=D.topics.map(t=>'<option>'+t+'</option>').join('');
  renderTabs(); draw();
}
init();
</script></body></html>"""


def make_viz():
    import gzip, base64
    data = json.loads(DATA_JSON.read_text())
    _reorder(data)
    b64 = base64.b64encode(gzip.compress(json.dumps(data, ensure_ascii=True).encode(), 6)).decode()
    HTML.write_text(HTML_TEMPLATE.replace("__SCALE__", SCALE).replace("__B64__", b64))
    print(f"wrote {HTML.name} ({HTML.stat().st_size/1e6:.2f} MB, gzip-inline)", flush=True)


if __name__ == "__main__":
    print(f"scale={SCALE} out={OUT}", flush=True)
    if not ARGS.viz_only:
        compute()
    make_viz()
