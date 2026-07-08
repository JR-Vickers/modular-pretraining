"""Build a dataset of sentences that introduce/define a technical term, from
papers-<domain>_test.bin, by scanning the entire test set.

Two recall modes (``--recall``):
  strict : only "X is defined as Y" / "X is characterized by Y" (the original gate).
  broad  : many definitional cue families (refers to, known/called/termed as, is-a
           category, denotes/represents/means, consists of/comprises, expressed/given
           as, author-framed "we define ...", "X is the <measure> of Y"), PLUS an
           acronym-introduction detector ("Long Form (LF)"). Higher false-positive
           rate by design -- intended to be filtered by an LLM downstream.

All matches are scored (cleaner, named, specific definitions rank higher) and written
sorted best-first, so taking the top-N is a quality cut. Default writes ALL unique
matches; pass --n_out to cap.
"""
import os, re, sys, random, argparse
os.environ.setdefault("HF_HOME", "/workspace-vast/ethanr/hf_cache")
import numpy as np
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--domain", default="biology", help="papers-<domain>_test.bin (biology, nuclear, cyber)")
ap.add_argument("--n_chunks", type=int, default=None, help="chunks to scan; default = ALL (entire test set)")
ap.add_argument("--n_out", type=int, default=None, help="max sentences to write; default = ALL unique matches")
ap.add_argument("--out", default=None)
ap.add_argument("--tsv", default=None)
ap.add_argument("--keywords", default=None,
                help="comma-separated keyword filter (case-insensitive substring); default = no filter.")
ap.add_argument("--recall", choices=["strict", "broad"], default="broad",
                help="strict = defined-as/characterized-by only; broad = many cues + acronym intro")
args = ap.parse_args()

# Keyword filter is OFF by default so we capture every definitional sentence in the test set.
KEYWORDS = []
if args.keywords is not None:
    KEYWORDS = [k.strip() for k in args.keywords.split(",") if k.strip()]
_KW = [k.lower() for k in KEYWORDS]
def has_keyword(s):
    if not _KW:
        return True
    sl = s.lower()
    return any(k in sl for k in _KW)

BIN = f"src/data/papers/papers-{args.domain}_test.bin"
CHUNK, EOS, SEED = 1024, 50256, 0
N_CHUNKS = args.n_chunks
N_OUT = args.n_out
OUT = args.out or f"analysis/logp/{args.domain}_definitions.txt"
TSV = args.tsv or f"analysis/logp/{args.domain}_definitions_scored.tsv"

tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M", use_fast=True)
arr = np.memmap(BIN, dtype=np.uint16, mode="r")
n_possible = arr.shape[0] // CHUNK
if N_CHUNKS is None or N_CHUNKS >= n_possible:        # iterate the ENTIRE test set
    starts = range(n_possible)
    print(f"scanning ALL {n_possible} chunks of {BIN} (recall={args.recall})", file=sys.stderr)
else:
    random.seed(SEED)
    starts = sorted(random.sample(range(n_possible), N_CHUNKS))
    print(f"sampling {len(starts)}/{n_possible} chunks (recall={args.recall})", file=sys.stderr)

SENT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9])')
# leading abstract section headers glued onto the sentence ("Objective:", "Background and Aim:")
SECTION = re.compile(r'^(?:objectives?|background|conclusions?|results?|methods?|introduction|aims?|'
                     r'purpose|summary|abstract|importance|findings?|rationale|significance)'
                     r'(?:\s+(?:and|&)\s+\w+)?\s*[:.—-]\s+', re.I)
def split_sentences(t):
    out = []
    for para in t.split("\n"):
        for s in SENT_RE.split(para):
            s = s.strip()
            if s:
                out.append(s)
    return out

# --------------------------------------------------------------------------- #
# shared lexical pieces
# --------------------------------------------------------------------------- #
ANAPHORIC = {
    "this", "that", "these", "those", "it", "they", "there", "here", "such", "we",
    "our", "their", "he", "she", "its", "his", "her", "in", "as", "when", "while",
    "although", "however", "thus", "therefore", "because", "since", "if", "for",
    "both", "most", "many", "some", "each", "one", "first", "second", "third",
    "also", "where", "after", "before", "during", "by", "with", "to", "from",
    "additionally", "moreover", "furthermore", "notably", "importantly", "interestingly",
}
PRONOUN = {"it", "they", "this", "that", "these", "those", "he", "she", "there", "we", "one"}
BAD = re.compile(r'\b(et al|Fig\.|Table\s*\d|http)\b', re.I)
GENERIC_SUBJ = re.compile(
    r'^The\s+(disease|virus|infection|condition|syndrome|disorder|illness|process|reaction|'
    r'response|complex|protein|gene|enzyme|cells?|patients?|symptoms?|results?|study|model|'
    r'pathogen|bacteri(?:um|a)|fungus|parasite|tumou?r|cancer|lesion)\s+(?:is|are|was|were)\b', re.I)
ACRONYM = re.compile(r'\([A-Z][A-Za-z0-9\-]{1,7}\)')                 # e.g. (RV), (HCV)
CATEGORY = re.compile(r'^\W*[A-Z][\w\-\(\)/ ]{1,45}?\bis an?\b', re.I)  # "X is a/an <category> ..."
OPER = re.compile(r'(^\s*\W*\d)|[<>]|≥|≤|\bMIC\b|cut[- ]?off|endpoint|'
                  r'positive (?:test|result)|the (?:presence|absence) of', re.I)


# --------------------------------------------------------------------------- #
# strict scorer (original behaviour)
# --------------------------------------------------------------------------- #
CHAR = re.compile(r'characteri[sz]ed\s+by', re.I)
PRES_CHAR = re.compile(r'\b(?:is|are)\s+characteri[sz]ed\s+by', re.I)
DEF = re.compile(r'\b(?:is|are|can\s+be|may\s+be)\s+defined\s+as\b', re.I)

def score_strict(s):
    mc, md = CHAR.search(s), DEF.search(s)
    if mc and (not md or mc.start() <= md.start()):
        m, is_char = mc, True
    elif md:
        m, is_char = md, False
    else:
        return None
    if is_char and not (CATEGORY.search(s) or PRES_CHAR.search(s)):
        return None
    nwords = len(s.split())
    if nwords < 8 or nwords > 55:
        return None
    if s.rstrip()[-1] not in '.!?]")\'':
        return None
    if re.search(r'\b(?:it|they)\s*$', s[:m.start()], re.I):
        return None
    fw = re.match(r"[A-Za-z][\w\-]*", s)
    if not fw or not s[0].isupper() or fw.group(0).lower() in ANAPHORIC:
        return None
    if GENERIC_SUBJ.match(s):
        return None
    subj = s[:m.start()].split()
    if not (1 <= len(subj) <= 12):
        return None
    pred = s[m.end():].split()
    if len(pred) < 4:
        return None
    sc = 3.0
    head = s[:55]
    if ACRONYM.search(head): sc += 2.0
    if CATEGORY.search(s): sc += 1.5
    if re.search(r'[A-Z]{2,}|\d', " ".join(pred)): sc += 1.0
    if is_char: sc += 0.3
    elif OPER.search(" ".join(pred[:7])): sc -= 2.0
    if s.rstrip().endswith('.'): sc += 0.3
    if BAD.search(s): sc -= 1.0
    if '?' in s: sc -= 2.0
    return sc


# --------------------------------------------------------------------------- #
# broad scorer: many definitional cue families + acronym introduction
# --------------------------------------------------------------------------- #
# (name, pattern, base weight). Strong/explicit cues weigh more than vague ones.
CUE_SPECS = [
    ("defas",   re.compile(r'\b(?:is|are|was|were|can\s+be|may\s+be)\s+defined\s+as\b', re.I), 3.0),
    ("charby",  re.compile(r'\bcharacteri[sz]ed\s+by\b', re.I), 3.0),
    ("knownas", re.compile(r'\b(?:is|are|is\s+also|are\s+also)\s+(?:known|referred\s+to|called|termed)\s+as\b', re.I), 3.0),
    ("refers",  re.compile(r'\brefers?\s+to\b', re.I), 2.8),
    ("wedef",   re.compile(r'\bwe\s+(?:define|refer\s+to|call|denote|term)\b', re.I), 2.8),
    ("expras",  re.compile(r'\b(?:is|are|can\s+be|may\s+be)\s+(?:expressed|written|computed|calculated|given|obtained|estimated|measured|quantified|described)\s+as\b', re.I), 2.5),
    ("isa",     re.compile(r'\b(?:is|are)\s+an?\s+[A-Za-z][\w-]*(?:\s+[A-Za-z][\w-]*){0,2}\s+(?:that|which|in\s+which|used|consisting|comprising|defined|characteri[sz]ed|with|of|for)\b', re.I), 2.5),
    ("isthe",   re.compile(r'\b(?:is|are)\s+the\s+(?:process|measure|ratio|rate|amount|number|fraction|proportion|probability|degree|sum|product|study|set|phenomenon|property|ability|tendency|condition|state|distance|energy|likelihood|extent|tool|technique|method|model)\s+(?:of|by|that|in|at|whereby|to\s+which)\b', re.I), 2.2),
    ("letdenote", re.compile(r'\blet\s+[A-Za-z][\w\-]*\s+(?:denote|be|represent)\b', re.I), 2.2),
    ("consist", re.compile(r'\b(?:consists?\s+of|comprises?\b|(?:is|are)\s+composed\s+of|(?:is|are)\s+made\s+up\s+of)\b', re.I), 1.8),
    ("denote",  re.compile(r'\bdenote[sd]?\b|\bdenoted\s+(?:by|as)\b', re.I), 1.8),
    ("repr",    re.compile(r'\b(?:represents?|describes?|corresponds?\s+to)\b', re.I), 1.4),
    ("means",   re.compile(r'\bmeans?\b', re.I), 1.2),
]
AUTHOR = {"wedef", "letdenote"}     # author-framed: subject is not sentence-initial
WEAK = {"repr", "means", "denote", "consist"}  # need a term-like subject to count
# cheap one-shot pre-screen so we don't run all cue regexes on every sentence
HINT = re.compile(r'defined|characteri[sz]|refer|known\s+as|called|termed|denote|represent|'
                  r'describe|consist|comprise|composed|made\s+up|\bmeans?\b|corresponds|'
                  r'expressed|computed|calculated|given\s+as|written\s+as|obtained\s+as|'
                  r'\bis\s+an?\b|\bare\s+an?\b|\bis\s+the\b|\([A-Z]', re.I)

STOP = {"of", "the", "and", "for", "in", "to", "a", "an", "on", "with", "by", "or"}
ACRO_PAREN = re.compile(r'\(([A-Za-z][A-Za-z0-9\-]{1,6})\)')

def acronym_intro(s):
    """True if a parenthetical acronym's letters subsequence-match the initials of the
    preceding content words -- a strong sign a named term is being introduced."""
    for m in ACRO_PAREN.finditer(s):
        ac = m.group(1)
        letters = [c.lower() for c in ac if c.isalpha()]
        if len(letters) < 2 or not any(c.isupper() for c in ac):
            continue
        pre = re.findall(r"[A-Za-z][\w'\-]*", s[:m.start()])[-3 * len(letters):]
        inits = iter(w[0].lower() for w in pre if w.lower() not in STOP)
        if all(c in inits for c in letters):     # acl is a subsequence of the initials
            return True
    return False

def _best_cue(s):
    best = None
    for name, pat, w in CUE_SPECS:
        m = pat.search(s)
        if m and (best is None or w > best[2]):
            best = (name, m, w)
    return best

def score_broad(s):
    cue = _best_cue(s)
    acro = acronym_intro(s)
    if cue is None and not acro:
        return None
    nwords = len(s.split())
    if nwords < 8 or nwords > 60:
        return None
    if s.rstrip()[-1] not in '.!?]")\'':          # drop chunk-boundary fragments
        return None
    fw = re.match(r"[A-Za-z][\w\-]*", s)
    if not fw or not s[0].isupper():
        return None

    name, m, w = cue if cue else ("acro", None, 2.0)
    head = s[:60]
    pred = s[m.end():] if m else s
    sc = w
    if ACRONYM.search(head): sc += 2.0          # named concept w/ acronym
    if CATEGORY.search(s): sc += 1.0
    if re.search(r'[A-Z]{2,}|\d', pred): sc += 1.0   # specific predicate
    if s.rstrip().endswith('.'): sc += 0.3

    first = fw.group(0).lower()
    if name not in AUTHOR:                        # subject is sentence-initial -> judge it
        if first in PRONOUN: sc -= 2.5            # "It refers to ...", "This is the ..."
        elif first in ANAPHORIC: sc -= 1.2
        if GENERIC_SUBJ.match(s): sc -= 1.5
        if m:
            subj_len = len(s[:m.start()].split())
            if not (1 <= subj_len <= 14): sc -= 1.0   # subject implausibly long/empty
    # vague cues only count when the subject looks like a named term
    if name in WEAK and not (ACRONYM.search(head) or CATEGORY.search(s)):
        sc -= 1.0
    if OPER.search(" ".join(pred.split()[:7])): sc -= 0.5   # operational/threshold "defined as 3.5"
    if BAD.search(s): sc -= 1.0
    if '?' in s: sc -= 2.0
    return sc


SCORER = score_strict if args.recall == "strict" else score_broad

# --------------------------------------------------------------------------- #
# scan
# --------------------------------------------------------------------------- #
scored = []
nsent = 0
for ci in starts:
    ids = [t for t in arr[ci*CHUNK:(ci+1)*CHUNK].astype(np.int64).tolist() if t != EOS]
    for x in split_sentences(tok.decode(ids)):
        s = SECTION.sub('', x).strip()
        if not s or not HINT.search(s):
            continue
        nsent += 1
        if not has_keyword(s):
            continue
        sc = SCORER(s)
        if sc is not None:
            scored.append((sc, s))
print(f"cue-bearing sentences: {nsent:,} | keyword filter: {_KW or 'NONE'} | matched: {len(scored):,}",
      file=sys.stderr)

# dedup (by lowercased prefix), keep highest score
seen, uniq = set(), []
for sc, s in sorted(scored, key=lambda x: -x[0]):
    k = s.lower()[:80]
    if k in seen:
        continue
    seen.add(k)
    uniq.append((sc, s))

limit = len(uniq) if N_OUT is None else min(N_OUT, len(uniq))
print(f"matched: {len(scored):,}  unique: {len(uniq):,}  -> writing {limit}", file=sys.stderr)

with open(TSV, "w") as f:
    for sc, s in uniq:
        f.write(f"{sc:.1f}\t{s}\n")
with open(OUT, "w") as f:
    for _, s in uniq[:limit]:
        f.write(s + "\n")
print(f"wrote {OUT} and {TSV}", file=sys.stderr)

print("\n===== TOP 15 =====")
for sc, s in uniq[:15]:
    print(f"[{sc:.1f}] {s}\n")
print("===== AROUND RANK 2500 =====")
for sc, s in uniq[2498:2503]:
    print(f"[{sc:.1f}] {s}\n")
print("===== AROUND RANK 5000 (likely cutoff) =====")
for sc, s in uniq[4998:5003]:
    print(f"[{sc:.1f}] {s}\n")
