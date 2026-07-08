"""Curate the broad definition sets down to the N most textbook-general, context-
independent definitions per domain -- i.e. "terms of art" a smart student/professor
could answer WITHOUT paper-specific context (could plausibly appear in a textbook or
lit review), not study-specific findings.

Reads analysis/logp/<domain>_definitions_scored.tsv (definitional score + sentence),
re-scores by subtracting paper-specific penalties / adding generality bonuses, and
writes the top N to analysis/logp/<domain>_curated.txt.
"""
import re, sys

DOMAINS = ["biology", "cyber", "nuclear"]
N = 2000
HERE = "analysis/logp"

# --- paper-specific markers (penalized: a reader needs the paper to answer) ---
DEICTIC = re.compile(r'\b(?:we|our|us|herein|hereafter|this\s+(?:study|paper|work|article|report|'
                     r'analysis|experiment|section|approach|method|model|dataset|figure|table|equation)|'
                     r'the\s+present|the\s+proposed|the\s+current\s+(?:study|work|paper)|in\s+this|'
                     r'aforementioned|as\s+(?:shown|described|discussed|mentioned|defined)\s+(?:above|below|in)|'
                     r'our\s+results?)\b', re.I)
CITE = re.compile(r'\b(?:et\s+al|Fig\.?|Figure|Tab\.?|Table|Eq\.?|Eqn|Section|Ref\.?|Appendix)\b'
                  r'|\[\d|\(\s*\d{1,3}\s*\)|\bhttps?:|\bdoi\b', re.I)
# institutions / facilities / administrative bodies -- not textbook concepts.
# NOTE: no trailing \b -- these are prefix-roots (Universit -> University/Universities).
ORG = re.compile(r'\b(?:Universit|Laborator|Compan|Association|Institut|Corporation|Department|'
                 r'Agency|Administration|Foundation|Consortium|Office|Facilit|Cent(?:er|re)|Division|'
                 r'Bureau|Programme|Committee|Council|Authority|Directorate|Commission|'
                 r'Establishment|Headquarters|Hospital|Clinic)', re.I)
LABS = re.compile(r'\b(?:SNL|LANL|ORNL|LLNL|INL|INEEL|PNNL|BNL|ANL|DOE|NNSA|EPA|NRC|IAEA|NASA|DARPA|'
                  r'NIST|WIPP|Sandia|Livermore|Hanford|Oak\s?Ridge|Los\s?Alamos)\b')
# regulatory / dosimetric / standards language -- needs the regulation, not textbook art
REG = re.compile(r'\b(?:rems?|mrem|mSv|µSv|uSv|ACL|dose\s+(?:limit|constraint|equivalent)|'
                 r'administrative\s+control|control\s+level|per\s+plant-year|nano-?curie|micro-?curie|'
                 r'pCi|mR/h|exposure\s+to|recommends?|shall\s+not\s+exceed|standard\s+(?:recommends|states))\b', re.I)
METHODS = re.compile(r'\b(?:were|was|is|are)\s+(?:cultured|performed|used|measured|obtained|purchased|'
                     r'provided|prepared|conducted|isolated|incubated|grown|collected|supplied|seeded|'
                     r'maintained|harvested|administered|recorded|carried\s+out|kindly|implemented|'
                     r'deployed|evaluated|trained|tested|simulated|fabricated)\b', re.I)
# paper-internal procedural verbs (protocol descriptions, not definitions)
PROC = re.compile(r'\b(?:generates?|publishes?|encrypts?|decrypts?|transmits?|broadcasts?|sends?|'
                  r'computes?|allocates?|schedules?|assigns?|outputs?|returns?|stores?|forwards?|'
                  r'verifies|signs?)\b', re.I)
# fabrication / facility-build language (paper-specific apparatus, not a concept)
BUILT = re.compile(r'\b(?:was|were|is|are|been)\s+(?:built|constructed|installed|fabricated|'
                   r'manufactured|assembled|designed|operated|commissioned|located)\b', re.I)
# clearly off-domain computing/networking terms (penalized only for non-cyber domains)
FOREIGN_CS = re.compile(r'\b(?:LAN|WAN|VLAN|IoT|ICS|SCADA|firewall|malware|ransomware|botnet|phishing|'
                        r'blockchain|cryptograph|encryption|cipher|TCP|HTTP|packet|router|smartphone|'
                        r'Android|operating\s+system|machine\s+learning|deep\s+learning|neural\s+network|'
                        r'software|internet|cyber|local\s+area\s+network|wireless\s+network)\b', re.I)
# a real definitional cue (used to flag short cue-less fragments)
STRONG_CUE = re.compile(r'\b(?:defined\s+as|characteri[sz]ed\s+by|refers?\s+to|known\s+as|'
                        r'called|termed|is\s+an?\b|are\s+an?\b|denotes?|consists?\s+of)\b', re.I)
STUDY_SUBJ = re.compile(r'^\W*(?:the\s+)?(?:proposed|present|current|above|following|aforementioned|'
                        r'first|second|third|fourth|fifth|latter|former|same|resulting|observed|measured|'
                        r'reported|selected|chosen|corresponding|remaining|new|two|three|four)\b', re.I)
# enumerated / paper-structural openers and short-label "SR1 -" / "AT:" style
ENUM = re.compile(r'^\W*(?:Step|Phase|Stage|Case|Def(?:inition)?|Algorithm|Property|Theorem|Lemma|'
                  r'Proposition|Requirement|Req|SR|Rule|Axiom|Assumption|Round|Tier|Zone|Example|Note|'
                  r'Remark|Claim|Equation|Eq|Table|Figure|Fig|Section|Sec|Appendix)\b[\s\d:.\-]', re.I)
LABEL_DASH = re.compile(r'^\W*[A-Z][A-Za-z0-9]{0,5}\s*[-–—]\s')             # "SR1 - ...", "AT - ..."
# reference/citation lines: author (year), quoted titles, venue names
REFLINE = re.compile(r'\(\d{4}\)|"\s*[A-Z]|[“”]'
                     r'|\b(?:IEEE|ACM|Springer|Elsevier|arXiv|Proc\.|Proceedings|Journal|Trans\.|'
                     r'vol\.|pp\.|et\s+al)\b', re.I)
# operational / facility / safety-admin language (common in DOE-site nuclear docs)
ADMIN = re.compile(r'\b(?:Team|Personnel|Headquarters|Establishment|Operations|Mission|\bSite\b|Building|'
                   r'Storage\s+Area|Waste\s+(?:Management|Handling|Storage)|Rescue|Emergency\s+Response|'
                   r'Disposal\s+Area|responsible\s+for|comprised\s+of|operates?|was\s+developed|'
                   r'is\s+located|in\s+charge\s+of|established\s+(?:in|by|to)|tasked\s+with)\b', re.I)
GLOSS = re.compile(r'\([A-Z][A-Za-z0-9]{1,6}\)\s*[-–—]')        # "Full Name (ACR) - definition" glossary entry
LEGAL = re.compile(r'\bCFR\b|U\.?S\.?C\.?|Public\s+Law|\bStatute\b|§|\b[A-Z][a-z]+\s+Act\b|\bRCRA\b|'
                   r'\bFarm\s+Bill\b|Amendment|\bDirective\b', re.I)    # legal/policy documents
REL_ANAPH = re.compile(r'^\W*(?:the\s+)?(?:former|latter|it|they|this|that|these|those|here|such)\b', re.I)
NUM_DEF = re.compile(r'\bdefined\s+as\b[^.]{0,20}?[-+]?\d', re.I)          # "defined as 4.7%/3.5/..."
PCT_RESULT = re.compile(r'[-+]?\d+(?:\.\d+)?\s*%|p\s*[<>=]\s*0?\.\d|±', re.I)

# --- generality bonus: named concept defined as a recognizable category ---
CATEGORY = re.compile(r'\bis\s+an?\b[^.]{0,40}?\b(disease|disorder|syndrome|infection|virus|bacteri\w*|'
                      r'pathogen|protein|enzyme|receptor|gene|hormone|cell|process|technique|method|'
                      r'algorithm|protocol|framework|architecture|model|attack|scheme|metric|measure|'
                      r'mechanism|phenomenon|reaction|particle|reactor|isotope|nucleus|condition|'
                      r'system|structure|approach|paradigm|quantity|coefficient|parameter|distribution)s?\b', re.I)
ACRONYM = re.compile(r'\([A-Z][A-Za-z0-9\-]{1,7}\)')


def curate_score(base, s, domain):
    sc = base
    if BUILT.search(s):       sc -= 3.0     # "was built/constructed/installed ..." (apparatus/facility)
    if domain != "cyber" and FOREIGN_CS.search(s):
        sc -= 3.0                            # off-domain computing/networking term
    if len(s.split()) < 11 and not STRONG_CUE.search(s):
        sc -= 2.0                            # short cue-less fragment ("... (LAN) technology.")
    if DEICTIC.search(s):     sc -= 3.0
    if CITE.search(s):        sc -= 1.5
    if ORG.search(s):         sc -= 4.0     # institution / facility / admin body
    if LABS.search(s):        sc -= 4.0     # national lab / agency name
    if REG.search(s):         sc -= 3.0     # regulatory / dosimetric statement
    if METHODS.search(s):     sc -= 2.5
    if PROC.search(s):        sc -= 1.5     # paper-internal procedural description
    if STUDY_SUBJ.match(s):   sc -= 2.5
    if REL_ANAPH.match(s):    sc -= 3.0
    if ENUM.match(s):         sc -= 4.0     # "Step 2 -", "SR1", "Definition 4.1"
    if LABEL_DASH.match(s):   sc -= 3.0
    if REFLINE.search(s):     sc -= 3.5     # citation / reference line
    if ADMIN.search(s):       sc -= 3.0     # operational / facility / safety-admin
    if GLOSS.search(s):       sc -= 2.5     # "Name (ACR) - ..." glossary entry
    if LEGAL.search(s):       sc -= 2.5     # legal / policy document
    if NUM_DEF.search(s):     sc -= 1.5     # "X is defined as <number>" -> a study measurement
    if PCT_RESULT.search(s):  sc -= 0.8
    if CATEGORY.search(s):    sc += 1.0     # clean "X is a <category> ..." textbook form
    if ACRONYM.search(s[:60]):sc += 0.5     # named term w/ acronym
    if len(s.split()) > 50:   sc -= 0.5     # very long -> often study description
    return sc


def norm_key(s):
    return re.sub(r'[^a-z0-9]', '', s.lower())[:55]


for dom in DOMAINS:
    rows = []
    with open(f"{HERE}/{dom}_definitions_scored.tsv") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) != 2:
                continue
            base, s = float(parts[0]), parts[1]
            rows.append((curate_score(base, s, dom), s))
    rows.sort(key=lambda x: -x[0])
    seen, uniq = set(), []
    for sc, s in rows:
        k = norm_key(s)
        if k in seen:
            continue
        seen.add(k)
        uniq.append((sc, s))
    top = uniq[:N]
    with open(f"{HERE}/{dom}_curated.txt", "w") as f:
        for _, s in top:
            f.write(s + "\n")
    cut = top[-1][0] if top else 0
    print(f"{dom}: {len(rows):,} scored -> {len(uniq):,} unique -> wrote {len(top)} "
          f"(score {top[0][0]:.1f}..{cut:.1f})")
