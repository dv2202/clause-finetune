"""Phase 1: turn CUAD into chat-format JSONL for fine-tuning + eval.

Pipeline: load CUAD -> normalise whitespace (keep an offset map) -> split
contracts into train/valid/test BY TITLE -> chunk each contract -> label chunks
per clause type -> sample positives + (hard) negatives -> write JSONL.
"""

import hashlib
import json
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from prompts import CLAUSE_TYPES, MODEL_ID, build_training_example

RAW = Path("data/raw/CUADv1.json")
OUT = Path("data/processed")
SEED = 42

# Chunk size in words. 400-600 words keeps each example ~600-900 tokens (fast training).
TARGET_WORDS, MAX_WORDS = 400, 600
# Overlap = the last paragraph of the previous chunk, but only if it's short
# (a huge overlap would just duplicate data).
MAX_OVERLAP_WORDS = 150

# Examples per clause type, per split: (positives, negatives). ~1:1 ratio.
# 5 types x (100+100) = 1000 train; 5 x (15+15) = 150 valid; 5 x (20+20) = 200 test.
PER_TYPE = {"train": (100, 100), "valid": (15, 15), "test": (20, 20)}
GOLDEN_PER_TYPE = (6, 6)  # 5 x 12 = 60 hand-check examples, a balanced subset of test.jsonl
HARD_NEG_SHARE = 0.5  # aim for half hard negatives (spec minimum is 1/3)
# CUAD sometimes highlights tiny fragments ("You must:", a bare heading). Training on
# them teaches incomplete spans, so a chunk whose only gold span is < 8 words is skipped.
MIN_POS_WORDS = 8
# Hand-reviewed bad labels (Phase 1 checkpoint), keyed by (title, clause_type, chunk_id).
OVERRIDES = Path("data/label_overrides.json")
MAX_POS_PER_CONTRACT = 2  # spread positives over many contracts, so no single contract dominates

# Words that make a chunk "look like" the clause. A negative chunk containing
# them is a hard negative: it forces the model to read meaning, not keywords.
HARD_NEG_KEYWORDS = {
    "Governing Law": ["governed", "governing", "laws of", "jurisdiction", "construed"],
    "Termination For Convenience": ["terminate", "termination"],
    "Non-Compete": ["compete", "competitive", "competing", "competitor"],
    "Cap On Liability": ["liability", "liable", "damages"],
    "Anti-Assignment": ["assign", "assignment", "transfer", "successors"],
}


# ---------- 1. whitespace normalisation with an offset map ----------

def normalise(text: str) -> tuple[str, list[int]]:
    """Collapse messy whitespace; return new text + map old_index -> new_index.

    CUAD text has runs like "laws          of". Collapsing them saves tokens and
    makes "copy the clause exactly" learnable. The map lets us move gold spans
    (character offsets in the ORIGINAL text) into the new text.
    Rules: spaces/tabs runs -> 1 space; spaces around newlines dropped;
    3+ newlines -> 2 (a blank line = paragraph break).
    """
    out, old_to_new = [], []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch in " \t\r\n\f\v ":
            j = i
            while j < n and text[j] in " \t\r\n\f\v ":
                j += 1
            newlines = text[i:j].count("\n")
            sep = "\n\n" if newlines >= 2 else ("\n" if newlines == 1 else " ")
            if not out:  # drop leading whitespace
                sep = ""
            start = len(out)
            out.extend(sep)
            # every old char in this whitespace run maps to the start of the separator
            old_to_new.extend([start] * (j - i))
            i = j
        else:
            old_to_new.append(len(out))
            out.append(ch)
            i += 1
    old_to_new.append(len(out))  # sentinel so an end offset == len(text) maps too
    return "".join(out).rstrip(), old_to_new


def norm_span(s: str) -> str:
    """Same whitespace rule applied to a standalone answer string (for verification)."""
    s = re.sub(r"[ \t\r\f\v ]*\n[ \t\r\f\v ]*", "\n", s.strip())
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"\n\s*\n", "\n\n", s)
    return re.sub(r"[ \t\r\f\v ]+", " ", s)


# ---------- 2. paragraph-aware chunking ----------

def paragraphs(text: str) -> list[tuple[int, int]]:
    """Split on blank lines; break any paragraph > MAX_WORDS on single newlines,
    then on sentence ends, then hard by words. Returns (start, end) char offsets."""
    spans = [(m.start(), m.end()) for m in re.finditer(r"[^\n](?:.|\n(?!\n))*", text)]
    result = []
    for s, e in spans:
        result.extend(_split_long(text, s, e))
    return result


SEPARATORS = (r"\n", r"(?<=[.;:])\s", r" ")  # line breaks, then sentence ends, then any space


def _split_long(text: str, s: int, e: int, seps=SEPARATORS) -> list[tuple[int, int]]:
    if len(text[s:e].split()) <= MAX_WORDS or not seps:
        return [(s, e)]
    cuts = [s + m.end() for m in re.finditer(seps[0], text[s:e])]
    if not cuts:
        return _split_long(text, s, e, seps[1:])  # separator absent: try a finer one
    bounds = [s] + cuts + [e]
    # Greedily merge the small pieces back into blocks of ~TARGET_WORDS.
    blocks, cur = [], s
    for a, b in zip(bounds, bounds[1:]):
        if a > cur and len(text[cur:b].split()) > TARGET_WORDS:
            blocks.append((cur, a))
            cur = a
    blocks.append((cur, e))
    # Any block still too long gets split with the next, finer separator.
    return [p for a, b in blocks for p in _split_long(text, a, b, seps[1:])]


def chunk(text: str) -> list[tuple[int, int]]:
    """Merge paragraphs until ~TARGET_WORDS; overlap one short paragraph."""
    paras = paragraphs(text)
    words = [len(text[s:e].split()) for s, e in paras]
    chunks, i = [], 0
    while i < len(paras):
        j, total = i, 0
        # add paragraphs while under target and not exceeding MAX_WORDS
        while j < len(paras) and (total < TARGET_WORDS) and (total + words[j] <= MAX_WORDS or j == i):
            total += words[j]
            j += 1
        chunks.append((paras[i][0], paras[j - 1][1]))
        if j >= len(paras):
            break
        # overlap: next chunk starts at the last paragraph if it is short, and we still move forward
        i = j - 1 if (j - 1 > i and words[j - 1] <= MAX_OVERLAP_WORDS) else j
    return chunks


# ---------- 3. labelling ----------

def label_chunk(cs: int, ce: int, gold: list[tuple[int, int]]):
    """'pos' (longest fully-inside span), 'skip' (a span is cut by the boundary), or 'neg'."""
    inside = [(s, e) for s, e in gold if cs <= s and e <= ce]
    cut = [(s, e) for s, e in gold if s < ce and e > cs and not (cs <= s and e <= ce)]
    if inside:
        return "pos", max(inside, key=lambda p: p[1] - p[0])
    if cut:
        return "skip", None  # never create a wrong label
    return "neg", None


def is_hard(chunk_text: str, clause_type: str) -> bool:
    low = chunk_text.lower()
    return any(k in low for k in HARD_NEG_KEYWORDS[clause_type])


# ---------- 4. main ----------

def build_candidates(contract: dict, dropped: set = frozenset()) -> list[dict]:
    """All labelled (chunk, clause_type) candidates for one contract."""
    raw_ctx = contract["paragraphs"][0]["context"]
    ctx, m = normalise(raw_ctx)
    qas = {q["id"].split("__")[-1]: q for q in contract["paragraphs"][0]["qas"]}
    chunks = chunk(ctx)
    cands = []
    for ctype in CLAUSE_TYPES:
        gold = []
        for a in qas[ctype]["answers"]:
            s0, e0 = a["answer_start"], a["answer_start"] + len(a["text"])
            s, e = m[s0], m[e0]
            # trim whitespace the mapping may have pulled in at the edges
            while s < e and ctx[s].isspace():
                s += 1
            while e > s and ctx[e - 1].isspace():
                e -= 1
            assert ctx[s:e] == norm_span(a["text"]), (contract["title"], ctype, ctx[s:e][:80], norm_span(a["text"])[:80])
            gold.append((s, e))
        has_clause_anywhere = bool(gold)
        for k, (cs, ce) in enumerate(chunks):
            lab, span = label_chunk(cs, ce, gold)
            if lab == "skip":
                continue
            if lab == "pos" and len(ctx[span[0]:span[1]].split()) < MIN_POS_WORDS:
                continue  # fragment label: skip rather than teach an incomplete span
            if (contract["title"], ctype, k) in dropped:
                continue  # hand-reviewed bad label
            excerpt = ctx[cs:ce]
            cands.append({
                "title": contract["title"], "clause_type": ctype, "chunk_id": k,
                "chunk_start": cs, "chunk_end": ce, "excerpt": excerpt,
                "label": lab,
                "text": ctx[span[0]:span[1]] if span else None,
                "hard": lab == "neg" and is_hard(excerpt, ctype),
                "contract_has_clause": has_clause_anywhere,
            })
    return cands


def priority(c: dict) -> str:
    """Stable random order: a hash of the item itself, not a shared RNG stream.
    Dropping one bad label then doesn't reshuffle every other pick (reviews stay valid)."""
    return hashlib.sha1(f"{SEED}|{c['title']}|{c['clause_type']}|{c['chunk_id']}".encode()).hexdigest()


def sample(cands: list[dict], n_pos: int, n_neg: int):
    """Pick positives (max per contract) then negatives from the SAME contracts, ~half hard."""
    pool = sorted(cands, key=priority)
    pos_all = [c for c in pool if c["label"] == "pos"]
    per_contract, pos = Counter(), []
    for c in pos_all:
        if len(pos) < n_pos and per_contract[c["title"]] < MAX_POS_PER_CONTRACT:
            pos.append(c)
            per_contract[c["title"]] += 1
    # negatives come from the same contracts as the positives (spec), same clause type
    titles = {c["title"] for c in pos}
    neg_pool = [c for c in pool if c["label"] == "neg" and c["title"] in titles]
    hard = [c for c in neg_pool if c["hard"]]
    easy = [c for c in neg_pool if not c["hard"]]
    n_hard = min(len(hard), round(n_neg * HARD_NEG_SHARE))
    neg = hard[:n_hard] + easy[: n_neg - n_hard]
    if len(neg) < n_neg:  # not enough easy ones: top up with more hard ones
        neg += hard[n_hard : n_hard + n_neg - len(neg)]
    return pos, neg


def to_record(c: dict) -> dict:
    rec = build_training_example(c["clause_type"], c["excerpt"], c["text"])
    # meta is ignored by mlx-lm (it only reads "messages"); we use it for eval + human review
    rec["meta"] = {k: c[k] for k in ("title", "clause_type", "chunk_id", "chunk_start", "chunk_end", "label", "hard")}
    return rec


def write_jsonl(path: Path, records: list[dict]):
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    rng = random.Random(SEED)
    data = json.loads(RAW.read_text())["data"]

    # Split by contract TITLE first, so a contract's chunks can never leak across splits.
    titles = sorted(c["title"] for c in data)
    rng.shuffle(titles)
    n = len(titles)
    split_of = {}
    for i, t in enumerate(titles):
        split_of[t] = "train" if i < 0.8 * n else ("valid" if i < 0.9 * n else "test")
    sets = {s: {t for t, v in split_of.items() if v == s} for s in ("train", "valid", "test")}
    assert not (sets["train"] & sets["valid"] or sets["train"] & sets["test"] or sets["valid"] & sets["test"])

    # Build candidates per split
    cands = defaultdict(list)
    chunk_words = []
    dropped = {(o["title"], o["clause_type"], o["chunk_id"]) for o in json.loads(OVERRIDES.read_text())}
    print(f"label overrides: {len(dropped)} (title, clause_type, chunk) items dropped; positives < {MIN_POS_WORDS} words skipped")
    for c in data:
        cs = build_candidates(c, dropped)
        cands[split_of[c["title"]]].extend(cs)
        chunk_words += [len(x["excerpt"].split()) for x in cs if x["clause_type"] == CLAUSE_TYPES[0]]

    print(f"contracts: {n}  split sizes: " + ", ".join(f"{s}={len(v)}" for s, v in sets.items()))
    print(f"chunk words: p5={sorted(chunk_words)[len(chunk_words)//20]} median={statistics.median(chunk_words):.0f} "
          f"p95={sorted(chunk_words)[len(chunk_words)*19//20]} max={max(chunk_words)}  (n chunks={len(chunk_words)})")
    print("available candidates (pos / neg / hard-neg) per split and type:")
    for s in ("train", "valid", "test"):
        for t in CLAUSE_TYPES:
            cc = [x for x in cands[s] if x["clause_type"] == t]
            print(f"  {s:5s} {t:28s} pos={sum(x['label']=='pos' for x in cc):4d} "
                  f"neg={sum(x['label']=='neg' for x in cc):5d} hard={sum(x['hard'] for x in cc):5d}")

    # Sample examples
    out = {"train": [], "valid": [], "test": [], "golden": []}
    for s in ("train", "valid", "test"):
        for t in CLAUSE_TYPES:
            pool = [x for x in cands[s] if x["clause_type"] == t]
            p, q = sample(pool, *PER_TYPE[s])
            out[s] += p + q
            if s == "test":
                # golden = balanced SUBSET of test, so hand-fixed labels apply to test too
                # (test has too few Non-Compete positives to carve out a disjoint set)
                half = GOLDEN_PER_TYPE[1] // 2  # same hard/easy mix as test (q lists hard ones first)
                out["golden"] += p[: GOLDEN_PER_TYPE[0]] + q[:half] + q[-(GOLDEN_PER_TYPE[1] - half):]
        rng.shuffle(out[s])
    rng.shuffle(out["golden"])

    # Verify: every positive's text appears exactly in its excerpt; no title leakage.
    for s, recs in out.items():
        for c in recs:
            if c["label"] == "pos":
                assert c["text"] in c["excerpt"], (s, c["title"], c["clause_type"])
    split_titles = {s: {c["title"] for c in recs} for s, recs in out.items()}
    assert not (split_titles["train"] & (split_titles["valid"] | split_titles["test"] | split_titles["golden"]))
    assert not (split_titles["valid"] & (split_titles["test"] | split_titles["golden"]))
    print("checks passed: exact-span containment, no contract overlap across splits")

    # Write files
    OUT.mkdir(parents=True, exist_ok=True)
    for s, recs in out.items():
        write_jsonl(OUT / f"{s}.jsonl", [to_record(c) for c in recs])

    # Stats: counts, pos/neg ratio, hard share, token lengths
    from transformers import AutoTokenizer  # tokenizer only; avoids loading the model weights
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    stats = {}
    print("\nfinal counts:")
    for s, recs in out.items():
        per_type = Counter((c["clause_type"], c["label"]) for c in recs)
        n_pos = sum(c["label"] == "pos" for c in recs)
        n_neg = len(recs) - n_pos
        n_hard = sum(c["hard"] for c in recs)
        # return_dict=False: transformers 5 otherwise returns a dict, and len() would count its keys
        lens = sorted(len(tok.apply_chat_template(to_record(c)["messages"], tokenize=True, return_dict=False)) for c in recs)
        pct = {p: lens[min(len(lens) - 1, int(len(lens) * p / 100))] for p in (50, 90, 95, 99)} | {"max": lens[-1]}
        print(f"  {s:6s} n={len(recs):4d} pos={n_pos} neg={n_neg} hard_neg={n_hard} ({n_hard / max(n_neg, 1):.0%} of neg) "
              f"contracts={len(split_titles[s])} tokens p50={pct[50]} p90={pct[90]} p95={pct[95]} p99={pct[99]} max={pct['max']}")
        for t in CLAUSE_TYPES:
            print(f"         {t:28s} pos={per_type[(t, 'pos')]:3d} neg={per_type[(t, 'neg')]:3d}")
        stats[s] = {"n": len(recs), "pos": n_pos, "neg": n_neg, "hard_neg": n_hard, "contracts": len(split_titles[s]),
                    "token_percentiles": pct,
                    "per_type": {t: {"pos": per_type[(t, "pos")], "neg": per_type[(t, "neg")]} for t in CLAUSE_TYPES}}
    Path("results").mkdir(exist_ok=True)
    Path("results/data_stats.json").write_text(json.dumps(stats, indent=2))
    print("\nwrote data/processed/{train,valid,test,golden}.jsonl and results/data_stats.json")


if __name__ == "__main__":
    main()
