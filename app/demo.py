"""ClauseFinder CLI demo: contract text in, verified clause JSON out.

Usage:
  uv run python app/demo.py app/samples/tuniu_cooperation_agreement.txt
  uv run python app/demo.py my_contract.txt --top-k 0          # check every chunk (slow, max recall)
  uv run python app/demo.py my_contract.txt --out results/demo_output.json

Pipeline: normalise -> chunk -> keyword retrieval (top-k chunks per clause type)
          -> fine-tuned model -> verify guard -> report.
"""

import argparse
import difflib
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))  # reuse the project's own code

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from eval import parse_output, ws
from prepare_data import HARD_NEG_KEYWORDS, chunk, normalise
from prompts import CLAUSE_TYPES, MODEL_ID, build_messages

REVIEW_THRESHOLD = 0.85  # word-level similarity needed to offer the closest real text for review


# ---------- retrieval: pick which chunks the model reads ----------

def keyword_score(text: str, clause_type: str) -> int:
    """Cheap first filter: how often the clause type's keywords appear in the chunk."""
    low = text.lower()
    return sum(low.count(k) for k in HARD_NEG_KEYWORDS[clause_type])


def pick_chunks(chunks: list[str], clause_type: str, top_k: int) -> list[int]:
    """Top-k chunk indices by keyword score; top_k=0 means every chunk (no retrieval)."""
    if top_k == 0:
        return list(range(len(chunks)))
    scored = [(keyword_score(c, clause_type), i) for i, c in enumerate(chunks)]
    scored = [(s, i) for s, i in scored if s > 0]  # no keyword at all -> very unlikely to hold the clause
    return [i for _, i in sorted(scored, reverse=True)[:top_k]]


# ---------- verify guard: never show text that isn't in the document ----------

def locate_exact(pred: str, doc: str) -> tuple[int, int] | None:
    """Find pred in doc, allowing any whitespace between words. Returns (start, end) in doc."""
    words = pred.split()
    if not words:
        return None
    m = re.search(r"\s+".join(re.escape(w) for w in words), doc)
    return (m.start(), m.end()) if m else None


def closest_real_text(pred: str, chunk_text: str) -> tuple[float, str]:
    """Slide a window of the same word length over the chunk; return the most similar real text."""
    p, c = pred.split(), chunk_text.split()
    n = min(len(p), len(c))
    best_ratio, best_start = 0.0, 0
    for s in range(0, len(c) - n + 1):
        r = difflib.SequenceMatcher(None, p, c[s:s + n], autojunk=False).ratio()
        if r > best_ratio:
            best_ratio, best_start = r, s
    return best_ratio, " ".join(c[best_start:best_start + n])


def verify(pred: str, chunk_text: str, chunk_start: int, doc: str) -> dict:
    """VERIFIED: exact copy (shown with its position). REVIEW: close, show the REAL text instead.
    BLOCKED: not close to anything in the chunk, show nothing."""
    span = locate_exact(pred, chunk_text)
    if span:
        s, e = chunk_start + span[0], chunk_start + span[1]
        return {"status": "VERIFIED", "text": doc[s:e], "char_start": s, "char_end": e}
    ratio, real = closest_real_text(pred, chunk_text)
    if ratio >= REVIEW_THRESHOLD:
        return {"status": "NEEDS_REVIEW", "text": real, "similarity": round(ratio, 3),
                "model_text": pred}
    return {"status": "BLOCKED", "text": None, "similarity": round(ratio, 3), "model_text": pred}


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("contract", help="path to a plain-text contract")
    ap.add_argument("--adapter", default="adapters/final_it300")
    ap.add_argument("--top-k", type=int, default=3, help="chunks per clause type; 0 = all chunks")
    ap.add_argument("--out", default=None, help="optional path to save the JSON report")
    args = ap.parse_args()

    doc, _ = normalise(Path(args.contract).read_text())  # same cleaning as the training data
    spans = chunk(doc)
    chunks = [doc[s:e] for s, e in spans]
    print(f"{args.contract}: {len(doc.split())} words -> {len(chunks)} chunks", flush=True)

    model, tokenizer = load(MODEL_ID, adapter_path=args.adapter)
    sampler = make_sampler(temp=0.0)

    report, calls, t0 = {}, 0, time.perf_counter()
    for clause_type in CLAUSE_TYPES:
        picked = pick_chunks(chunks, clause_type, args.top_k)
        hits = []
        for i in picked:
            prompt = tokenizer.apply_chat_template(build_messages(clause_type, chunks[i]),
                                                   add_generation_prompt=True, tokenize=False)
            raw = generate(model, tokenizer, prompt=prompt, max_tokens=800, sampler=sampler)
            calls += 1
            out = parse_output(raw)
            if out is None:
                hits.append({"chunk": i, "status": "INVALID_OUTPUT", "raw": raw[:200]})
            elif out.found:
                hits.append({"chunk": i, **verify(out.text, chunks[i], spans[i][0], doc)})
        report[clause_type] = {"found": any(h["status"] == "VERIFIED" for h in hits),
                               "chunks_checked": picked, "results": hits}
        print_type(clause_type, report[clause_type])

    secs = time.perf_counter() - t0
    print(f"\n{calls} model calls in {secs:.0f}s ({secs / max(calls, 1):.1f}s per call)")
    if args.out:
        Path(args.out).write_text(json.dumps({"contract": args.contract, "top_k": args.top_k,
                                              "adapter": args.adapter, "report": report},
                                             indent=2, ensure_ascii=False))
        print(f"saved {args.out}")


ICONS = {"VERIFIED": "[IN DOC]    ", "NEEDS_REVIEW": "[REVIEW]", "BLOCKED": "[BLOCKED]", "INVALID_OUTPUT": "[INVALID]"}


def print_type(clause_type: str, r: dict):
    print(f"\n=== {clause_type}  (read chunks {r['chunks_checked']})")
    if not r["results"]:
        print("  not found")
    for h in r["results"]:
        print(f"  {ICONS[h['status']]} chunk {h['chunk']}", end="")
        if h["status"] == "VERIFIED":
            print(f"  chars {h['char_start']}-{h['char_end']}\n    {h['text'][:300]}")
        elif h["status"] == "NEEDS_REVIEW":
            print(f"  similarity {h['similarity']}: model text was not an exact copy; closest REAL text:\n    {h['text'][:300]}")
        elif h["status"] == "BLOCKED":
            print(f"  similarity {h['similarity']}: model text not found in the document, not shown")
        else:
            print(f"  model output was not valid JSON")


if __name__ == "__main__":
    main()
