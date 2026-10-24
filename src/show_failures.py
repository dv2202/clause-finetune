"""Print failures from one eval run, grouped by a first-guess error type.

Usage: uv run python src/show_failures.py results/B1_few_shot.json [--n 10]
"""

import argparse
import json
from collections import Counter

from eval import SPAN_MATCH_THRESHOLD


def error_type(p: dict) -> str | None:
    """Mechanical bucket for a prediction; None = correct."""
    if not p["valid_json"]:
        return "invalid_json"
    if p["hallucinated"]:
        return "hallucinated_text"          # text not in the excerpt at all
    if not p["gold_found"] and p["pred_found"]:
        return "false_positive"             # claimed a clause that isn't labelled
    if p["gold_found"] and not p["pred_found"]:
        return "missed_clause"              # false negative
    if p["gold_found"] and p["span_f1"] < SPAN_MATCH_THRESHOLD:
        return "wrong_or_partial_span"      # found it, copied the wrong part
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()
    preds = json.load(open(args.run))["predictions"]
    typed = [(error_type(p), p) for p in preds]
    fails = [(t, p) for t, p in typed if t]
    print(f"{len(fails)} failures out of {len(preds)}:", dict(Counter(t for t, _ in fails)), "\n")
    # Round-robin over error types so the sample shows every kind, not just the most common one.
    by_type = {}
    for t, p in fails:
        by_type.setdefault(t, []).append(p)
    shown, k = 0, 0
    while shown < args.n and any(k < len(v) for v in by_type.values()):
        for t, ps in by_type.items():
            if k < len(ps) and shown < args.n:
                p = ps[k]
                shown += 1
                print(f"--- #{shown} [{t}] {p['clause_type']} | gold_found={p['gold_found']} hard_neg={p['hard']} | span_f1={p['span_f1']:.2f}")
                print(f"GOLD: {(p['gold_text'] or 'null')[:300]}")
                print(f"RAW : {p['raw_output'][:300]}\n")
        k += 1


if __name__ == "__main__":
    main()
