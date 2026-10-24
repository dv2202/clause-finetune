"""Trace one prediction back to its sources, so every metric can be checked by hand.

Usage: uv run python src/inspect_pred.py results/B1_few_shot.json 4
Shows: the excerpt the model saw, the gold label, what CUAD's experts marked for this
contract + clause type (from the raw file), and re-runs each check from eval.py.
"""

import json
import sys

from eval import token_f1, ws
from prepare_data import RAW, norm_span
from prompts import parse_excerpt


def main():
    run_path, idx = sys.argv[1], int(sys.argv[2])
    run = json.load(open(run_path))
    p = run["predictions"][idx]
    record = json.loads(open(f"data/processed/{run['config']['split']}.jsonl").readlines()[idx])
    excerpt = parse_excerpt(record["messages"][1]["content"])
    meta = record["meta"]

    print(f"=== {run_path} | idx {idx} | {p['clause_type']} | {meta['title']} | chunk {meta['chunk_id']} "
          f"(chars {meta['chunk_start']}-{meta['chunk_end']}) | hard_neg={meta['hard']}\n")
    print("--- EXCERPT THE MODEL SAW ---\n" + excerpt + "\n")

    # What CUAD's lawyers marked for this contract + clause type, anywhere in the contract.
    contract = next(c for c in json.loads(RAW.read_text())["data"] if c["title"] == meta["title"])
    qa = next(q for q in contract["paragraphs"][0]["qas"] if q["id"].endswith("__" + p["clause_type"]))
    print(f"--- CUAD EXPERT SPANS for '{p['clause_type']}' in this whole contract: {len(qa['answers'])} ---")
    for a in qa["answers"]:
        inside = norm_span(a["text"]) in excerpt
        print(f"  [{'INSIDE this chunk' if inside else 'elsewhere in contract'}] {a['text'][:200]!r}")
    print()

    print(f"--- GOLD (our label) ---\nfound={p['gold_found']}  text={p['gold_text']!r}\n")
    print(f"--- MODEL RAW OUTPUT ---\n{p['raw_output']}\n")

    # Re-run the eval checks by hand.
    print("--- CHECKS (recomputed) ---")
    print(f"valid_json      : {p['valid_json']}")
    print(f"pred_found      : {p['pred_found']}  vs gold_found {p['gold_found']}  -> "
          + {(True, True): "TRUE POSITIVE", (False, True): "FALSE POSITIVE",
             (True, False): "MISSED (false negative)", (False, False): "TRUE NEGATIVE"}[(p["gold_found"], p["pred_found"])])
    if p["pred_text"]:
        in_excerpt = ws(p["pred_text"]) in ws(excerpt)
        print(f"text in excerpt : {in_excerpt}  -> hallucinated={not in_excerpt}")
        if p["gold_text"]:
            print(f"span_f1         : {token_f1(p['pred_text'], p['gold_text']):.2f} (match if >= 0.5)")


if __name__ == "__main__":
    main()
