"""Phase 5 forgetting check: does the fine-tuned model still handle normal requests?

Usage: uv run python src/general_check.py --adapter adapters/final_it300
Runs 5 general prompts (not our task, not our system prompt) on the base model and on
base + adapter, and saves both answers side by side. This is read by a human, not scored.
"""

import argparse
import json
import time
from pathlib import Path

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from prompts import MODEL_ID

PROMPTS = [
    "Summarise this paragraph in one sentence: The supplier shall deliver the goods within 30 days "
    "of the purchase order. If delivery is late, the buyer may cancel the order and receive a full refund.",
    "What is the capital of Japan? Answer in one word.",
    "Write a short polite email asking a colleague to review my report by Friday.",
    "Explain what a non-disclosure agreement is to a 10-year-old, in 2 sentences.",
    "List three fruits that are red.",
]


def answer(model, tokenizer, prompt: str) -> str:
    # Plain user turn, no ClauseFinder system prompt: we want the model's general behaviour.
    text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                         add_generation_prompt=True, tokenize=False)
    return generate(model, tokenizer, prompt=text, max_tokens=150, sampler=make_sampler(temp=0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    args = ap.parse_args()

    rows = [{"prompt": p} for p in PROMPTS]
    for key, adapter in [("base", None), ("finetuned", args.adapter)]:
        model, tokenizer = load(MODEL_ID, adapter_path=adapter)
        for row in rows:
            row[key] = answer(model, tokenizer, row["prompt"])
            print(f"\n=== [{key}] {row['prompt'][:70]}\n{row[key]}", flush=True)

    out = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "adapter": args.adapter, "rows": rows}
    Path("results/phase5_general_check.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("\nsaved results/phase5_general_check.json")


if __name__ == "__main__":
    main()
