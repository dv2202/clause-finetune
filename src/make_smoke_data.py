"""Tiny dataset for the Phase 3 smoke test: 100 train + 20 valid examples.

Why: before a 30-min run we want a 2-min run that proves memory fits, loss goes down,
and the adapter still outputs our JSON format. mlx-lm wants a folder with train/valid.jsonl.
"""

import json
import random
from pathlib import Path

OUT = Path("data/smoke")


def take(split: str, n: int, rng: random.Random) -> list[str]:
    lines = open(f"data/processed/{split}.jsonl").readlines()
    return rng.sample(lines, n)  # random, so all 5 clause types and pos/neg show up


def main():
    rng = random.Random(42)
    OUT.mkdir(parents=True, exist_ok=True)
    for split, n in [("train", 100), ("valid", 20)]:
        rows = take(split, n, rng)
        (OUT / f"{split}.jsonl").write_text("".join(rows))
        labels = [json.loads(r)["meta"]["label"] for r in rows]
        print(f"{split}: {n} examples, labels={ {l: labels.count(l) for l in set(labels)} }")


if __name__ == "__main__":
    main()
