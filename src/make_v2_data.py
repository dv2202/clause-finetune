"""Phase 4b: build the v2 training set = ALL of v1's train examples + more from the same train contracts.

Usage: PYTHONPATH=src uv run python src/make_v2_data.py | tee results/v2_prepare_data.txt

The only change vs v1 is MORE data (v1 used 500 of ~1,569 train positives). Same chunking,
labels, overrides, hard-negative share and prompt. valid/test are copied byte-for-byte,
so v1 and v2 are scored on exactly the same examples.
"""

import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path

from prepare_data import (HARD_NEG_SHARE, OVERRIDES, RAW, SEED, build_candidates, priority,
                          sample, to_record, write_jsonl)
from prompts import CLAUSE_TYPES

V1 = Path("data/processed")
OUT = Path("data/processed_v2")
N_POS = 170  # per type; Termination For Convenience (171) and Non-Compete (176) cap the balanced count


def key(c: dict) -> tuple:
    return (c["title"], c["clause_type"], c["chunk_id"])


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def main():
    data = json.loads(RAW.read_text())["data"]

    # Same split as prepare_data.py: same seed, same shuffle -> same train contracts.
    titles = sorted(c["title"] for c in data)
    random.Random(SEED).shuffle(titles)
    train_titles = {t for i, t in enumerate(titles) if i < 0.8 * len(titles)}

    v1 = [json.loads(l) for l in (V1 / "train.jsonl").open()]
    v1_keys = {(r["meta"]["title"], r["meta"]["clause_type"], r["meta"]["chunk_id"]) for r in v1}
    assert {k[0] for k in v1_keys} <= train_titles, "split mismatch with v1"

    dropped = {(o["title"], o["clause_type"], o["chunk_id"]) for o in json.loads(OVERRIDES.read_text())}
    cands = [c for d in data if d["title"] in train_titles for c in build_candidates(d, dropped)]

    new = []
    for t in CLAUSE_TYPES:
        pool = [c for c in cands if c["clause_type"] == t]
        pos, _ = sample(pool, N_POS, 0)
        # sample() walks positives in a fixed order, so v1's 100 are a prefix of these: v2 is a true superset
        v1_pos = {k for k in v1_keys if k[1] == t and any(key(c) == k for c in pos)}
        assert len(v1_pos) == sum(1 for r in v1 if r["meta"]["clause_type"] == t and r["meta"]["label"] == "pos"), t
        add_pos = [c for c in pos if key(c) not in v1_keys]

        # Extra negatives: same type, same contracts as the positives, not already in v1, ~half hard.
        titles_t = {c["title"] for c in pos}
        negs = sorted((c for c in pool if c["label"] == "neg" and c["title"] in titles_t and key(c) not in v1_keys),
                      key=priority)
        n_neg = len(add_pos)  # keep the 1:1 ratio
        hard = [c for c in negs if c["hard"]][: round(n_neg * HARD_NEG_SHARE)]
        easy = [c for c in negs if not c["hard"]][: n_neg - len(hard)]
        new += add_pos + hard + easy
        print(f"{t:28s} positives {len(pos):3d} (v1 100 + new {len(add_pos)})  new negatives {len(hard) + len(easy)} "
              f"({len(hard)} hard)")

    records = v1 + [to_record(c) for c in new]
    random.Random(SEED).shuffle(records)

    # Checks: exact spans, no valid/test contract in train, no duplicates.
    for c in new:
        assert c["label"] == "neg" or c["text"] in c["excerpt"]
    held_out = {json.loads(l)["meta"]["title"] for s in ("valid", "test") for l in (V1 / f"{s}.jsonl").open()}
    assert not ({r["meta"]["title"] for r in records} & held_out), "leakage"
    keys = [(r["meta"]["title"], r["meta"]["clause_type"], r["meta"]["chunk_id"]) for r in records]
    assert len(keys) == len(set(keys)), "duplicate example"

    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT / "train.jsonl", records)
    for s in ("valid", "test"):  # unchanged held-out sets: copied, never rebuilt
        shutil.copyfile(V1 / f"{s}.jsonl", OUT / f"{s}.jsonl")
        assert sha(V1 / f"{s}.jsonl") == sha(OUT / f"{s}.jsonl")

    c = Counter(r["meta"]["label"] for r in records)
    hard = sum(r["meta"]["hard"] for r in records)
    print(f"\ntrain v2: n={len(records)} pos={c['pos']} neg={c['neg']} hard_neg={hard} ({hard / c['neg']:.0%} of neg) "
          f"contracts={len({r['meta']['title'] for r in records})}")
    print(f"valid sha256 {sha(OUT / 'valid.jsonl')}  test sha256 {sha(OUT / 'test.jsonl')}  (identical to v1)")
    print("checks passed: v1 subset, exact spans, no leakage, no duplicates")


if __name__ == "__main__":
    main()
