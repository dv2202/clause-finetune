"""Write human-readable review sheets for the Phase 1 checkpoint.

data/processed/review_train20.md : 20 random training examples
data/processed/review_golden.md  : all 60 golden examples
The gold span is wrapped in [[[ ]]] inside the excerpt so it's easy to spot.
"""

import json
import random
from pathlib import Path

OUT = Path("data/processed")


def load(name):
    return [json.loads(line) for line in (OUT / f"{name}.jsonl").open()]


def render(records, title, id_prefix):
    lines = [f"# {title}", "",
             "For each item: is `found` right? If found, is `text` the complete clause (not a fragment, not extra)?",
             "Reply with fixes like: `G07: should be found=false` or `T03: text should also include ...`.", ""]
    for i, r in enumerate(records, 1):
        meta = r["meta"]
        answer = json.loads(r["messages"][2]["content"])
        excerpt = r["messages"][1]["content"].split('Excerpt:\n"""\n', 1)[1].rsplit('\n"""', 1)[0]
        if answer["found"]:
            excerpt = excerpt.replace(answer["text"], f"[[[ {answer['text']} ]]]", 1)
        kind = "POSITIVE" if answer["found"] else ("HARD NEGATIVE" if meta["hard"] else "NEGATIVE")
        lines += [f"## {id_prefix}{i:02d} | {meta['clause_type']} | {kind}",
                  f"`{meta['title']}` chunk {meta['chunk_id']}", "",
                  f"**Label:** found={answer['found']}" + (f"  \n**Text:** {answer['text']}" if answer["found"] else ""),
                  "", "<details><summary>Excerpt</summary>", "", "```text", excerpt, "```", "</details>", ""]
    return "\n".join(lines)


def main():
    rng = random.Random(7)  # different seed from the split, just for picking review items
    train20 = rng.sample(load("train"), 20)
    (OUT / "review_train20.md").write_text(render(train20, "Review: 20 random TRAIN examples", "T"))
    golden = sorted(load("golden"), key=lambda r: (r["meta"]["clause_type"], r["meta"]["label"] != "pos"))
    (OUT / "review_golden.md").write_text(render(golden, "Review: 60 GOLDEN examples", "G"))
    print("wrote", OUT / "review_train20.md", "and", OUT / "review_golden.md")


if __name__ == "__main__":
    main()
