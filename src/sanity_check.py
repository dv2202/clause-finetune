"""Phase 0: run the BASE model on 2 hand-written examples and show raw outputs.

Goal: see how the untuned model behaves before we train anything.
"""

import json
import time

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from prompts import MODEL_ID, build_messages

# Example 1 (positive): a Governing Law clause is present, buried among other text.
EXCERPT_1 = """12. Notices. All notices under this Agreement shall be in writing and delivered by certified mail to the addresses set forth above.

13. Governing Law. This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware, without regard to its conflict of laws principles.

14. Entire Agreement. This Agreement constitutes the entire agreement between the parties and supersedes all prior understandings."""

# Example 2 (hard negative): talks about assignment-like words ("transfer", "successors")
# but has NO non-compete clause, so the right answer is found=false.
EXCERPT_2 = """7. Successors. This Agreement shall be binding upon and inure to the benefit of the parties and their respective successors.

8. Confidentiality. Each party shall keep confidential all non-public information received from the other party and shall not transfer such information to any third party.

9. Term. This Agreement shall commence on the Effective Date and continue for a period of two (2) years."""

EXAMPLES = [("Governing Law", EXCERPT_1), ("Non-Compete", EXCERPT_2)]


def main():
    # load() downloads (first time) and returns the 4-bit model + its tokenizer.
    model, tokenizer = load(MODEL_ID)
    # temp=0.0 means greedy decoding: same input gives same output (needed for fair eval).
    sampler = make_sampler(temp=0.0)

    for i, (clause_type, excerpt) in enumerate(EXAMPLES, 1):
        messages = build_messages(clause_type, excerpt)
        # The chat template turns role/content dicts into one string with special tokens.
        # add_generation_prompt=True appends "<|im_start|>assistant\n" so the model starts answering.
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        token_ids = tokenizer.encode(prompt)

        if i == 1:
            # Show the template once, so we can see what the model really reads.
            print("=" * 80, "\nFULL TEMPLATED PROMPT (example 1):\n", prompt, sep="")
            print("=" * 80)
            print(f"Prompt length: {len(token_ids)} tokens")
            print("First 25 tokens (id -> piece):")
            for tid in token_ids[:25]:
                print(f"  {tid:>7} -> {tokenizer.decode([tid])!r}")

        start = time.perf_counter()
        output = generate(model, tokenizer, prompt=prompt, max_tokens=256, sampler=sampler)
        elapsed = time.perf_counter() - start

        print("\n" + "=" * 80)
        print(f"EXAMPLE {i}: clause_type={clause_type!r}  ({elapsed:.1f}s)")
        print("RAW OUTPUT:\n" + repr(output))
        try:
            json.loads(output)
            print("-> parses as JSON: yes")
        except json.JSONDecodeError:
            print("-> parses as JSON: NO")


if __name__ == "__main__":
    main()
