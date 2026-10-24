# Learning Log: ClauseFinder fine-tuning

Revision notes, one section per phase: concepts learned + interview Q&A.

---

## Phase 0: Setup + sanity check

### What was done
- Python 3.12 env via `uv`; installed `mlx-lm[train]`, `datasets`, `pandas`, `pydantic`.
- Base model: `mlx-community/Qwen2.5-1.5B-Instruct-4bit`.
  - Why not `Qwen3.5-2B`: it is a vision-language, hybrid-architecture model (`Qwen3_5ForConditionalGeneration`); LoRA support in mlx-lm is risky for a one-night sprint.
  - Why not `Qwen3-1.7B`: "thinking mode" adds `<think>` tokens, which complicate training data and loss masking, with no benefit for strict-JSON extraction.
  - Qwen2.5-1.5B-Instruct: plain causal LM, simple ChatML template, proven with mlx-lm LoRA.
- `src/prompts.py`: the ONE prompt builder shared by eval, training data and the demo.
- `src/sanity_check.py`: runs the base model on 2 hand-written examples (1 positive, 1 hard negative).

### Concepts
- **Instruct model**: a base LLM further trained on (instruction, response) chat data, so it follows instructions instead of just continuing text.
- **4-bit quantization**: weights stored in 4 bits instead of 16. 1.5B params is ~3 GB in fp16 and under 1 GB in 4-bit. LoRA on top of a frozen 4-bit model is "QLoRA-style".
- **Chat template**: the model-specific way roles are turned into one token string (Qwen uses `<|im_start|>role\n...<|im_end|>`). Train and inference must use the same template, so all prompts come from one file.

- **How a training/inference example becomes tokens**:
  1. `prompts.py` builds a Python list: `[{"role": "system", ...}, {"role": "user", ...}]` (+ `{"role": "assistant", ...}` for training).
  2. `tokenizer.apply_chat_template(messages)` runs a Jinja template stored in the model's `tokenizer_config.json`. It wraps each message as `<|im_start|>{role}\n{content}<|im_end|>\n`.
  3. `add_generation_prompt=True` appends `<|im_start|>assistant\n` so the model knows it is its turn.
  4. The string is tokenized into ids. Our example 1 prompt = 210 tokens.
- **Special tokens**: `<|im_start|>` is ONE reserved id (151644). The same characters typed as plain text become 6 ordinary tokens (`<`, `|`, `im`, `_start`, `|`, `>`). Reserved ids give the model an unambiguous signal for "a new turn starts / this role is speaking / the turn ended", which normal text (that might contain `<|` by chance) cannot fake. `<|im_end|>` is also how generation knows when to stop.

### Base model observations (HUMAN CHECKPOINT, 2026-10-24)
Raw outputs saved in `results/phase0_sanity_check.txt`.
- Format: both outputs were valid JSON with no extra text and no markdown fences. Format is not the problem for this model on these 2 examples.
- Example 1 (Governing Law, present): `found=true` is correct, but `text` is only a fragment ("without regard to its conflict of laws principles."), the part naming Delaware law is missing. **Incomplete span.**
- Example 2 (Non-Compete, absent, hard negative): `found=true` is wrong; it returned the "Term" clause as a non-compete. **False positive / hallucinated clause.**
- Which is worse in legal work: example 2. A false positive tells the lawyer a restriction exists when it does not, and the lawyer may advise the client on a clause that isn't there. The incomplete span at least points to the right location, so a human reviewer can recover. This is why we track **hallucination rate** and **precision** separately.
- Note: 2 examples is a smoke test, not a measurement. Real numbers come from the Phase 2 baselines.

### Interview Q&A
**Q: Why did you choose a 4-bit 1.5B instruct model?**
A: It fits a 16 GB Mac with room for LoRA training (about 1 GB of weights), it already follows chat instructions so we only teach the task, and a plain causal LM with a simple chat template avoids risk with architecture support and thinking tokens. Tradeoff: some quality loss from quantization, which LoRA on top partially recovers (QLoRA idea).

**Q: What is a chat template and what breaks if it's wrong?**
A: It's the model-specific format that turns role-tagged messages into one token string with special tokens. If training data and inference use different templates, the model sees inputs it was never trained on and quality drops silently. We prevent this with one shared `prompts.py` and the tokenizer's own template.

**Q: The base model already outputs valid JSON. Why fine-tune?**
A: Valid format isn't correct content. The base model returned a fragment instead of the full clause and labelled an unrelated clause as a non-compete. Fine-tuning on expert-labelled CUAD data teaches when to say `found: false` and where clause boundaries are. We'll prove the gap with baselines in Phase 2, not assume it.

---

## Phase 1: Build the dataset

### What was done
- Source: official CUAD zip (`data/raw/CUADv1.json`, 510 contracts, SQuAD format). The HF `theatticusproject/cuad-qa` repo is only a loading script, which `datasets` 5.x refuses to run, so we used the spec's fallback.
- `src/prepare_data.py` pipeline: whitespace normalisation (with an old->new offset map) -> split by contract title 80/10/10 (seed 42) -> paragraph-aware chunking with 1-paragraph overlap (target 400, max 600 words) -> label each (chunk, clause type) -> sample positives + negatives -> chat JSONL.
- Labelling: positive if a gold span lies fully inside the chunk (longest one used); skip if a span is cut by the boundary; negative otherwise. Hard negative = negative chunk containing clause keywords ("governed", "terminate", "assign", ...).
- Final data (numbers from `results/phase1_prepare_data.txt`): train 1000 (500/500, 262 contracts), valid 146 (71/75), test 194 (94/100), golden 60 (30/30, subset of test). Half of all negatives are hard. Tokens: train p50 758, p99 1118, max 1389, so max_seq_length 2048 truncates nothing.
- Non-Compete is rare: only 11 valid and 14 test positives exist under our rules, so valid/test are slightly below 150/200.
- `data/raw/train_separate_questions.json` + `test.json` are CUAD's official 408/102 benchmark split. Not used: we need a validation set, so we made our own contract-level 80/10/10.

### Label review (checkpoint delegated to Claude, 2026-10-24)
Devansh had no time, so Claude reviewed instead. **Golden is "Claude-reviewed", not human-verified.** Method: a regex screen for strong clause patterns inside negatives, plus fragment flags on positives; then all flagged negatives (all splits) and all flagged valid/test positives were read; flagged train positives were skimmed.
- Rule: positives whose gold text is < 8 words are skipped (CUAD sometimes highlights fragments like "You must:" or a bare heading).
- 6 hand overrides in `data/label_overrides.json` (each with a reason): 3 train negatives that actually contain the clause, 2 train positives with the wrong clause type / a competitor list, 1 test positive that was a fragment whose real clause sat outside the chunk.
- Sampling made stable (hash-based priority per item instead of a shared shuffled RNG), so dropping one item doesn't reshuffle every other pick.
- Remaining known noise: CUAD often highlights clauses in pieces (mid-sentence starts/ends). These are expert labels and were kept.

### Concepts
- **Leakage**: if chunks of one contract are in both train and test, the model can memorise that contract's wording and look better than it is. Split by contract first, then chunk, and assert no overlap.
- **Negatives**: with only positives the model learns "always say found=true" (exactly the Phase 0 hallucination). ~1:1 negatives teach when to say no; hard negatives teach meaning over keywords.
- **Frozen test set**: test is used once at the end. If we tune on it, the score stops measuring generalisation.
- **Never create a wrong label**: when a span is cut by a chunk boundary, skip that chunk instead of guessing.

### Interview Q&A
**Q: How did you avoid data leakage?**
A: Split by contract title (seed 42) before chunking, so every chunk of a contract lives in exactly one split, and asserted zero title overlap across train/valid/test in code. Chunk-level random splitting would leak near-duplicate text through the overlapping chunks.

**Q: Why include negatives, and what's a hard negative?**
A: An extraction model in production mostly sees chunks without the clause; without negatives it learns to always answer found=true and hallucinate. We used ~1:1 negatives from the same contracts, and half are hard negatives (they contain words like "terminate" or "governed" but no labelled clause), so the model must understand meaning, not match keywords.

**Q: How did you handle label noise in CUAD?**
A: Three layers: skip ambiguous chunks automatically (span cut by a boundary), a rule for fragment labels (< 8 words), and a reviewed override file with a written reason per item. Overrides are keyed by (contract, clause type, chunk) and sampling is hash-based, so fixes are reproducible and don't reshuffle the dataset.

**Q: Why did you normalise whitespace?**
A: CUAD text has long runs of spaces from PDF extraction. They waste tokens, and the task is exact copying, so the model would have to reproduce random spacing. We collapsed whitespace and kept an offset map so gold spans still point to the exact text, then asserted every positive's text is a substring of its excerpt.
