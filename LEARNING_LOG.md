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
