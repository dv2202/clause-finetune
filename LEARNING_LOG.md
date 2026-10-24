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

---

## Phase 2: Eval harness + baselines

### What was done
- `src/eval.py`: our own harness. Greedy decoding (temp 0), strict Pydantic parse of the whole output, metrics overall + per clause type, saves `results/<run>.json` (config + metrics + every prediction) and rebuilds `results/summary.md`.
- `src/show_failures.py` (failure buckets) and `src/inspect_pred.py` (trace one prediction back to the excerpt, our label and CUAD's raw expert spans, and recompute every check).
- `src/prompts.py` got the B1 few-shot variant: stricter rules + 3 hand-written examples (not from CUAD, so no eval leakage). Training data keeps the plain prompt.

### Metric definitions
- **valid JSON**: whole stripped output parses into the schema, and found/text agree (found=true needs text, found=false needs null). Invalid output counts as found=false.
- **precision / recall / F1** on `found`.
- **span match**: of all gold positives, share where the model found the clause AND token-overlap F1 with the gold text >= 0.5.
- **hallucination rate**: of the model's found=true answers, share whose text (whitespace-collapsed) is not a substring of the excerpt. Narrow on purpose: fabricated text. A real sentence with the wrong label is a false positive (precision), not a hallucination.

### Baselines on valid (n=146), from `results/summary.md`
| run | valid JSON | precision | recall | F1 | span match | hallucination | latency |
|---|---|---|---|---|---|---|---|
| B0 zero-shot | 94.5% | 0.507 | 0.986 | 0.670 | 62.0% | 21.7% | 4.54s |
| B1 few-shot | 96.6% | 0.667 | 0.901 | 0.766 | 60.6% | 27.1% | 5.89s |

### Failure analysis (B1, checkpoint 2026-10-24)
B1 failure buckets (first matching rule wins): 26 hallucinated text, 20 false positive, 15 wrong/partial span, 5 missed, 5 invalid JSON (71 of 146).
- **Hallucination patterns**: (1) copying the few-shot examples' sentences into answers (Governing Law hallucination rose from 16.7% in B0 to 36.7% in B1); (2) meaning flips ("may not be assigned" became "may be assigned"); (3) drift while copying long clauses (e.g. `[***]` rewritten as `[X]`).
- **False positives**: keyword traps, e.g. "Termination for Material Breach" labelled as Termination For Convenience, or a termination-for-cause clause returned for Cap On Liability.
- **Invalid JSON** (checked against the raw outputs, not guessed): unescaped `"` inside the copied text (9 of 13 across B0+B1), a raw newline inside the string (2), clause_type outside the allowed list (2, B0 only). None hit the 800-token limit (max 473).
- Takeaway: prompting improves precision but makes hallucination worse and costs latency. Training data written with `json.dumps` teaches proper `\"` and `\n` escaping.

### Targets for "done" (locked before any training, valid split, fine-tuned model)
- valid JSON >= 99%
- precision >= 0.80
- F1 >= 0.85
- span match >= 70%
- hallucination <= 10%
- latency < B1 (5.89s)

### Concepts
- **Precision vs recall vs F1**: precision = of the clauses we claimed, how many were right; recall = of the real clauses, how many we found; F1 = harmonic mean, low if either is low.
- **Gold vs pred vs golden**: gold = the correct label; pred = the model's answer; golden.jsonl = the 60-example hand-check file (Phase 5 only).
- **Why baselines first**: without "before" numbers, "the fine-tune helped" is a feeling. B1 is the bar to beat, because it's the cheapest strong alternative.
- **Why targets are written before training**: so we can't move the goalposts after seeing results.

### Interview Q&A
**Q: Why write your own eval script instead of using a platform?**
A: We have ground-truth labels and a fixed output schema, so evaluation is deterministic code: parse, compare, count. A 200-line script is transparent, versioned with the repo, and every number can be traced to a prediction (we built `inspect_pred.py` for exactly that).

**Q: Why is hallucination rate tracked separately from precision?**
A: They fail differently. A false positive quotes real contract text with the wrong label, so a lawyer can find it with Ctrl+F and catch the mistake. A hallucination quotes text that doesn't exist, sometimes with flipped meaning ("may not" became "may"), and that error is invisible unless someone checks. In legal work that's the most dangerous failure, so it gets its own metric and target.

**Q: Few-shot improved precision. Why not stop there?**
A: B1 raised precision from 0.51 to 0.67 but hallucination went up from 21.7% to 27.1%, because the model copied sentences from the few-shot examples into its answers, and latency rose 30% from the longer prompt. Fine-tuning puts the behaviour in the weights, so the prompt stays short and there are no examples to leak.

## Phase 3: First fine-tune (LoRA v1)

### What we did
- Smoke test (`configs/smoke.yaml`, 100 train / 20 valid from `src/make_smoke_data.py`, 50 iters, lr 2e-5): memory fit, val loss 0.703 -> 0.055, ~8.6 s/iter (9m03s total). Log: `results/phase3_smoke_train.txt`.
- Smoke adapter on 20 valid examples (`results/smoke_check.json`): valid JSON 100%, latency 1.31s, but it answered `found: false` on all 20 (12 gold positives missed, F1 0). Data was checked: 55 of 100 smoke targets are `found: true`.
- Real run `configs/lora_v1.yaml`: 300 iters, batch 2 (600 examples, ~0.6 epoch), lr 5e-5 with 30-step warmup + cosine decay to 5e-6, rank 8, scale 20, dropout 0.05, top 16 of 28 layers, mask_prompt, grad checkpointing. Log: `results/phase3_lora_v1_train.txt`.
- Wall time 54 min (11:40 -> 12:34, from adapter file timestamps), above the 48 min estimate: the fanless Air throttled (~10.5 s/iter incl. evals).
- Val loss (30 shortest valid examples): iter 1 0.519, 75 0.048, 150 0.026, 225 0.026, 300 0.038. Checkpoints saved every 75 iters in `adapters/lora_v1/`.

### Concepts
- **LoRA**: freeze the base weights W, learn a low-rank update B·A (rank r) added to chosen linear layers. Trainable params per matrix = r × (d_in + d_out). Ours: 5.28M = 0.34% of the model.
- **QLoRA-style**: the frozen base is 4-bit (small memory), the LoRA matrices train in normal precision.
- **mask_prompt**: loss only on the assistant's JSON, so the model learns to answer, not to predict contract text.
- **Loss is not the metric**: the found true/false decision is 1 token among ~50 answer tokens, so loss can be 0.05 while every decision is wrong (the smoke adapter). Only eval.py tells us task quality.
- **Reading loss curves**: both falling = learning; train falling but val rising = overfitting (use an earlier checkpoint); both flat and high = underfitting. Our val went 0.026 -> 0.038 at the end: a small rise on only 30 examples, so possibly noise, possibly early overfit. Checkpoint 150/225 vs 300 is decided by eval.py on full valid in Phase 4.
- **Warmup + cosine schedule**: ramp lr up so the first random-ish updates don't wreck the model, then decay so late steps make small refinements.
- **Grad checkpointing**: don't store all activations, recompute them during backprop. Less memory, ~30% slower.

### Interview Q&A
**Q: Why LoRA instead of full fine-tuning?**
A: Full fine-tuning of 1.5B params needs memory for the weights, gradients and Adam states (several times the model size) and creates a full model copy per task. LoRA trains 0.34% of the params, fits on a 16 GB laptop on top of a 4-bit base, and the adapter is a 21 MB file that can be swapped per task. For a narrow extraction task the low-rank update is enough.

**Q: Your training loss was 0.05 after 20 steps. Was the model done?**
A: No. The smoke test proved it: loss 0.055 but the model answered found=false on every example. Most answer tokens are format and copied text, which are easy, while the decision is one token. So I always run the task metric (eval.py) on checkpoints instead of trusting loss.

**Q: How did you choose iterations and learning rate?**
A: From the smoke test, not by guessing: it measured ~8.6 s/iter, so 300 iters fit the ~45 min budget. It also showed lr 2e-5 for 50 steps learned format but not the decision, so I raised it to 5e-5 with warmup and cosine decay, and saved checkpoints every 75 steps to pick the best by valid score.
