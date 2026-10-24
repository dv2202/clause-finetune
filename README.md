# ClauseFinder

Fine-tuning a small open model (Qwen2.5-1.5B-Instruct, 4-bit) with LoRA to find five clause types in contracts and copy them **exactly**, as strict JSON. Trained and evaluated on a MacBook Air M3 (16 GB) with MLX.

**Model (LoRA adapter):** [huggingface.co/rambo2202/clausefinder-qwen2.5-1.5b-lora-mlx](https://huggingface.co/rambo2202/clausefinder-qwen2.5-1.5b-lora-mlx)

```json
{"clause_type": "Governing Law", "found": true, "text": "This Agreement shall be governed by and interpreted pursuant to the laws of the People's Republic of China ..."}
```

**Result on a held-out test set (194 examples from 37 contracts never used in training or for any decision):** F1 went from 0.740 (best prompting) to **0.889**, hallucinated text from 24.6% to **2.3%**, valid JSON from 92.3% to **99.5%**, and latency halved (5.68s to 2.81s per call).

## 1. Problem

Input: one contract excerpt (~400–600 words) and one clause type. Output: one JSON object saying whether the clause is in the excerpt and, if it is, the clause text copied word for word.

Clause types: **Governing Law, Termination For Convenience, Non-Compete, Cap On Liability, Anti-Assignment.**

In legal review a quoted clause must exist in the document. A clause that is "almost" the original (for example "may be assigned" instead of "may not be assigned") is worse than no answer, because it looks right.

## 2. Why fine-tuning and not just prompting

We fine-tuned because we needed a specific behaviour from the model: answer in strict JSON, say whether the clause is found or not, and if it is found, give the supporting text copied exactly.

RAG can't do this on its own. RAG (retrieval-augmented generation) can find the parts of a contract that look related to a clause, but deciding whether that clause is really present needs a model that reads and understands the text. A big model could do this from a prompt, but contracts are often confidential, so sending them to an external API may not be allowed. A small fine-tuned model runs locally.

The baselines confirmed that prompting the small model was not enough. Adding few-shot examples raised precision (0.507 to 0.667 on valid) but made hallucination worse (21.7% to 27.1%), because the model copied sentences from the examples, and it was slower (4.54s to 5.89s). After fine-tuning, the behaviour is in the weights: short prompt, no examples to leak, 2.3% hallucination on test.

## 3. Data

- **Source:** [CUAD v1](https://www.atticusprojectai.org/cuad) (The Atticus Project, CC BY 4.0): 510 real commercial contracts with clause spans labelled by legal experts.
- **Chunking** (`src/prepare_data.py`): whitespace normalised, then paragraph-aware chunks of ~400 words (max 600) with a one-paragraph overlap. Character offsets are kept, so every label comes from CUAD's offsets, never retyped.
- **Labels per (chunk, clause type):** positive if an expert span lies fully inside the chunk (longest span used as the answer); skipped if a span is cut by the chunk boundary (so no wrong labels); otherwise negative. Half the negatives are **hard negatives** that contain related words ("terminate", "assign", "liable") but no labelled clause.
- **Split by contract** (seed 42, 80/10/10), asserted in code so no contract appears in two splits.

| split | examples | positive / negative | hard negatives | contracts | tokens (median / max) |
|---|---|---|---|---|---|
| train | 1000 | 500 / 500 | 250 | 262 | 758 / 1389 |
| valid | 146 | 71 / 75 | 40 | 38 | 783 / 1257 |
| test | 194 | 94 / 100 | 50 | 37 | — |
| golden | 60 (subset of test) | 30 / 30 | 15 | 29 | — |

Source: `results/data_stats.json`. All labels come from CUAD, where they were made by human legal experts. Six labels that looked wrong were dropped (`data/label_overrides.json`). Our extra check of the 60 golden examples was done by an AI assistant, not by a lawyer.

**Why negatives:** the model has to learn both answers, not only "found". Trained on positives only, it would say `found: true` every time. The base model already behaves like that: on test, recall 0.968 but 89 false alarms. Hard negatives teach it to tell look-alike text ("terminate for breach" vs "terminate for convenience") apart.

**Why split by contract:** so no part of a test contract is seen in training. Chunks of the same contract overlap and share wording, names and section structure, so a random split would put near-copies of test examples into training and the score would look better than it really is (data leakage).

## 4. Method

- **Base model:** `mlx-community/Qwen2.5-1.5B-Instruct-4bit`. Chosen over newer small Qwen models because they are vision-language or "thinking" models, which add complexity for a pure extraction task.
- **One prompt builder** (`src/prompts.py`) for training data, eval and the demo, so the model is always tested on the prompt it was trained on.
- **LoRA on the 4-bit base (QLoRA-style)** with `mlx_lm.lora` (`configs/lora_v1.yaml`): rank 8, scale 20, dropout 0.05, all linear layers of the top 16 of 28 blocks (5.28M trainable params, 0.34%), batch 2, 300 iterations, AdamW, lr 5e-5 with 30-step warmup and cosine decay, loss on the answer only (`mask_prompt`), gradient checkpointing.
- **Smoke test first** (50 iterations on 100 examples): memory fit, ~8.6 s/iteration, and an important lesson: loss was already 0.055 while the model answered `found: false` to everything. The found decision is one token among ~50, so **loss is not the metric**.
- **Own eval harness** (`src/eval.py`): greedy decoding, strict Pydantic parsing, and per-run JSON with every prediction, so every number can be traced to an example (`src/inspect_pred.py`).
- **Checkpoint chosen on the full valid set**, not on val loss. Targets were written before any training (`LEARNING_LOG.md`, Phase 2).

### Metrics

- **valid JSON:** output parses into the schema and is consistent (found=true has text, found=false has null).
- **precision / recall / F1** on the found decision.
- **span match:** share of gold clauses that were found *and* copied with token-overlap F1 ≥ 0.5 against the expert span.
- **hallucination:** share of "found" answers whose text does not appear in the excerpt (whitespace-insensitive).

## 5. Results

### Validation (146), used for all decisions (`results/summary.md`)

| run | valid JSON | precision | recall | F1 | span match | hallucination | latency |
|---|---|---|---|---|---|---|---|
| B0 base, zero-shot | 94.5% | 0.507 | 0.986 | 0.670 | 62.0% | 21.7% | 4.54s |
| B1 base, few-shot | 96.6% | 0.667 | 0.901 | 0.766 | 60.6% | 27.1% | 5.89s |
| LoRA it150 | 99.3% | 1.000 | 0.648 | 0.786 | 57.7% | 6.5% | 2.36s |
| LoRA it225 | 100.0% | 1.000 | 0.803 | 0.891 | 67.6% | 5.3% | 2.13s |
| **LoRA it300 (chosen)** | **100.0%** | **1.000** | **0.873** | **0.932** | 66.2% | 8.1% | 2.11s |
| LoRA it400 (resumed) | 100.0% | 1.000 | 0.577 | 0.732 | 47.9% | 4.9% | 3.31s |
| LoRA it500 (resumed) | 100.0% | 0.982 | 0.789 | 0.875 | 69.0% | 12.3% | 3.40s |

Resuming for 200 more iterations made the model worse (it400 recall 0.577), so iteration 300 was kept.

### Test (194), run once (`results/FINAL_*_test.json`)

| run | valid JSON | precision | recall | F1 | span match | hallucination | latency |
|---|---|---|---|---|---|---|---|
| B0 base, zero-shot | 92.8% | 0.506 | 0.968 | 0.664 | 56.4% | 21.7% | 3.10s |
| B1 base, few-shot | 92.3% | 0.675 | 0.819 | 0.740 | 56.4% | 24.6% | 5.68s |
| **LoRA it300** | **99.5%** | **0.930** | **0.851** | **0.889** | **68.1%** | **2.3%** | **2.81s** |

F1 per clause type on test, LoRA vs B1: Governing Law 0.947 vs 0.727, Termination For Convenience 0.919 vs 0.750, Non-Compete 0.815 vs 0.583, Cap On Liability 0.800 vs 0.700, Anti-Assignment 0.947 vs 0.878.

### Golden (60, balanced subset of test) (`results/FINAL_*_golden.json`)

| run | valid JSON | precision | recall | F1 | span match | hallucination | latency |
|---|---|---|---|---|---|---|---|
| B1 base, few-shot | 85.0% | 0.639 | 0.767 | 0.697 | 46.7% | 27.8% | 5.27s |
| **LoRA it300** | **100.0%** | 0.964 | 0.900 | **0.931** | **76.7%** | **0.0%** | 3.43s |

### Targets set before training (checked on test)

| target | result | |
|---|---|---|
| valid JSON ≥ 99% | 99.5% | pass |
| precision ≥ 0.80 | 0.930 | pass |
| F1 ≥ 0.85 | 0.889 | pass |
| hallucination ≤ 10% | 2.3% | pass |
| latency < B1 | 2.81s vs 5.68s | pass |
| span match ≥ 70% | 68.1% | **miss** |

**General ability:** 5 general prompts (summarise, a fact question, an email, explain an NDA to a child, a list) give normal answers almost identical to the base model's, with no JSON leaking in (`results/phase5_general_check.json`).

Latency is measured on a fanless laptop that slows down when hot, so compare it only within one table.

## 6. Failure examples and fixes

**1. Few-shot prompting created hallucinations (B1).** The model copied sentences from the prompt's own examples into its answers, flipped meaning ("may not be assigned" became "may be assigned"), and drifted on long copies (`[***]` became `[X]`). Hallucination rose from 21.7% (B0) to 27.1% (B1) on valid.
*Fix:* fine-tuning on answers that are exact substrings of the excerpt, with a short prompt and no examples. Test hallucination fell to 2.3%. The 5 hallucinations left on valid are all two real pieces of the excerpt joined with words skipped in between, with no invented words.

**2. Invalid JSON from unescaped characters (B0/B1).** Checked against the raw outputs: unescaped `"` inside the copied text (9 of 13 cases), raw newlines inside the string (2), and a clause type outside the allowed list (2). None were caused by the token limit.
*Fix:* training targets are written with `json.dumps`, so the model learned correct escaping. Valid JSON: 92.3% to 99.5% on test.

**3. Merging the adapter into the 4-bit model broke it.** After `mlx_lm.fuse`, test F1 dropped from 0.889 to 0.756 and precision from 0.930 to 0.625 (`results/FINAL_FUSED_test.json`). Fusing into 16-bit (`--dequantize`) gave 40/40 outputs identical to base + adapter on a valid check, so fusing itself is lossless. The cause is re-quantizing to 4 bits: measured over all adapted matrices, the rounding error is 1.29× the size of the LoRA update at 4-bit (0.15× at 8-bit).
*Fix:* deploy the 4-bit base + the 20 MB adapter (the verified setup). Lesson: re-run eval after every conversion step.

**Still open: wrong or partial spans.** The model sometimes finds the right chunk but copies a neighbouring sentence or only a lead-in (e.g. "The Parties agree and acknowledge that:" instead of the liability clause that follows). This is why span match misses its target.

## 7. Demo

A CLI demo (contract text in, verified clause JSON out, with a guard that blocks text not found in the document) is in [`app/README.md`](app/README.md).

## 8. Limitations

- Span match is 68.1% on test, below the 70% target; wrong-sentence picks are the main remaining error.
- Small evaluation sets: 94 test positives (14 for Non-Compete), so one example moves recall by about 1 point.
- Labels are CUAD's human expert labels; our extra review of the golden set was done by an AI assistant, not a lawyer.
- Keyword retrieval in the demo was checked on one contract only.
- CUAD is English, mostly US commercial contracts filed with the SEC; other contract types and languages are untested.
- Keyword retrieval in the demo can miss a clause that uses unusual wording.
- The MLX model runs only on Apple Silicon.
- Not legal advice.

## 9. Reproduce

```bash
uv sync                                                       # Python 3.12, mlx-lm, datasets, pandas, pydantic
uv run python src/prepare_data.py                             # needs data/raw/CUADv1.json (CUAD data.zip)
uv run python src/eval.py --split valid --run-name B0_zero_shot
uv run python src/eval.py --split valid --few-shot --run-name B1_few_shot
uv run mlx_lm.lora -c configs/lora_v1.yaml                    # ~54 min on an M3 Air
uv run python src/eval.py --split test --adapter adapters/lora_v1 --run-name FT_test
```

`results/summary.md` is rebuilt from `results/*.json` after every eval run.

**Run the demo from a fresh clone, without training** (Apple Silicon):

```bash
git clone https://github.com/dv2202/clause-finetune.git && cd clause-finetune
uv sync
uv run hf download rambo2202/clausefinder-qwen2.5-1.5b-lora-mlx --local-dir adapters/final_it300
uv run python app/demo.py app/samples/tuniu_cooperation_agreement.txt   # downloads the base model on first run
```

## Repo map

| path | what |
|---|---|
| `src/prompts.py` | the one shared prompt builder |
| `src/prepare_data.py` | CUAD to chunked, labelled, split JSONL |
| `src/eval.py` | eval harness and metrics |
| `src/show_failures.py`, `src/inspect_pred.py` | error analysis tools |
| `src/general_check.py` | general-ability (forgetting) check |
| `configs/` | smoke, v1 and resume training configs |
| `app/demo.py` | CLI demo with retrieval and the verify guard |
| `results/` | every run as JSON, logs, `summary.md` |
| `LEARNING_LOG.md` | concepts and interview notes per phase |
