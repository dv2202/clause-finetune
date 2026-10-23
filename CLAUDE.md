# ClauseFinder: LLM fine-tuning project (teach-me mode)

## Who I am and how you should work with me

I'm Devansh. I'm preparing for an Applied AI Engineer role (LLM fine-tuning) at a legal-AI startup. I know RAG, but I'm **new to fine-tuning**. The point of this project is that I can **explain every step in an interview**, not just that the code works.

So you are my **senior engineer + teacher**, not just a code generator:

1. **Before each phase:** explain in 5–10 lines of simple Hinglish (Hindi-English mix, easy words) *what* we're about to do and *why*. Use one small real example.
2. **While working:** keep each script small and readable. Add a short comment on every important line explaining *why* (not just what).
3. **After each phase:** give me:
   - a 5-line summary of what was built and what the output/numbers mean,
   - the 2–3 interview questions this phase prepares me for, with short model answers,
   - then **stop and wait** for me to say "next".
4. **HUMAN CHECKPOINT** steps (marked below) are mine. Stop, tell me exactly what to look at, and wait for my answer before continuing.
5. Never run a training job longer than ~10 minutes without asking me first. Tell me the expected time before starting.
6. Never invent or estimate results. Every number in the README must come from a script output saved in `results/`.
7. If something fails, show me the error, explain the cause in one line, then fix it.
8. Keep a running learning log in `LEARNING_LOG.md`: for each phase, the concepts learned + interview Q&A. This is my revision notes.

## Hardware and constraints

- MacBook Air M3, 16 GB unified memory, no fan (it throttles on long runs), ~45 GB free disk.
- macOS 27. Use `uv` for the Python env (Python 3.12).
- Training framework on Mac: **MLX / mlx-lm** (Apple's library; LoRA on a 4-bit model = QLoRA-style).
- Base model: a small instruct model from `mlx-community` in 4-bit, around 1.5B parameters (e.g. `mlx-community/Qwen2.5-1.5B-Instruct-4bit`, or a newer small Qwen if available). Check what exists before choosing and tell me why you picked it.
- Keep batch size small (1–2), use gradient checkpointing, max sequence length ≤ 2048 unless data needs more.

## The task

Input: one contract excerpt (a chunk of ~300–800 words) + one clause type.
Output: strict JSON only:

```json
{"clause_type": "Governing Law", "found": true, "text": "<exact clause text copied from the excerpt>"}
```

or, if not present:

```json
{"clause_type": "Governing Law", "found": false, "text": null}
```

Clause types (only these 5 to keep scope small): **Governing Law, Termination For Convenience, Non-Compete, Cap On Liability, Anti-Assignment.**

## Data source

- CUAD (Contract Understanding Atticus Dataset): 510 real contracts, labelled by legal experts, CC BY 4.0.
- Try Hugging Face `theatticusproject/cuad-qa` first. If the `datasets` library refuses its loading script, download the raw data from the official repo zip: `https://github.com/TheAtticusProject/cuad/raw/main/data.zip` (use `CUAD_v1.json`, SQuAD format).
- Inspect the real field names and the exact category wording before writing the converter. Show me 2 raw records first.

## Repo structure

```
clause-finetune/
  CLAUDE.md
  LEARNING_LOG.md
  README.md
  data/raw/            # downloaded CUAD
  data/processed/      # train.jsonl, valid.jsonl, test.jsonl, golden.jsonl
  src/prepare_data.py
  src/eval.py
  src/prompts.py       # ONE shared prompt builder used by eval, training data and the demo
  configs/lora_v1.yaml
  adapters/            # trained LoRA adapters (gitignored)
  results/             # every eval run as JSON + a comparison table
  app/                 # small demo (FastAPI or Gradio)
  kaggle/              # bonus: PyTorch + PEFT + TRL notebook
```

## ONE-NIGHT SPRINT MODE (use this unless I say otherwise)

I have one night. Keep the same phases, but shrink them:
- Model: ~1.5B 4-bit. Train set: ~800–1,200 examples. Valid: ~150. Test: ~200 (+ golden 60).
- Chunks: ~400–600 words so each example is ~600–900 tokens (faster training).
- Phase 3: smoke test (50 iters) → ONE real run (~600–1,000 iters, aim ≤ 45 min). Measure it/sec in the smoke test and give me the real time estimate.
- Phase 4: at most ONE extra run, only if there is an obvious data fix.
- Phase 6: README + a simple CLI demo (skip the web app). Phase 7 can be done later.
- Teaching stays: short explanations, interview Q&A after each phase, checkpoints still mine (keep them quick).

## Data-building spec (Phase 1 details)

- Source file: `CUAD_v1.json` (SQuAD format). Structure: `data[] → paragraphs[0].context` = full contract text; `qas[]` = 41 questions, one per category; each `answers[]` has `text` + `answer_start` (character offset). Category name is inside the question text — print a few and map them to my 5 types.
- Chunking: **paragraph-aware fixed-size chunking with overlap** (not parent-child; parent-child is a retrieval trick, not needed for building training data). Split on blank lines / numbered sections, merge paragraphs until ~400–600 words, overlap ~1 paragraph. Keep each chunk's start/end char offsets.
- Labelling a chunk for a clause type: positive if a gold span (`answer_start` … `answer_start+len(text)`) lies **fully inside** the chunk. If a span is cut by a boundary, don't label that chunk positive or negative for that type (skip it) — never create a wrong label. If several spans of the same type are in one chunk, use the longest one as `text`.
- Negatives: same clause type, chunks from the same contract with no gold span of that type. ~1:1 with positives. At least a third should be **hard negatives** (chunk contains related words like "governed", "terminate", "assign", "liability", "compete" but no label).
- Split by contract title with a fixed seed (42): 80/10/10. Assert no title overlap across splits.
- Verify: re-check that every positive's `text` appears exactly in its chunk.

## Phases

### Phase 0: Setup + sanity check (≈30 min)
- Create the env, install `mlx-lm[train]`, `datasets`, `pandas`, `pydantic`.
- Run the base model on 2 hand-written examples using the prompt from `src/prompts.py`.
- Teach: what "instruct", "4-bit" and "chat template" mean, using the actual tokenized prompt (print it).
- **HUMAN CHECKPOINT:** I read the base model's raw outputs and write down what's wrong (format, extra talk, wrong text).

### Phase 1: Build the dataset (≈1–2 hrs)
- `src/prepare_data.py`:
  1. Load CUAD, keep the 5 clause types.
  2. Split contracts into chunks (~300–800 words, split on paragraph boundaries, small overlap). Keep character offsets.
  3. Positives: chunks containing a labelled clause. Negatives: chunks without it, ~1:1 with positives. Include some "hard negatives" (similar wording, wrong clause).
  4. **Split by contract title** (80/10/10) so no contract is in two splits. Assert this in code.
  5. Write chat-format JSONL (`{"messages": [...]}`) that mlx-lm accepts.
  6. Cap train at ~1,500–2,500 examples; print counts per split, per clause type, positive/negative ratio, and token-length percentiles.
- Create `golden.jsonl`: 60 test examples (balanced) for me to verify by hand.
- Teach: why split by contract (leakage), why negatives (hallucination), why the test set is frozen.
- **HUMAN CHECKPOINT:** I review 20 random training examples and the 60 golden examples, fix/flag bad labels. You apply my fixes.

### Phase 2: Eval harness + baselines, the "before" numbers (≈1–2 hrs)
- Write our **own** eval script (no LangSmith or other platforms needed: we have ground-truth labels, so a Python script is the right tool and is what I must be able to explain).
- `src/eval.py --model ... [--adapter ...] --split valid|test|golden`:
  - temperature 0; parse JSON with a Pydantic schema.
  - Metrics, overall and per clause type: valid-JSON rate, precision, recall, F1 on `found`, span match (token-overlap ≥ 0.5 with the gold text), **hallucination rate** (`found: true` but `text` not in the excerpt), avg latency.
  - Save predictions + metrics to `results/<run_name>.json`; update `results/summary.md` table.
- Baselines on **valid**:
  - B0: base model, zero-shot prompt.
  - B1: base model, few-shot prompt (2–3 examples) with stricter instructions.
- Teach: precision vs recall vs F1, why hallucination rate matters in legal work, why baselines come first.
- **HUMAN CHECKPOINT:** I read 10 failures from B1 and group them into types. Then I write down target numbers for "done" in `LEARNING_LOG.md` before any training.

### Phase 3: First fine-tune (≈1–2 hrs)
- Smoke test first: ~50 iterations on 100 examples, to confirm memory, loss going down and output format.
- Then `configs/lora_v1.yaml` for the real run. Explain every field (rank, scale/alpha, dropout, target layers, learning rate, iterations, batch size, grad checkpointing, prompt masking, eval interval, save interval) in 1 line each.
- Tell me the estimated training time. **HUMAN CHECKPOINT:** I approve the config before the full run.
- During training, explain how to read train vs validation loss. Save checkpoints.

### Phase 4: Evaluate, analyse, iterate (≈2–3 hrs)
- Run `eval.py` with the adapter on **valid**; compare with B0/B1 in `results/summary.md`.
- Error analysis: group failures by type, show examples.
- **HUMAN CHECKPOINT:** I choose what to change next (data fix vs hyperparameter). Change ONE thing per run. Max 3 runs.
- Teach: overfitting signs, why data fixes usually beat hyperparameter tweaks.

### Phase 5: Final test, run once (≈30 min)
- Pick the best adapter by **valid** score. Run on **test** and **golden** exactly once.
- Also spot-check 5 general prompts (e.g. "summarise this paragraph") to check the model didn't forget general ability.
- Produce the final before/after table.

### Phase 6: Package and demo (≈1–2 hrs)
- `mlx_lm.fuse` the best adapter; re-run eval on the fused model to confirm same numbers.
- Small demo in `app/`: paste contract text → split into chunks → run all 5 clause types → show JSON results. (Stretch: simple retrieval to pick the top chunks per clause type.)
- `README.md`: problem, why fine-tuning (not just prompting), data + split, method, results table, 3 failure examples and fixes, limitations, next steps. Every number from `results/`.
- **HUMAN CHECKPOINT:** I write the "why" sections in my own words; you only correct mistakes.

### Phase 7 (bonus): Same training on NVIDIA GPU with the industry stack
- Write `kaggle/qlora_peft_trl.ipynb`: Hugging Face Transformers + PEFT + TRL `SFTTrainer` + bitsandbytes 4-bit (NF4) on the same `train.jsonl`/`valid.jsonl`, same base model family (non-MLX version).
- Use fp16 (Kaggle T4 has no bf16). Explain each config line and map it to the MLX config.
- **HUMAN CHECKPOINT:** I upload the data and run the notebook on Kaggle myself (you can't run Kaggle). Then we eval the downloaded adapter and add a row to the results table.

## Definition of done
- Fine-tuned model beats B1 on valid and test (targets written in Phase 2).
- Valid-JSON rate ≥ 99%, hallucination rate clearly lower than B1.
- All numbers reproducible from scripts; README complete; `LEARNING_LOG.md` covers every phase.
- I can explain every file in the repo without help.

Start with Phase 0. Explain first, then do.