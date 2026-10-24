# ClauseFinder demo

`app/demo.py`: contract text in, verified clause JSON out. Runs the fine-tuned model (4-bit base + LoRA adapter) on Apple Silicon.

```bash
HF_HUB_OFFLINE=1 uv run python app/demo.py app/samples/tuniu_cooperation_agreement.txt
HF_HUB_OFFLINE=1 uv run python app/demo.py my_contract.txt --top-k 0                 # read every chunk (slower, no retrieval misses)
HF_HUB_OFFLINE=1 uv run python app/demo.py my_contract.txt --out results/my_report.json
```

## Pipeline

```
contract.txt -> normalise -> chunks (~400 words) -> keyword retrieval (top 3 chunks per clause type)
             -> fine-tuned model -> verify guard -> report
```

- **Normalise and chunk:** the same functions as the training data (`src/prepare_data.py`), so the model sees text in the format it was trained on.
- **Keyword retrieval:** each chunk is scored by how often the clause type's keywords appear ("governed", "assign", "liable", ...). Only the top 3 chunks per clause type are sent to the model. On a 7-chunk contract that is 12 calls instead of 35. `--top-k 0` reads every chunk.
- **Model:** the same prompt as training (`src/prompts.py`), greedy decoding.

## Verify guard

The model's text is never shown directly. Every answer is checked against the document first:

| status | when | what is shown |
|---|---|---|
| `[OK]` | the answer is an exact copy from the document (whitespace may differ) | the document's own text, with its character position |
| `[REVIEW]` | not exact, but at least 85% similar (word-level) | the closest **real** text from the document, never the model's version |
| `[BLOCKED]` | not close to anything in the document | nothing |
| `[INVALID]` | the output was not valid JSON | nothing |

Tested on the 5 hallucinated answers from the validation run: all 5 became `[REVIEW]` (similarity 0.92–0.99, because they were real sentences with a piece skipped). An invented sentence ("governed by the laws of Mars") was `[BLOCKED]`.

## Example run

Sample: a cooperation agreement from the CUAD **test** split (never used in training), 2,760 words, 7 chunks. 12 model calls in 42s (`results/phase6_demo_tuniu.json`, `results/phase6_demo_tuniu_log.txt`).

Compared with CUAD's expert labels:

| clause type | demo answer | vs expert label |
|---|---|---|
| Governing Law | "This Agreement shall be governed by and interpreted pursuant to the laws of the People's Republic of China ..." | exact match |
| Termination For Convenience | "Party B shall have the right to terminate this Agreement in advance ..." | exact match |
| | "... shall only be terminated upon occurrence of the following circumstances:" | extra answer, a lead-in sentence |
| Non-Compete | "Party A irrevocably undertakes that, without Party B's consent, Party A shall not conduct any other business ..." | exact match |
| Cap On Liability | "The Parties agree and acknowledge that:" | right chunk, wrong text (the clause follows this lead-in) |
| Anti-Assignment | "... this Agreement shall constitute all understanding ..." | right chunk, wrong sentence (an entire-agreement clause) |

Retrieval picked the right chunk for all 5 types.

**Every answer above was `[OK]`, including the wrong ones.** The guard guarantees the quoted text is real, not that it is the right clause. Wrong-sentence picks still need a better model or a human reviewer.

## Limitations

- Apple Silicon only (MLX).
- Keyword retrieval can miss clauses with unusual wording; use `--top-k 0` when recall matters.
- Checked on one contract; the numbers in the main README come from the full test set.
