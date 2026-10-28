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
- **Model:** v2 adapter (`adapters/final_v2_it480`), the same prompt as training (`src/prompts.py`), greedy decoding with repetition penalty 1.05, the exact setting used in the valid and test evals.

## Verify guard

The model's text is never shown directly. Every answer is checked against the document first:

| status | when | what is shown |
|---|---|---|
| `[IN DOC]` | the answer is an exact copy from the document (whitespace may differ) | the document's own text, with its character position |
| `[REVIEW]` | not exact, but at least 85% similar (word-level) | the closest **real** text from the document, never the model's version |
| `[BLOCKED]` | not close to anything in the document | nothing |
| `[INVALID]` | the output was not valid JSON | nothing |

Tested on the 5 hallucinated answers from the validation run: all 5 became `[REVIEW]` (similarity 0.92–0.99, because they were real sentences with a piece skipped). An invented sentence ("governed by the laws of Mars") was `[BLOCKED]`.

## Example run

Sample: a cooperation agreement from the CUAD **test** split (never used in training), 2,760 words, 7 chunks. 12 model calls in 20s (`results/phase6b_demo_tuniu_v2.json`, `results/phase6b_demo_tuniu_v2_log.txt`).

Compared with CUAD's expert labels:

| clause type | v2 answer | vs expert label | v1 answer (for comparison) |
|---|---|---|---|
| Governing Law | "This Agreement shall be governed by and interpreted pursuant to the laws of the People's Republic of China ..." | exact match | exact match |
| Termination For Convenience | "Party B shall have the right to terminate this Agreement in advance ..." | exact match | exact match |
| | "... shall only be terminated upon occurrence of the following circumstances: 6.1.1 Party B shall have the right to terminate ..." | extra answer: lead-in plus the right clause | extra answer: lead-in only |
| Non-Compete | "Party A irrevocably undertakes that, without Party B's consent, Party A shall not conduct any other business ..." | exact match | exact match |
| Cap On Liability | not found | **missed** (experts marked 2 sentences) | 2 wrong sentences (a lead-in and a late-payment penalty) |
| Anti-Assignment | "The rights and obligations of each Party under this Agreement shall not be transferred, except for the transfer by Party B to its affiliates." | exact match | wrong sentence (an entire-agreement clause) |

v2 gets 4 of 5 clause types right on this contract (v1: 3 of 5) and gives no answer from a different clause; its one miss is Cap On Liability, one of the two types (with Non-Compete) where v2 is weakest on the test set too.

**`[IN DOC]` means the quote is real, not that it is the right clause.** v1's wrong answers on this contract were also `[IN DOC]`. Wrong-sentence picks still need a better model or a human reviewer.

## Limitations

- Apple Silicon only (MLX).
- Keyword retrieval can miss clauses with unusual wording; use `--top-k 0` when recall matters.
- Checked on one contract; the numbers in the main README come from the full test set.
