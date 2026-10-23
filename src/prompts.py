"""ONE shared prompt builder.

Eval, training data and the demo all import from here, so the model sees the
exact same prompt at train time and test time (a mismatch silently hurts scores).
"""

import json

MODEL_ID = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"

# Only these 5 CUAD categories are in scope.
# Short definitions (paraphrased from CUAD's category descriptions) tell the
# model what each label means, so it doesn't have to guess from the name alone.
CLAUSE_DEFINITIONS = {
    "Governing Law": "Which state's or country's law governs the interpretation of the contract.",
    "Termination For Convenience": "A party can terminate the contract without cause, e.g. just by giving notice.",
    "Non-Compete": "A restriction on a party's ability to compete with the counterparty or operate in a certain business, geography or sector.",
    "Cap On Liability": "A limit on the amount (or time window) a party can recover or claim for a breach.",
    "Anti-Assignment": "Consent or notice is required if a party wants to assign the contract to a third party.",
}
CLAUSE_TYPES = list(CLAUSE_DEFINITIONS)

# System prompt fixes the role and the output contract. Kept short: every token
# here is repeated in every training example, so it costs training time.
SYSTEM_PROMPT = (
    "You are a contract analysis assistant. You extract clauses from contract excerpts. "
    "Reply with a single JSON object and nothing else."
)


def build_user_prompt(clause_type: str, excerpt: str) -> str:
    """User turn: task + definition + output schema + the excerpt."""
    if clause_type not in CLAUSE_DEFINITIONS:
        raise ValueError(f"Unknown clause type: {clause_type}")
    return (
        f"Clause type: {clause_type}\n"
        f"Definition: {CLAUSE_DEFINITIONS[clause_type]}\n\n"
        "If the excerpt contains this clause, copy the clause text EXACTLY as it appears. "
        "If it does not, set found to false and text to null.\n"
        'Output format: {"clause_type": "<type>", "found": true|false, "text": "<exact text>"|null}\n\n'
        f"Excerpt:\n\"\"\"\n{excerpt}\n\"\"\""
    )


def build_messages(clause_type: str, excerpt: str) -> list[dict]:
    """Chat messages WITHOUT the answer: used for inference (eval + demo)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(clause_type, excerpt)},
    ]


def build_target(clause_type: str, text: str | None) -> str:
    """The exact assistant answer we train on. Compact JSON = fewer tokens to learn."""
    return json.dumps(
        {"clause_type": clause_type, "found": text is not None, "text": text},
        ensure_ascii=False,
    )


def build_training_example(clause_type: str, excerpt: str, text: str | None) -> dict:
    """mlx-lm 'chat' format: {"messages": [...]} including the assistant answer."""
    messages = build_messages(clause_type, excerpt)
    messages.append({"role": "assistant", "content": build_target(clause_type, text)})
    return {"messages": messages}
