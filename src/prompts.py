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


def build_messages(clause_type: str, excerpt: str, few_shot: bool = False) -> list[dict]:
    """Chat messages WITHOUT the answer: used for inference (eval + demo).

    few_shot=True is baseline B1: stricter rules + 3 worked examples as earlier
    chat turns. Training data always uses the plain (zero-shot) version.
    """
    if not few_shot:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(clause_type, excerpt)},
        ]
    messages = [{"role": "system", "content": SYSTEM_PROMPT + STRICT_RULES}]
    for ex_type, ex_excerpt, ex_text in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": build_user_prompt(ex_type, ex_excerpt)})
        messages.append({"role": "assistant", "content": build_target(ex_type, ex_text)})
    messages.append({"role": "user", "content": build_user_prompt(clause_type, excerpt)})
    return messages


# ---------- baseline B1 (few-shot) only ----------

STRICT_RULES = (
    "\nRules:\n"
    "1. Set found to true ONLY if the excerpt contains a clause that matches the definition. "
    "Mentioning a related word (e.g. 'terminate', 'assign', 'liability') is not enough.\n"
    "2. When found is true, copy the COMPLETE clause sentence(s) character for character. "
    "Do not shorten, paraphrase or fix typos.\n"
    "3. If you are unsure, set found to false and text to null.\n"
    "4. Output only the JSON object: no markdown, no explanation."
)

# Hand-written (not from CUAD), so no eval data leaks into the prompt.
# One positive, one hard negative, one positive with a long clause.
FEW_SHOT_EXAMPLES = [
    (
        "Governing Law",
        "9. Notices. All notices shall be sent by courier to the addresses above.\n\n"
        "10. Governing Law. This Agreement shall be governed by and construed in accordance with "
        "the laws of the State of New York, without giving effect to its conflict of laws rules.\n\n"
        "11. Counterparts. This Agreement may be executed in counterparts.",
        "This Agreement shall be governed by and construed in accordance with the laws of the State "
        "of New York, without giving effect to its conflict of laws rules.",
    ),
    (
        "Anti-Assignment",
        "5. Successors. This Agreement shall bind and benefit the parties and their successors.\n\n"
        "6. Term. This Agreement starts on the Effective Date and lasts three (3) years, unless "
        "terminated earlier for material breach on thirty (30) days' written notice.",
        None,
    ),
    (
        "Cap On Liability",
        "12. Limitation of Liability. IN NO EVENT SHALL EITHER PARTY'S TOTAL LIABILITY ARISING OUT "
        "OF THIS AGREEMENT EXCEED THE FEES PAID BY CUSTOMER IN THE TWELVE (12) MONTHS PRECEDING THE "
        "CLAIM.\n\n13. Insurance. Supplier shall maintain general liability insurance of at least "
        "$1,000,000 per occurrence.",
        "IN NO EVENT SHALL EITHER PARTY'S TOTAL LIABILITY ARISING OUT OF THIS AGREEMENT EXCEED THE "
        "FEES PAID BY CUSTOMER IN THE TWELVE (12) MONTHS PRECEDING THE CLAIM.",
    ),
]


def parse_excerpt(user_content: str) -> str:
    """Recover the excerpt from a user prompt (eval reads it back from the JSONL)."""
    return user_content.split('Excerpt:\n"""\n', 1)[1].rsplit('\n"""', 1)[0]


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
