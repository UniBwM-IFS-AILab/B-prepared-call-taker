from .schemas import MAX_ACTS_PER_TURN

SLOT_FALLBACK_CHOICE = "NO_SLOT"


SHARED_SELECTION_RULES = "\n".join(
    (
        "- Prefer the minimal act set; most turns require 1-3 acts.",
        f"- Return at most {MAX_ACTS_PER_TURN} acts.",
        "- If no dialogue act applies, return acts=[].",
    )
)

SINGLE_STAGE_SYSTEM_PROMPT = """You label ISO 24617-2 communicative functions for emergency-call transcripts.

The user prompt provides a JSON object that must match the DialogueLabellingInput schema.
Return a JSON object that matches the DialogueLabellingOutput schema.

Rules:
- Only use CommunicativeFunction values and EmergencyCall slots defined by the schema.
- Label only the current utterance. Use context only to resolve references or ellipsis in the current utterance.
- Do not infer a communicative function, slot, or value from context alone if the current utterance does not express it.
{shared_selection_rules}
- Never repeat the same act object or the same communicative_function+slot+value tuple.
- Return only JSON that matches DialogueLabellingOutput.
""".format(shared_selection_rules=SHARED_SELECTION_RULES)

FUNCTION_STAGE_SYSTEM_PROMPT = """You label ISO 24617-2 communicative functions for emergency-call transcripts.

The user prompt provides a JSON object that must match the DialogueLabellingInput schema.
Return only communicative_function labels.

Rules:
- Return JSON matching the provided communicative-function list schema.
- Do not include slot/value fields in this stage.
- Predict labels for the current utterance only.
- Use context only to resolve underspecified wording in the current utterance, not as independent evidence for extra labels.
{shared_selection_rules}
- Never repeat the same communicative_function.
""".format(shared_selection_rules=SHARED_SELECTION_RULES)

SLOT_STAGE_SYSTEM_PROMPT = """Choose the best slot for one communicative function.

Rules:
- Return JSON matching the provided slot-choice schema.
- Use "{slot_fallback_choice}" only when no slot clearly applies from the utterance.
- Use context only to resolve what the current utterance refers to; do not pick a slot that is unsupported by the current utterance itself.
- Do not output any explanation.
""".format(slot_fallback_choice=SLOT_FALLBACK_CHOICE)

VALUE_STAGE_SYSTEM_PROMPT = """Extract the slot value for a communicative function and slot.

Rules:
- Return JSON that matches the provided value schema.
- Use null if the value is not explicitly present in the utterance.
- Do not copy values from context unless the current utterance itself expresses that value.
- Do not output explanation text.
"""

USER_PROMPT_TEMPLATE = """Label the following turn using the provided input JSON:
{input_json}
"""

SLOT_STAGE_USER_PROMPT_TEMPLATE = """Select the best EmergencyCall slot.

communicative_function: {communicative_function}
input_json:
{input_json}
"""

VALUE_STAGE_USER_PROMPT_TEMPLATE = """Extract slot value.

communicative_function: {communicative_function}
slot: {slot}
input_json:
{input_json}
"""
