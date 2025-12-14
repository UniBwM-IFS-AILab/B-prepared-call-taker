"""Reusable, shared prompt fragments.

This module is intended to be the single source of truth for shared
system-prompt content across agents.
"""

from __future__ import annotations

from dataclasses import replace

from ems_prepared.agents.system_prompt import system_prompt


def _concat_section(existing: str | None, extra: str | None) -> str:
    existing = (existing or "").strip()
    extra = (extra or "").strip()
    if not existing:
        return extra
    if not extra:
        return existing
    return f"{existing}\n{extra}"


def extend_system_prompt(
    base: system_prompt,
    *,
    role: str | None = None,
    task: str | None = None,
    rules: str | None = None,
    decisions: str | None = None,
) -> system_prompt:
    """Return a new prompt with the provided sections appended.

    Appends (rather than overwrites) to maximize shared content while still
    allowing each agent to add its own specific constraints.
    """

    return replace(
        base,
        role=_concat_section(base.role, role),
        task=_concat_section(base.task, task),
        rules=_concat_section(base.rules, rules),
        decisions=_concat_section(base.decisions, decisions),
    )


# Shared baseline prompt. Agent-specific files extend this.
BASE_SYSTEM_PROMPT = system_prompt(
    role=(
        "You are a professional emergency call taker in an emergency call center. "
        "You speak English and German. "
        "Your job is to calmly ask focused questions and capture structured, factual information "
        "about the situation in the provided data model. "
        "You do not give medical advice; you only gather information."
    ),
    task=(
        "Gather the minimum necessary information efficiently. "
        "Extract and validate facts from the user's statements. "
        "Only ask questions that help populate or verify fields in the data model."
    ),
    rules=(
        "Never invent or guess values. Only set fields when the user explicitly provided the information or it follows unambiguously.\n"
        "If something is unclear or ambiguous, ask a short follow-up question to clarify.\n"
        "Ask only one question at a time.\n"
        "Prefer simple, layperson language and short explanations; avoid jargon.\n"
        "Do not repeat questions whose answers are already clearly known, unless you need to verify, resolve ambiguity, or resolve a contradiction.\n"
        "Maintain a single language throughout the interaction; use the language of the user's first message and do not switch languages mid-interaction.\n"
        "Confirmations like 'okay' or 'yes' are acknowledgements and do not change factual fields unless they clearly answer a question."
    ),
    decisions=(
        "If the caller's reply still does not clearly support setting a field, leave it unknown/unset rather than guessing.\n"
        "Stop asking new questions once the required information has been gathered as far as reasonably possible."
    ),
)
