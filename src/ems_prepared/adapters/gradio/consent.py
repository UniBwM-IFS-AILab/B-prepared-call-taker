"""Consent helpers shared by Gradio standard and guided flows."""

from __future__ import annotations

from pathlib import Path

from ems_prepared.adapters.gradio.args import GradioAppArgs

DEFAULT_CONSENT_MARKDOWN = (
    "### Consent\n\n"
    "By starting this simulation, you confirm that you understand this is a **training scenario** "
    "and does not connect to real emergency services.\n\n"
    "You agree that your interactions may be logged for research and quality improvement."
)

DEFAULT_CONSENT_CHECKBOX_LABEL = "I have read and accept the consent information."
DEFAULT_CONSENT_CONTEXT_MARKDOWN = (
    "### Before You Begin\n\n"
    "Please review the consent information before continuing to the simulation."
)


def resolve_consent_markdown(args: GradioAppArgs) -> str | None:
    """Return consent markdown when enabled, otherwise None."""
    if not args.require_consent:
        return None
    if args.consent_file is None:
        return DEFAULT_CONSENT_MARKDOWN

    path = Path(args.consent_file)
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(
            f"Unable to read --consent-file '{path}': {exc}"
        ) from exc
