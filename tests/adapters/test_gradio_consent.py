"""Tests for Gradio consent configuration helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from ems_prepared.adapters.gradio.args import GradioAppArgs
from ems_prepared.adapters.gradio.consent import (
    DEFAULT_CONSENT_MARKDOWN,
    resolve_consent_markdown,
)
from ems_prepared.model.context import Locale


def make_args(*, require_consent: bool, consent_file: str | None = None) -> GradioAppArgs:
    return GradioAppArgs(
        scenario_dir="scenarios",
        user_id=0,
        policy="graph",
        locale=Locale.EN.value,
        debug=False,
        random_scenario=False,
        require_consent=require_consent,
        consent_file=consent_file,
    )


def test_resolve_consent_markdown_disabled_returns_none() -> None:
    assert resolve_consent_markdown(make_args(require_consent=False)) is None


def test_resolve_consent_markdown_uses_default_when_enabled_without_file() -> None:
    assert resolve_consent_markdown(make_args(require_consent=True)) == DEFAULT_CONSENT_MARKDOWN


def test_resolve_consent_markdown_reads_file(tmp_path: Path) -> None:
    consent_path = tmp_path / "consent.md"
    consent_path.write_text("## Consent\n\nCustom text", encoding="utf-8")

    assert resolve_consent_markdown(
        make_args(require_consent=True, consent_file=str(consent_path))
    ) == "## Consent\n\nCustom text"


def test_resolve_consent_markdown_raises_for_unreadable_file() -> None:
    with pytest.raises(ValueError, match="Unable to read --consent-file"):
        _ = resolve_consent_markdown(
            make_args(require_consent=True, consent_file="/missing/consent.md")
        )

