"""Tests for Gradio adapter CLI argument parsing."""

from __future__ import annotations

import argparse

import pytest

from ems_prepared.adapters.gradio.args import from_namespace, register_arguments
from ems_prepared.model.context import Locale


def build_parser() -> argparse.ArgumentParser:
    """Create one parser with shared and Gradio-specific flags."""
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--policy", default="graph")
    _ = parser.add_argument(
        "--locale",
        choices=[Locale.EN.value, Locale.DE.value],
        default=Locale.EN.value,
    )
    _ = parser.add_argument(
        "--experiment-name",
        "--experiment",
        "-e",
        dest="experiment_name",
        default=None,
    )
    register_arguments(parser)
    return parser


def test_parse_args_accepts_locale() -> None:
    """Gradio args should preserve shared locale/policy values from main parser."""
    parser = build_parser()
    namespace = parser.parse_args(
        [
            "--policy",
            "agent",
            "--locale",
            "german",
            "--experiment",
            "exp_shared",
        ]
    )
    args = from_namespace(namespace)

    assert args.locale == Locale.DE.value
    assert args.policy == "agent"
    assert args.experiment_name == "exp_shared"
    assert args.ui == "standard"
    assert args.guided_scenarios == ()
    assert args.enable_asr is False
    assert args.asr_model == "openai/whisper-base.en"
    assert args.require_consent is False
    assert args.consent_file is None


def test_parse_args_uses_locale_default() -> None:
    """Gradio args should use shared locale default when not provided."""
    parser = build_parser()
    namespace = parser.parse_args([])
    args = from_namespace(namespace)

    assert args.locale == Locale.EN.value
    assert args.experiment_name is None
    assert args.picker_interactive is True


def test_parse_args_accepts_guided_mode_without_explicit_scenarios() -> None:
    """Guided mode should allow the scenario directory to define the run order."""
    parser = build_parser()
    namespace = parser.parse_args(
        [
            "--ui",
            "guided",
            "--scenario-dir",
            "./experiments/llm_vs_graph/scenarios",
        ]
    )
    args = from_namespace(namespace)

    assert args.ui == "guided"
    assert args.guided_scenarios == ()
    assert args.picker_interactive is False


def test_parse_args_accepts_dynamic_guided_scenario_count() -> None:
    """Guided mode should preserve any explicit scenario count and order."""
    parser = build_parser()
    namespace = parser.parse_args(
        [
            "--ui",
            "guided",
            "--scenario-dir",
            "./experiments/llm_vs_graph/scenarios",
            "--guided-scenario",
            "Scenario_01.md",
            "--guided-scenario",
            "Scenario_02.md",
            "--guided-scenario",
            "Scenario_03.md",
            "--guided-scenario",
            "Scenario_04.md",
        ]
    )
    args = from_namespace(namespace)

    assert args.guided_scenarios == (
        "Scenario_01.md",
        "Scenario_02.md",
        "Scenario_03.md",
        "Scenario_04.md",
    )


def test_parse_args_rejects_guided_mode_without_scenario_dir() -> None:
    """Guided mode should require an explicit scenario directory."""
    parser = build_parser()
    namespace = parser.parse_args(["--ui", "guided"])

    with pytest.raises(ValueError, match="requires --scenario-dir"):
        _ = from_namespace(namespace)


def test_parse_args_rejects_random_scenario_in_guided_mode() -> None:
    """Guided mode should reject random scenario selection."""
    parser = build_parser()
    namespace = parser.parse_args(
        [
            "--ui",
            "guided",
            "--scenario-dir",
            "./experiments/llm_vs_graph/scenarios",
            "--random-scenario",
        ]
    )

    with pytest.raises(ValueError, match="random-scenario"):
        _ = from_namespace(namespace)


def test_parse_args_accepts_verbose_without_debug() -> None:
    """Gradio args should support verbose logging without enabling debug mode."""
    parser = build_parser()
    namespace = parser.parse_args(["--verbose"])
    args = from_namespace(namespace)

    assert args.debug is False
    assert args.verbose is True
    assert args.should_skip_policy_calls() is False
    assert args.is_verbose_logging_enabled() is True


def test_parse_args_accepts_asr_flags() -> None:
    """ASR flags should be preserved in adapter args."""
    parser = build_parser()
    namespace = parser.parse_args(
        ["--enable-asr", "--asr-model", "openai/whisper-small.en"]
    )
    args = from_namespace(namespace)

    assert args.enable_asr is True
    assert args.asr_model == "openai/whisper-small.en"


def test_parse_args_debug_skips_policy_calls() -> None:
    """Debug mode should bypass policy calls in the Gradio adapter."""
    parser = build_parser()
    namespace = parser.parse_args(["--debug"])
    args = from_namespace(namespace)

    assert args.debug is True
    assert args.should_skip_policy_calls() is True
    assert args.is_verbose_logging_enabled() is True


def test_parse_args_accepts_consent_flags() -> None:
    """Consent flags should be preserved in Gradio args."""
    parser = build_parser()
    namespace = parser.parse_args(
        ["--require-consent", "--consent-file", "/tmp/consent.md"]
    )
    args = from_namespace(namespace)

    assert args.require_consent is True
    assert args.consent_file == "/tmp/consent.md"


def test_parse_args_gradio_debug_env_is_ignored_by_adapter_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GRADIO_DEBUG should not affect adapter-level debug/verbose behavior."""
    monkeypatch.setenv("GRADIO_DEBUG", "1")
    parser = build_parser()
    namespace = parser.parse_args([])
    args = from_namespace(namespace)

    assert args.debug is False
    assert args.is_debug_enabled() is False
    assert args.should_skip_policy_calls() is False
    assert args.is_verbose_logging_enabled() is False
