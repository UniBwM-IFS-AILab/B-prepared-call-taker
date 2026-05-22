"""CLI argument parsing for the Gradio adapter."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GradioAppArgs:
    """Parsed CLI arguments for the Gradio app."""

    scenario_dir: str | None
    user_id: int
    policy: str
    locale: str
    debug: bool
    random_scenario: bool
    verbose: bool = False
    experiment_name: str | None = None
    exit_on_launch: bool = False
    ui: str = "standard"
    guided_scenarios: tuple[str, ...] = ()
    enable_asr: bool = False
    asr_model: str = "openai/whisper-base.en"
    require_consent: bool = False
    consent_file: str | None = None

    @property
    def picker_interactive(self) -> bool:
        """Whether the scenario picker should be interactive.

        Disabled when random scenario mode is enabled.
        """
        return self.ui == "standard" and not self.random_scenario

    def is_debug_enabled(self) -> bool:
        """Determine whether debug visuals should be shown."""
        return self.debug

    def is_verbose_logging_enabled(self) -> bool:
        """Determine whether verbose Gradio runtime logging should be enabled."""
        return self.debug or self.verbose

    def should_skip_policy_calls(self) -> bool:
        """Return whether Gradio should bypass backend policy calls."""
        return self.debug

    def resolve_policy(self) -> str:
        """Resolve the policy setting for the session start request."""
        return self.policy

    def validate(self) -> GradioAppArgs:
        """Validate mode-specific argument combinations."""
        if self.ui != "guided":
            return self

        if self.random_scenario:
            raise ValueError("--random-scenario is not supported with --ui guided.")
        if self.scenario_dir is None:
            raise ValueError("--ui guided requires --scenario-dir.")
        return self


def register_arguments(subparser: argparse.ArgumentParser) -> None:
    """Register Gradio frontend specific CLI arguments."""
    _ = subparser.add_argument(
        "-s",
        "--scenario-dir",
        type=str,
        default=None,
        help="Path to the directory containing scenario descriptions.",
    )
    _ = subparser.add_argument(
        "--user-id",
        "--user",
        "-u",
        dest="user_id",
        type=int,
        default=0,
        help="User ID as an integer (default: 0).",
    )
    _ = subparser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help=(
            "Enable Gradio debug mode. This bypasses policy calls and auto-completes "
            "after one static response, while still enabling debug logging/UI details."
        ),
    )
    _ = subparser.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "Enable verbose Gradio logging without bypassing policy calls."
        ),
    )
    _ = subparser.add_argument(
        "-r",
        "--random-scenario",
        "--rs",
        dest="random_scenario",
        action="store_true",
        help="Enable random scenario selection on each session reset.",
    )
    _ = subparser.add_argument(
        "--ui",
        choices=["standard", "guided"],
        default="standard",
        help="Select the Gradio UI mode.",
    )
    _ = subparser.add_argument(
        "--guided-scenario",
        dest="guided_scenarios",
        action="append",
        default=[],
        help="Optional guided-mode scenario filename. Repeat to control the run order.",
    )
    _ = subparser.add_argument(
        "--enable-asr",
        action="store_true",
        help="Enable microphone input with streaming ASR.",
    )
    _ = subparser.add_argument(
        "--asr-model",
        default="openai/whisper-base.en",
        help="Transformers ASR model id used when --enable-asr is enabled.",
    )
    _ = subparser.add_argument(
        "--require-consent",
        action="store_true",
        help=(
            "Require explicit consent acceptance in a preliminary step before the scenario flow can start."
        ),
    )
    _ = subparser.add_argument(
        "--consent-file",
        default=None,
        help=(
            "Optional path to a markdown file used as consent text when --require-consent is enabled."
        ),
    )
    _ = subparser.add_argument(
        "-x",
        "--exit-on-launch",
        action="store_true",
        help="Exit process after launching the Gradio server instead of staying attached for logs.",
    )


def from_namespace(args: argparse.Namespace) -> GradioAppArgs:
    """Convert parser namespace to strongly-typed Gradio configuration."""
    return GradioAppArgs(
        scenario_dir=args.scenario_dir,
        user_id=args.user_id,
        policy=args.policy,
        locale=args.locale,
        debug=args.debug,
        verbose=args.verbose,
        random_scenario=args.random_scenario,
        experiment_name=args.experiment_name,
        exit_on_launch=args.exit_on_launch,
        ui=args.ui,
        guided_scenarios=tuple(args.guided_scenarios),
        enable_asr=args.enable_asr,
        asr_model=args.asr_model,
        require_consent=args.require_consent,
        consent_file=args.consent_file,
    ).validate()
