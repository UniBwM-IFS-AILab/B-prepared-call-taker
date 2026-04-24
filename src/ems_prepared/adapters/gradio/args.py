"""CLI argument parsing for the Gradio adapter."""

from __future__ import annotations

import argparse
import logging
import os
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
    experiment_name: str | None = None
    exit_on_launch: bool = False
    ui: str = "standard"
    guided_scenarios: tuple[str, ...] = ()

    @property
    def picker_interactive(self) -> bool:
        """Whether the scenario picker should be interactive.

        Disabled when random scenario mode is enabled.
        """
        return self.ui == "standard" and not self.random_scenario

    def is_debug_enabled(self) -> bool:
        """Determine whether debug visuals should be shown.

        Checks both CLI flag and GRADIO_DEBUG environment variable.
        """
        env_debug = os.getenv("GRADIO_DEBUG", "") == "1"
        debug_enabled: bool = self.debug or env_debug
        if debug_enabled:
            os.environ["GRADIO_LOG_LEVEL"] = "debug"
        return debug_enabled

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
        help="Enable debug mode to show additional session information in the UI.",
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
        random_scenario=args.random_scenario,
        experiment_name=args.experiment_name,
        exit_on_launch=args.exit_on_launch,
        ui=args.ui,
        guided_scenarios=tuple(args.guided_scenarios),
    ).validate()
