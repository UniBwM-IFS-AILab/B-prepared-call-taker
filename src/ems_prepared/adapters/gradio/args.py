"""CLI argument parsing for the Gradio adapter."""

import argparse
import logging
import os
import random
from dataclasses import dataclass
from typing import Literal, get_args

from ems_prepared.model.contracts import PolicyName

logger = logging.getLogger(__name__)

# TODO: pydantic configdict or typer?
# TODO: use one CLI for gradio, cli loop and fastapi server


@dataclass(frozen=True)
class GradioAppArgs:
    """Parsed CLI arguments for the Gradio app."""

    scenario_dir: str | None
    user_id: int
    policy: PolicyName | Literal["random"]
    debug: bool
    random_scenario: bool
    experiment_name: str | None
    exit_on_launch: bool

    @property
    def picker_interactive(self) -> bool:
        """Whether the scenario picker should be interactive.

        Disabled when random scenario mode is enabled.
        Can be extended with other conditions in the future.
        """
        return not self.random_scenario

    def is_debug_enabled(self) -> bool:
        """Determine whether debug visuals should be shown.

        Checks both CLI flag and GRADIO_DEBUG environment variable.
        """
        env_debug = os.getenv("GRADIO_DEBUG", "") == "1"
        debug_enabled: bool = self.debug or env_debug
        if debug_enabled:
            os.environ["GRADIO_LOG_LEVEL"] = "debug"
        return debug_enabled

    def resolve_policy(self) -> PolicyName:
        """Resolve the policy setting, handling 'random' by selecting one at random.

        Returns:
            The resolved policy name ('graph' or 'agent')

        """
        if self.policy == "random":
            selected = random.choice(get_args(PolicyName))
            logger.info(f"Random policy selection: chose '{selected}'")
            return selected
        return self.policy


def parse_args() -> GradioAppArgs:
    """Parse CLI arguments and return typed GradioAppArgs."""
    parser = argparse.ArgumentParser(
        description="CLI for configuring and running the Gradio app.",
        epilog=(
            "Examples:\n"
            "  python cli.py --scenario-dir ./scenarios --user-id 42 --policy graph --debug --random-scenario --experiment-name exp_2024_12\n"
        ),
    )
    _ = parser.add_argument(
        "--scenario-dir",
        type=str,
        default=None,
        help="Path to the directory containing scenario descriptions.",
    )
    _ = parser.add_argument(
        "--user-id",
        "--user",
        "-u",
        dest="user_id",
        type=int,
        default=0,
        help="User ID as an integer (default: 0).",
    )
    _ = parser.add_argument(
        "--policy",
        "-p",
        dest="policy",
        type=str,
        default="graph",
        choices=[*get_args(PolicyName), "random"],
        help="Policy to use: 'graph' (default) for pydantic_graph, 'agent' for LLM-only agent, or 'random' to select one randomly.",
    )
    _ = parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode to show additional session information in the UI.",
    )
    _ = parser.add_argument(
        "--random-scenario",
        "--rs",
        dest="random_scenario",
        action="store_true",
        help="Enable random scenario selection on each session reset.",
    )
    _ = parser.add_argument(
        "--experiment-name",
        "--experiment",
        "-e",
        dest="experiment_name",
        type=str,
        default=None,
        help="Optional experiment/run name; logs are saved under experiments/<name>/logs/...",
    )
    _ = parser.add_argument(
        "--exit-on-launch",
        action="store_true",
        help="Exit process after launching the Gradio server instead of staying attached for logs.",
    )

    args, _ = parser.parse_known_args()

    return GradioAppArgs(
        scenario_dir=args.scenario_dir,
        user_id=args.user_id,
        policy=args.policy,
        debug=args.debug,
        random_scenario=args.random_scenario,
        experiment_name=args.experiment_name,
        exit_on_launch=args.exit_on_launch,
    )
