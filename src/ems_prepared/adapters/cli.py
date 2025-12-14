"""CLI argument parsing for the Gradio app."""

import argparse
import os
import random
from dataclasses import dataclass

from loguru import logger

# Available policies for random selection
AVAILABLE_POLICIES = ["graph", "agent"]

# TODO: pydantic configdict or typer?
# TODO: use one CLI for gradio, cli loop and fastapi server


@dataclass(frozen=True)
class GradioAppArgs:
    """Parsed CLI arguments for the Gradio app."""

    scenario_dir: str | None
    user_id: int
    policy: str
    debug: bool
    random_scenario: bool
    experiment_name: str | None

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

    def resolve_policy(self) -> str:
        """Resolve the policy setting, handling 'random' by selecting one at random.

        Returns:
            The resolved policy name ('graph' or 'agent')
        """
        if self.policy == "random":
            selected = random.choice(AVAILABLE_POLICIES)
            logger.info(f"Random policy selection: chose '{selected}'")
            return selected
        return self.policy


def parse_args() -> GradioAppArgs:
    """Parse CLI arguments and return typed GradioAppArgs."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--scenario-dir",
        type=str,
        default=None,
        help="Path to the directory for the scenario descriptions.",
    )
    parser.add_argument(
        "--user-id",
        "--user",
        "-u",
        dest="user_id",
        type=int,
        default=0,
        help="Integer user ID (default: 0).",
    )
    parser.add_argument(
        "--policy",
        "-p",
        dest="policy",
        type=str,
        default="graph",
        choices=["graph", "agent", "random"],
        help="Policy to use: 'graph' for pydantic_graph (default), 'agent' for LLM-only agent, or 'random' to select one with equal probability.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show additional session debug information in the UI.",
    )
    parser.add_argument(
        "--random-scenario",
        "--rs",
        dest="random_scenario",
        action="store_true",
        help="Randomize scenario selection on each session reset.",
    )
    parser.add_argument(
        "--experiment-name",
        "--experiment",
        "-e",
        dest="experiment_name",
        type=str,
        default=None,
        help="Optional experiment/run name to group logs into a subdirectory (e.g., 'exp_2024_12').",
    )

    args, _ = parser.parse_known_args()

    return GradioAppArgs(
        scenario_dir=args.scenario_dir,
        user_id=args.user_id,
        policy=args.policy,
        debug=args.debug,
        random_scenario=args.random_scenario,
        experiment_name=args.experiment_name,
    )


# Module-level singleton (parsed once on import)
args = parse_args()
