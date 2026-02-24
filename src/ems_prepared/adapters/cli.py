"""CLI argument parsing for the Gradio app."""

import argparse
import logging
import os
import random
from dataclasses import dataclass

logger = logging.getLogger(__name__)

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
    parser = argparse.ArgumentParser(
        description="CLI for configuring and running the Gradio app.",
        epilog=(
            "Examples:\n"
            "  python cli.py --scenario-dir ./scenarios --user-id 42 --policy graph --debug --random-scenario --experiment-name exp_2024_12\n"
        ),
    )
    parser.add_argument(
        "--scenario-dir",
        type=str,
        default=None,
        help="Path to the directory containing scenario descriptions.",
    )
    parser.add_argument(
        "--user-id",
        "--user",
        "-u",
        dest="user_id",
        type=int,
        default=0,
        help="User ID as an integer (default: 0).",
    )
    parser.add_argument(
        "--policy",
        "-p",
        dest="policy",
        type=str,
        default="graph",
        choices=["graph", "agent", "random"],
        help="Policy to use: 'graph' (default) for pydantic_graph, 'agent' for LLM-only agent, or 'random' to select one randomly.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode to show additional session information in the UI.",
    )
    parser.add_argument(
        "--random-scenario",
        "--rs",
        dest="random_scenario",
        action="store_true",
        help="Enable random scenario selection on each session reset.",
    )
    parser.add_argument(
        "--experiment-name",
        "--experiment",
        "-e",
        dest="experiment_name",
        type=str,
        default=None,
        help="Optional name for the experiment/run to group logs into a subdirectory (e.g., 'exp_2024_12').",
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
