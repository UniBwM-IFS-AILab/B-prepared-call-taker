"""Tests for Gradio adapter CLI argument parsing."""

from __future__ import annotations

import argparse

from ems_prepared.adapters.gradio.args import GradioAppArgs, register_arguments
from ems_prepared.model.context import Locale


def test_parse_args_accepts_locale() -> None:
    """Gradio args should preserve shared locale/policy values from main parser."""
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
    args = GradioAppArgs(
        scenario_dir=namespace.scenario_dir,
        user_id=namespace.user_id,
        policy=namespace.policy,
        locale=namespace.locale,
        debug=namespace.debug,
        random_scenario=namespace.random_scenario,
        experiment_name=namespace.experiment_name,
        exit_on_launch=namespace.exit_on_launch,
    )
    assert args.locale == Locale.DE.value
    assert args.policy == "agent"
    assert args.experiment_name == "exp_shared"


def test_parse_args_uses_locale_default() -> None:
    """Gradio args should use shared locale default when not provided."""
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
    namespace = parser.parse_args([])
    args = GradioAppArgs(
        scenario_dir=namespace.scenario_dir,
        user_id=namespace.user_id,
        policy=namespace.policy,
        locale=namespace.locale,
        debug=namespace.debug,
        random_scenario=namespace.random_scenario,
        experiment_name=namespace.experiment_name,
        exit_on_launch=namespace.exit_on_launch,
    )
    assert args.locale == Locale.EN.value
    assert args.experiment_name is None
