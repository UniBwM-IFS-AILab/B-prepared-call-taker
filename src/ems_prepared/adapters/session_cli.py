"""Terminal adapter that runs one interactive session via SessionService."""

from __future__ import annotations

import argparse
from typing import Final, get_args

from rich import print
from rich.prompt import Prompt

from ems_prepared.model.context import InputMode, Locale
from ems_prepared.model.contracts import (
    BackendEvent,
    PolicyName,
    SessionManager,
    SessionParameters,
)
from ems_prepared.model.errors import UnsupportedPolicyError

EXIT_COMMANDS: Final[set[str]] = {"exit", "quit", "/exit", "/quit"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the terminal session adapter."""
    parser = argparse.ArgumentParser(
        description="Run an interactive EMS session in the terminal.",
    )
    _ = parser.add_argument(
        "--policy",
        choices=list(get_args(PolicyName)),
        default="graph",
        help="Policy runtime to use.",
    )
    _ = parser.add_argument(
        "--scenario-name",
        type=str,
        default=None,
        help="Optional scenario name metadata for the run.",
    )
    _ = parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Optional experiment name to group outputs under experiments/<experiment>/logs/...",
    )
    _ = parser.add_argument(
        "--locale",
        choices=[Locale.EN.value, Locale.DE.value],
        default=Locale.EN.value,
        help="Language locale for operator/caller prompts.",
    )
    return parser.parse_args()


def _render_events(events: list[BackendEvent]) -> tuple[str | None, bool, int]:
    """Print backend events and return next question + terminal status."""
    next_question: str | None = None

    for event in events:
        if event.kind == "message":
            if event.text:
                print(event.text)
            continue

        if event.kind == "question":
            next_question = event.text
            continue

        if event.kind == "completed":
            if event.text:
                print(event.text)
            return None, True, 0

        if event.kind == "error":
            print(f"[red]{event.text or 'Internal error'}[/red]")
            return None, True, 1

        if event.text:
            print(event.text)

    return next_question, False, 0


async def run_terminal_session(
    manager: SessionManager,
    args: argparse.Namespace,
) -> int:
    """Run one interactive terminal session and return a shell exit code."""
    try:
        handle, events = await manager.start_session(
            SessionParameters(
                frontend_name="cli",
                policy_name=args.policy,
                scenario_name=args.scenario_name,
                locale=Locale(args.locale),
                call_origin=InputMode.CLI,
                experiment_name=args.experiment_name,
            )
        )
    except UnsupportedPolicyError as exc:
        print(f"[red]{exc}[/red]")
        return 2

    session_id = handle.session_id
    try:
        pending_events = events
        while True:
            question, stop, code = _render_events(pending_events)
            if stop:
                return code
            if not question or not question.strip():
                print("[yellow]No follow-up question emitted; ending session.[/yellow]")
                return 1

            user_input = Prompt.ask(question)
            if user_input.strip().lower() in EXIT_COMMANDS:
                return 0

            pending_events = await manager.handle_input(session_id, user_input)
            if not pending_events:
                return 0
    finally:
        _ = await manager.end_session(session_id)
