"""Terminal adapter that runs one interactive session via SessionService."""

from __future__ import annotations

import asyncio
from argparse import ArgumentParser, Namespace
from typing import Final
from uuid import UUID

from rich import print
from rich.prompt import Prompt

from ems_prepared.model.context import InputMode, Locale
from ems_prepared.model.contracts import (
    BackendEvent,
    FrontendPlugin,
    SessionManager,
    SessionParameters,
)
from ems_prepared.model.errors import UnsupportedPolicyError

EXIT_COMMANDS: Final[set[str]] = {"exit", "quit", "/exit", "/quit"}


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


def run_terminal_session(
    manager: SessionManager,
    args: Namespace,
) -> int:
    """Run one interactive terminal session and return a shell exit code.

    This keeps one persistent asyncio runner and executes blocking prompt input
    outside async code so a single Ctrl-C exits cleanly.
    """
    runner = asyncio.Runner()
    session_id: UUID | None = None
    try:
        try:
            handle, pending_events = runner.run(
                manager.start_session(
                    SessionParameters(
                        frontend_name="cli",
                        policy_name=args.policy,
                        scenario_name=None,
                        locale=Locale(args.locale),
                        call_origin=InputMode.CLI,
                        experiment_name=args.experiment_name,
                    )
                )
            )
        except UnsupportedPolicyError as exc:
            print(f"[red]{exc}[/red]")
            return 2

        session_id = handle.session_id
        while True:
            question, stop, code = _render_events(pending_events)
            if stop:
                return code
            if not question or not question.strip():
                print("[yellow]No follow-up question emitted; ending session.[/yellow]")
                return 1

            try:
                user_input = Prompt.ask(question)
            except (KeyboardInterrupt, EOFError):
                print("\n[yellow]Interrupted. Closing session...[/yellow]")
                return 130
            if user_input.strip().lower() in EXIT_COMMANDS:
                return 0

            pending_events = runner.run(manager.handle_input(session_id, user_input))
            if not pending_events:
                return 0
    except KeyboardInterrupt:
        print("\n[yellow]Interrupted. Closing session...[/yellow]")
        return 130
    finally:
        if session_id is not None:
            _ = runner.run(manager.end_session(session_id))
        runner.close()


class CliFrontend(FrontendPlugin):
    """Frontend plugin for the interactive terminal session adapter."""

    def register_arguments(self, subparser: ArgumentParser, /) -> None:
        """Register terminal frontend specific arguments."""
        del subparser

    def run(
        self,
        session_manager: SessionManager,
        parsed_args: Namespace,
        /,
    ) -> int | None:
        """Run the interactive terminal session."""
        return run_terminal_session(session_manager, parsed_args)
