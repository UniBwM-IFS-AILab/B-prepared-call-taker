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
    BackendEventKind,
    FrontendPlugin,
    MessageType,
    SessionManager,
    SessionParameters,
)

EXIT_COMMANDS: Final[set[str]] = {"exit", "quit", "/exit", "/quit"}


def _print_interrupt_message() -> None:
    print("\n[yellow]Interrupted. Leaving session active.[/yellow]")


def _render_events(events: list[BackendEvent]) -> tuple[str | None, bool, int]:
    """Print backend events and return next question + terminal status."""
    next_question: str | None = None

    for event in events:
        if event.kind is BackendEventKind.MESSAGE:
            if event.text:
                print(event.text)
            continue

        if event.kind is BackendEventKind.QUESTION:
            next_question = event.text
            continue

        if event.kind is BackendEventKind.COMPLETED:
            if event.text:
                print(event.text)
            return None, True, 0

        if event.kind is BackendEventKind.ERROR:
            print(f"[red]{event.text or 'Internal error'}[/red]")
            return None, True, 1

        if event.text:
            print(event.text)

    return next_question, False, 0


def _load_resumed_events(
    manager: SessionManager,
    raw_session_id: str,
) -> tuple[UUID, list[BackendEvent]]:
    try:
        session_id = UUID(raw_session_id)
    except ValueError:
        if not raw_session_id.isdigit():
            raise ValueError(f"Invalid session id: {raw_session_id}") from None
        session_id = UUID(int=int(raw_session_id))

    history = manager.get_history(session_id)
    if history is None:
        raise LookupError(f"Unknown session: {session_id}")
    if not history.messages or history.messages[-1].type is not MessageType.QUESTION:
        raise ValueError(f"Session has no pending question to resume: {session_id}")

    return session_id, [
        BackendEvent(
            kind=BackendEventKind.QUESTION,
            text=history.messages[-1].content,
        )
    ]


def _start_session(
    runner: asyncio.Runner,
    manager: SessionManager,
    args: Namespace,
) -> tuple[UUID, list[BackendEvent]]:
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
    return handle.session_id, pending_events


def _drive_session(
    runner: asyncio.Runner,
    manager: SessionManager,
    session_id: UUID,
    pending_events: list[BackendEvent],
) -> int:
    while True:
        question, stop, code = _render_events(pending_events)
        if stop:
            return code
        if not question or not question.strip():
            print("[yellow]No follow-up question emitted; stopping CLI.[/yellow]")
            return 1

        try:
            user_input = Prompt.ask(question)
        except EOFError:
            print("\n[yellow]Input closed. Leaving session active.[/yellow]")
            return 130
        if user_input.strip().lower() in EXIT_COMMANDS:
            return 0

        pending_events = runner.run(manager.handle_input(session_id, user_input))
        if not pending_events:
            return 0


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
        raw_session_id = getattr(args, "session_id", None)
        if raw_session_id:
            try:
                session_id, pending_events = _load_resumed_events(
                    manager, raw_session_id
                )
            except (LookupError, ValueError) as exc:
                print(f"[red]{exc}[/red]")
                return 2
        else:
            try:
                session_id, pending_events = _start_session(runner, manager, args)
            except Exception as exc:
                print(f"[red]{exc}[/red]")
                return 2

        return _drive_session(runner, manager, session_id, pending_events)
    except KeyboardInterrupt:
        _print_interrupt_message()
        return 130
    finally:
        runner.close()


class CliFrontend(FrontendPlugin):
    """Frontend plugin for the interactive terminal session adapter."""

    def register_arguments(self, subparser: ArgumentParser, /) -> None:
        """Register terminal frontend specific arguments."""
        _ = subparser.add_argument(
            "--session-id",
            default=None,
            help="Resume an existing active session by UUID or decimal integer.",
        )

    def run(
        self,
        session_manager: SessionManager,
        parsed_args: Namespace,
        /,
    ) -> int | None:
        """Run the interactive terminal session."""
        return run_terminal_session(session_manager, parsed_args)
