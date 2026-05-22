"""Subcommand launcher for discoverable frontend and policy plugins."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Mapping
from functools import partial

from ems_prepared.model.context import Locale
from ems_prepared.model.contracts import (
    FrontendPlugin,
    PolicyFactory,
    SessionManager,
    SessionBackend,
)
from ems_prepared.model.session_backend_file import FileSessionBackend
from ems_prepared.model.session_backend_sqlalchemy import SqlAlchemySessionBackend
from ems_prepared.model.session_service import SessionService
from ems_prepared.plugins import load_frontend_plugins, load_policy_factories


def create_session_manager(
    *,
    policy_factories: Mapping[str, PolicyFactory],
    session_backend: SessionBackend | None = None,
) -> SessionManager:
    """Create a session manager from explicitly provided dependencies."""
    resolved_backend = session_backend or FileSessionBackend()
    return SessionService(
        policy_factories=dict(policy_factories),
        session_backend=resolved_backend,
    )


def _add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    """Add global session configuration flags to one parser."""
    _ = parser.add_argument(
        "-p",
        "--policy",
        default="graph",
        help="Policy runtime name (for example: graph or agent).",
    )
    _ = parser.add_argument(
        "-l",
        "--locale",
        choices=[Locale.EN.value, Locale.DE.value],
        default=Locale.EN.value,
        help="Session locale.",
    )
    _ = parser.add_argument(
        "-e",
        "--experiment",
        "--experiment-name",
        dest="experiment_name",
        default=None,
        help="Optional experiment/run name for session outputs.",
    )
    _ = parser.add_argument(
        "--session-backend",
        choices=["file", "sqlite"],
        default="file",
        help=(
            "Session metadata/logging backend. "
            "Graph policy persistence remains file-based."
        ),
    )
    _ = parser.add_argument(
        "--session-db-url",
        default="sqlite:///logs/sessions.db",
        help=(
            "SQLAlchemy database URL for session metadata/logging when "
            "--session-backend sqlite."
        ),
    )


def _build_parser(
    *,
    frontends: Mapping[str, FrontendPlugin],
) -> argparse.ArgumentParser:
    """Build the root parser and discovered command subparsers."""
    shared = argparse.ArgumentParser(add_help=False)
    _add_shared_arguments(shared)

    parser = argparse.ArgumentParser(
        description="Run EMS Prepared with discoverable frontend and policy plugins.",
        parents=[shared],
        allow_abbrev=False,
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    _ = subparsers.add_parser(
        "list",
        allow_abbrev=False,
        help="List discovered frontend and policy plugins.",
    )
    for frontend_name, plugin in sorted(frontends.items()):
        subparser = subparsers.add_parser(
            frontend_name,
            parents=[shared],
            allow_abbrev=False,
            help=f"Run the {frontend_name} frontend.",
        )
        plugin.register_arguments(subparser)
    return parser


def _print_plugins(title: str, names: Iterable[str]) -> None:
    """Print one plugin group in a stable order."""
    print(title)
    printed_any = False
    for name in sorted(names):
        printed_any = True
        print(f"- {name}")
    if not printed_any:
        print("- <none>")


def _run_list_plugins(
    *,
    frontend_plugins: Mapping[str, FrontendPlugin],
    policy_factories: Mapping[str, PolicyFactory],
) -> int:
    """Run the list management command."""
    _print_plugins("Frontends:", frontend_plugins.keys())
    _print_plugins("Policies:", policy_factories.keys())
    return 0


def _run_frontend_command(
    *,
    plugin: FrontendPlugin,
    args: argparse.Namespace,
    policy_factories: Mapping[str, PolicyFactory],
) -> int:
    """Run one frontend command with a shared session manager."""
    backend: SessionBackend
    if args.session_backend == "sqlite":
        backend = SqlAlchemySessionBackend(database_url=args.session_db_url)
    else:
        backend = FileSessionBackend()

    manager = create_session_manager(
        policy_factories=policy_factories,
        session_backend=backend,
    )
    try:
        code = plugin.run(manager, args)
    except SystemExit as exc:
        return int(exc.code or 0)
    return int(code or 0)


def main(argv: list[str] | None = None) -> int:
    """Run one selected frontend subcommand with a shared session manager."""
    policy_factories = load_policy_factories()
    frontend_plugins = load_frontend_plugins()
    parser = _build_parser(frontends=frontend_plugins)
    args = parser.parse_args(argv)

    command = args.command
    if command is None:
        parser.print_help()
        return 0

    command_handlers: dict[str, Callable[[], int]] = {
        "list": partial(
            _run_list_plugins,
            frontend_plugins=frontend_plugins,
            policy_factories=policy_factories,
        )
    }
    command_handlers.update(
        {
            frontend_name: partial(
                _run_frontend_command,
                plugin=plugin,
                args=args,
                policy_factories=policy_factories,
            )
            for frontend_name, plugin in frontend_plugins.items()
        }
    )

    handler = command_handlers.get(command)
    if handler is None:
        parser.error(f"Unknown command: {command}")
    return handler()


if __name__ == "__main__":
    raise SystemExit(main())
