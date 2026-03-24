"""CLI composition root for running one terminal EMS session."""

from __future__ import annotations

import asyncio

from ems_prepared.adapters.session_cli import parse_args, run_terminal_session
from ems_prepared.model.session_service import create_session_manager


def main() -> None:
    """Build dependencies and run the terminal adapter."""
    args = parse_args()
    manager = create_session_manager()
    raise SystemExit(asyncio.run(run_terminal_session(manager, args)))


if __name__ == "__main__":
    main()
