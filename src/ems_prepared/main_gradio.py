"""Gradio composition root."""

from __future__ import annotations

from ems_prepared.adapters.gradio.app import run_gradio_app
from ems_prepared.adapters.gradio.args import parse_args
from ems_prepared.model.session_service import create_session_manager


def main() -> None:
    """Wire session manager and launch Gradio adapter."""
    run_gradio_app(create_session_manager(), parse_args())


if __name__ == "__main__":
    main()
