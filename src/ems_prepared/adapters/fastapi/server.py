"""Standalone FastAPI server entry point for reload-safe uvicorn startup."""

from __future__ import annotations

import argparse
import os

import uvicorn

from ems_prepared.adapters.fastapi.app import create_app
from ems_prepared.main import create_session_manager
from ems_prepared.model.context import Locale
from ems_prepared.plugins import load_policy_factories

_POLICY_ENV = "EMS_PREPARED_FASTAPI_POLICY"
_LOCALE_ENV = "EMS_PREPARED_FASTAPI_LOCALE"
_EXPERIMENT_ENV = "EMS_PREPARED_FASTAPI_EXPERIMENT"


def _build_parser() -> argparse.ArgumentParser:
    """Build the dedicated FastAPI server parser."""
    parser = argparse.ArgumentParser(
        description="Run the FastAPI adapter through a reload-safe entry point."
    )
    _ = parser.add_argument("-H", "--host", default="127.0.0.1")
    _ = parser.add_argument("--port", type=int, default=8000)
    _ = parser.add_argument("--reload", action="store_true")
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
    return parser


def create_runtime_app(
    *,
    policy: str,
    locale: Locale,
    experiment_name: str | None,
):
    """Create the FastAPI app from runtime dependencies."""
    policy_factories = load_policy_factories()
    session_manager = create_session_manager(policy_factories=policy_factories)
    return create_app(
        session_manager,
        default_policy=policy,
        default_locale=locale,
        app_default_experiment_name=experiment_name,
    )


def create_app_from_environment():
    """Create the FastAPI app from the environment passed to uvicorn workers."""
    experiment_name = os.environ.get(_EXPERIMENT_ENV)
    return create_runtime_app(
        policy=os.environ.get(_POLICY_ENV, "graph"),
        locale=Locale(os.environ.get(_LOCALE_ENV, Locale.EN.value)),
        experiment_name=experiment_name or None,
    )


def main(argv: list[str] | None = None) -> int:
    """Run uvicorn through an import-string entry point so reload works correctly."""
    args = _build_parser().parse_args(argv)
    os.environ[_POLICY_ENV] = args.policy
    os.environ[_LOCALE_ENV] = args.locale
    if args.experiment_name is None:
        os.environ.pop(_EXPERIMENT_ENV, None)
    else:
        os.environ[_EXPERIMENT_ENV] = args.experiment_name

    uvicorn.run(
        "ems_prepared.adapters.fastapi.server:create_app_from_environment",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
