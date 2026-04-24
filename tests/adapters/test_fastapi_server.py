"""Tests for the standalone FastAPI server entry point."""

from __future__ import annotations

import os

import pytest

from ems_prepared.model.context import Locale


def test_server_main_uses_import_string_and_sets_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dedicated entry point should use an import string so reload works."""
    import ems_prepared.adapters.fastapi.server as server

    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_run(*args: object, **kwargs: object) -> None:
        calls.append((args, kwargs))

    monkeypatch.setattr(server.uvicorn, "run", fake_run)
    monkeypatch.delenv(server._POLICY_ENV, raising=False)
    monkeypatch.delenv(server._LOCALE_ENV, raising=False)
    monkeypatch.delenv(server._EXPERIMENT_ENV, raising=False)

    code = server.main(
        [
            "--reload",
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
            "--policy",
            "agent",
            "--locale",
            "german",
            "--experiment",
            "exp_01",
        ]
    )

    assert code == 0
    assert calls == [
        (
            ("ems_prepared.adapters.fastapi.server:create_app_from_environment",),
            {
                "factory": True,
                "host": "0.0.0.0",
                "port": 9000,
                "reload": True,
            },
        )
    ]
    assert os.environ[server._POLICY_ENV] == "agent"
    assert os.environ[server._LOCALE_ENV] == "german"
    assert os.environ[server._EXPERIMENT_ENV] == "exp_01"


def test_create_app_from_environment_uses_runtime_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The uvicorn factory should read policy, locale, and experiment from env."""
    import ems_prepared.adapters.fastapi.server as server

    expected_app = object()
    seen: list[tuple[str, Locale, str | None]] = []

    def fake_create_runtime_app(
        *,
        policy: str,
        locale: Locale,
        experiment_name: str | None,
    ) -> object:
        seen.append((policy, locale, experiment_name))
        return expected_app

    monkeypatch.setattr(server, "create_runtime_app", fake_create_runtime_app)
    monkeypatch.setenv(server._POLICY_ENV, "agent")
    monkeypatch.setenv(server._LOCALE_ENV, Locale.DE.value)
    monkeypatch.setenv(server._EXPERIMENT_ENV, "exp_02")

    app = server.create_app_from_environment()

    assert app is expected_app
    assert seen == [("agent", Locale.DE, "exp_02")]
