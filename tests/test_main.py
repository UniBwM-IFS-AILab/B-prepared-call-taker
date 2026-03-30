"""Tests for the generic `main` launcher."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace

import pytest

import ems_prepared.main as launcher
from ems_prepared.model.context import Settings
from tests.fakes import FakeConversationPolicy


class _DummyFrontendPlugin:
    """Frontend plugin fake used to exercise subcommand dispatch."""

    def __init__(self, *, return_code: int = 0) -> None:
        self.return_code = return_code
        self.run_calls: list[tuple[object, Namespace]] = []

    def register_arguments(self, subparser: ArgumentParser) -> None:
        _ = subparser.add_argument("--frontend-value", default=None)

    def run(self, session_manager: object, parsed_args: Namespace) -> int:
        self.run_calls.append((session_manager, parsed_args))
        return self.return_code


async def _dummy_policy(_deps: Settings):
    return FakeConversationPolicy()


def test_main_lists_frontends_and_policies(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`list` should print discovered plugin names and exit cleanly."""
    monkeypatch.setattr(
        launcher,
        "load_frontend_plugins",
        lambda: {"cli": _DummyFrontendPlugin()},
    )
    monkeypatch.setattr(
        launcher,
        "load_policy_factories",
        lambda: {"graph": _dummy_policy},
    )

    code = launcher.main(["list"])
    output = capsys.readouterr().out

    assert code == 0
    assert "Frontends:" in output
    assert "- cli" in output
    assert "Policies:" in output
    assert "- graph" in output


def test_main_rejects_unknown_frontend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unknown frontend names should fail with parser error status."""
    monkeypatch.setattr(
        launcher,
        "load_frontend_plugins",
        lambda: {"cli": _DummyFrontendPlugin()},
    )
    monkeypatch.setattr(
        launcher,
        "load_policy_factories",
        lambda: {"graph": _dummy_policy},
    )

    with pytest.raises(SystemExit) as exc:
        _ = launcher.main(["missing"])

    assert exc.value.code == 2


def test_main_runs_selected_frontend_with_parsed_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Launcher should parse shared args and dispatch to selected frontend plugin."""
    sentinel_manager = object()
    fake_frontend = _DummyFrontendPlugin(return_code=7)

    monkeypatch.setattr(
        launcher,
        "load_frontend_plugins",
        lambda: {"cli": fake_frontend},
    )
    monkeypatch.setattr(
        launcher,
        "load_policy_factories",
        lambda: {"graph": _dummy_policy},
    )
    monkeypatch.setattr(
        launcher,
        "create_session_manager",
        lambda **kwargs: sentinel_manager,
    )

    code = launcher.main(["cli", "--frontend-value", "s1"])

    assert code == 7
    assert fake_frontend.run_calls
    manager, parsed_args = fake_frontend.run_calls[-1]
    assert manager is sentinel_manager
    assert parsed_args.frontend_value == "s1"
    assert parsed_args.policy == "graph"
    assert parsed_args.locale == "english"
    assert parsed_args.experiment_name is None


def test_main_parses_shared_policy_locale_and_experiment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shared flags should be parsed once by main."""
    sentinel_manager = object()
    fake_frontend = _DummyFrontendPlugin(return_code=0)

    monkeypatch.setattr(
        launcher,
        "load_frontend_plugins",
        lambda: {"cli": fake_frontend},
    )
    monkeypatch.setattr(
        launcher,
        "load_policy_factories",
        lambda: {"graph": _dummy_policy},
    )
    monkeypatch.setattr(
        launcher,
        "create_session_manager",
        lambda **kwargs: sentinel_manager,
    )

    code = launcher.main(
        [
            "cli",
            "--policy",
            "agent",
            "--locale",
            "german",
            "--experiment",
            "exp_shared",
            "--frontend-value",
            "s1",
        ]
    )

    assert code == 0
    assert fake_frontend.run_calls
    manager, parsed_args = fake_frontend.run_calls[-1]
    assert manager is sentinel_manager
    assert parsed_args.policy == "agent"
    assert parsed_args.locale == "german"
    assert parsed_args.experiment_name == "exp_shared"
    assert parsed_args.frontend_value == "s1"


def test_main_rejects_unknown_option_typo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown options should fail parser validation."""
    monkeypatch.setattr(
        launcher,
        "load_frontend_plugins",
        lambda: {"cli": _DummyFrontendPlugin()},
    )
    monkeypatch.setattr(
        launcher,
        "load_policy_factories",
        lambda: {"graph": _dummy_policy},
    )

    with pytest.raises(SystemExit) as exc:
        _ = launcher.main(["cli", "--polciy", "graph"])
    assert exc.value.code == 2


def test_main_subcommand_help_shows_shared_and_frontend_options(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`main <frontend> --help` should include shared and frontend-specific args."""
    monkeypatch.setattr(
        launcher,
        "load_frontend_plugins",
        lambda: {"cli": _DummyFrontendPlugin()},
    )
    monkeypatch.setattr(
        launcher,
        "load_policy_factories",
        lambda: {"graph": _dummy_policy},
    )

    with pytest.raises(SystemExit) as exc:
        _ = launcher.main(["cli", "--help"])
    output = capsys.readouterr().out

    assert exc.value.code == 0
    assert "--policy" in output
    assert "--locale" in output
    assert "--experiment" in output
    assert "--frontend-value" in output


def test_main_root_help_shows_shared_flags(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`main --help` should expose management command and shared session flags."""
    monkeypatch.setattr(
        launcher,
        "load_frontend_plugins",
        lambda: {"cli": _DummyFrontendPlugin()},
    )
    monkeypatch.setattr(
        launcher,
        "load_policy_factories",
        lambda: {"graph": _dummy_policy},
    )

    with pytest.raises(SystemExit) as exc:
        _ = launcher.main(["--help"])
    output = capsys.readouterr().out

    assert exc.value.code == 0
    assert "list" in output
    assert "--policy" in output
    assert "--locale" in output
    assert "--experiment" in output
