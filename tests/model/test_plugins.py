"""Tests for plugin discovery helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

import ems_prepared.plugins as plugins
from ems_prepared.model.context import Settings
from tests.fakes import FakeConversationPolicy


@dataclass(frozen=True)
class _FakeEntryPoint:
    name: str
    _obj: object

    def load(self) -> object:
        return self._obj


class _FakeFrontendPlugin:
    """Minimal frontend plugin implementation for discovery tests."""

    def register_arguments(self, subparser) -> None:
        _ = subparser

    def run(self, session_manager, parsed_args) -> int:
        _ = (session_manager, parsed_args)
        return 0


def test_load_policy_factories_loads_group_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Policy discovery should return a name->callable mapping."""

    async def fake_policy_factory(_deps: Settings):
        return FakeConversationPolicy()

    monkeypatch.setattr(
        plugins,
        "entry_points",
        lambda *, group: (
            [_FakeEntryPoint("fake", fake_policy_factory)]
            if group == "ems_prepared.policies"
            else []
        ),
    )

    loaded = plugins.load_policy_factories()

    assert list(loaded.keys()) == ["fake"]
    assert loaded["fake"] is fake_policy_factory


def test_load_policy_factories_rejects_non_callable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Policy discovery should reject entry points resolving to non-callables."""
    monkeypatch.setattr(
        plugins,
        "entry_points",
        lambda *, group: (
            [_FakeEntryPoint("bad_policy", object())]
            if group == "ems_prepared.policies"
            else []
        ),
    )

    with pytest.raises(TypeError, match="must resolve to a callable"):
        _ = plugins.load_policy_factories()


def test_load_policy_factories_accepts_callable_without_arity_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Policy discovery should not validate callable arity at load time."""

    async def invalid_policy_factory() -> FakeConversationPolicy:
        return FakeConversationPolicy()

    monkeypatch.setattr(
        plugins,
        "entry_points",
        lambda *, group: (
            [_FakeEntryPoint("invalid", invalid_policy_factory)]
            if group == "ems_prepared.policies"
            else []
        ),
    )

    loaded = plugins.load_policy_factories()
    assert list(loaded.keys()) == ["invalid"]
    assert loaded["invalid"] is invalid_policy_factory


def test_load_group_rejects_duplicate_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """Duplicate entry-point names in one group should fail fast."""

    async def policy_factory(_deps: Settings):
        return FakeConversationPolicy()

    monkeypatch.setattr(
        plugins,
        "entry_points",
        lambda *, group: (
            [
                _FakeEntryPoint("dup", policy_factory),
                _FakeEntryPoint("dup", policy_factory),
            ]
            if group == "ems_prepared.policies"
            else []
        ),
    )

    with pytest.raises(ValueError, match="Duplicate entry point name"):
        _ = plugins.load_policy_factories()


def test_load_frontend_plugins_loads_group_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Frontend discovery should instantiate class entry points."""
    monkeypatch.setattr(
        plugins,
        "entry_points",
        lambda *, group: (
            [_FakeEntryPoint("fake", _FakeFrontendPlugin)]
            if group == "ems_prepared.frontends"
            else []
        ),
    )

    loaded = plugins.load_frontend_plugins()

    assert list(loaded.keys()) == ["fake"]
    assert isinstance(loaded["fake"], _FakeFrontendPlugin)


def test_load_frontend_plugins_rejects_non_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Frontend discovery should reject objects that do not match the contract."""
    monkeypatch.setattr(
        plugins,
        "entry_points",
        lambda *, group: (
            [_FakeEntryPoint("bad_frontend", object())]
            if group == "ems_prepared.frontends"
            else []
        ),
    )

    with pytest.raises(TypeError, match="must satisfy protocol 'FrontendPlugin'"):
        _ = plugins.load_frontend_plugins()
