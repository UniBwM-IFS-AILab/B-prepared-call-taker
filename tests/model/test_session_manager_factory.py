"""Tests for shared backend session manager factory wiring."""

from __future__ import annotations

from ems_prepared.main import create_session_manager
from ems_prepared.model.contracts import SessionManager


async def _fake_policy_factory(_deps):
    from tests.fakes import FakeConversationPolicy

    return FakeConversationPolicy()


def test_create_session_manager_returns_session_manager_contract() -> None:
    """Factory helper should return an object satisfying SessionManager."""
    manager = create_session_manager(policy_factories={"fake": _fake_policy_factory})
    assert isinstance(manager, SessionManager)


def test_create_session_manager_returns_fresh_instances() -> None:
    """Factory helper should create new manager instances per call."""
    first = create_session_manager(policy_factories={"fake": _fake_policy_factory})
    second = create_session_manager(policy_factories={"fake": _fake_policy_factory})
    assert first is not second
