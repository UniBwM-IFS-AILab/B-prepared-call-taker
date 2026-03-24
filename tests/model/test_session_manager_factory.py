"""Tests for shared backend session manager factory wiring."""

from __future__ import annotations

from ems_prepared.model.contracts import SessionManager
from ems_prepared.model.session_service import create_session_manager


def test_create_session_manager_returns_session_manager_contract() -> None:
    """Composition function should return an object satisfying SessionManager."""
    manager = create_session_manager()
    assert isinstance(manager, SessionManager)


def test_create_session_manager_returns_fresh_instances() -> None:
    """Composition function should create new manager instances per call."""
    first = create_session_manager()
    second = create_session_manager()
    assert first is not second
