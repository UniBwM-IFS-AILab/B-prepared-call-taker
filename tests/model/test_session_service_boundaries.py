"""Dependency boundary checks for the shared session service module."""

from __future__ import annotations

from pathlib import Path

import ems_prepared.model.session_service as session_service_module


def test_session_service_does_not_import_policy_persistence() -> None:
    """Session service must not depend on graph persistence internals."""
    module_path = Path(session_service_module.__file__)
    source = module_path.read_text(encoding="utf-8")

    assert "policies.pydantic_graph.custom_persistence" not in source
    assert "resumable_file_persistence" not in source
    assert "clear_old_run" not in source
