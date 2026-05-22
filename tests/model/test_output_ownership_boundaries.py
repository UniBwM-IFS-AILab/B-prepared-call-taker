"""Boundary checks for canonical session output ownership."""

from __future__ import annotations

from pathlib import Path

from ems_prepared.model.session_backend_file import CANONICAL_SESSION_OUTPUT_FILES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLICIES_ROOT = PROJECT_ROOT / "src/ems_prepared/policies"
SESSION_BACKEND_FILE = PROJECT_ROOT / "src/ems_prepared/model/session_backend_file.py"
TCPR_SUBGRAPH_FILE = (
    PROJECT_ROOT / "src/ems_prepared/policies/pydantic_graph/tcpr_subgraph.py"
)


def test_policy_sources_do_not_write_canonical_session_output_files() -> None:
    """Policy modules should not directly target recorder-owned output files."""
    violations: list[str] = []
    for file_path in sorted(POLICIES_ROOT.rglob("*.py")):
        source = file_path.read_text(encoding="utf-8")
        for filename in CANONICAL_SESSION_OUTPUT_FILES:
            if filename in source:
                relative = file_path.relative_to(PROJECT_ROOT)
                violations.append(f"{relative}: {filename}")

    assert not violations, (
        "Canonical session outputs must be recorder-owned only. "
        f"Found policy-side references: {violations}"
    )


def test_session_backend_declares_canonical_output_files() -> None:
    """File backend implementation should own canonical output file names."""
    source = SESSION_BACKEND_FILE.read_text(encoding="utf-8")
    missing = [
        filename
        for filename in CANONICAL_SESSION_OUTPUT_FILES
        if filename not in source
    ]
    assert not missing, f"Missing canonical outputs in recorder module: {missing}"


def test_tcpr_subgraph_uses_shared_completion_artifact_helper() -> None:
    """TCPR flow should use shared completion helper instead of direct callback calls."""
    source = TCPR_SUBGRAPH_FILE.read_text(encoding="utf-8")
    assert "deps.record_completion_artifacts(" not in source
    assert "record_completion_artifacts(" in source
