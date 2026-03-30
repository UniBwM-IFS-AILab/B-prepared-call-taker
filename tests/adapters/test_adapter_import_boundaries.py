"""Import-boundary checks for adapter modules."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ADAPTERS_ROOT = PROJECT_ROOT / "src/ems_prepared/adapters"

DISALLOWED_IMPORT_SNIPPETS = (
    "from ems_prepared.model.session_service import",
    "import ems_prepared.model.session_service",
    "from ems_prepared.model.policy_runtime",
    "import ems_prepared.model.policy_runtime",
)

ADAPTERS_USING_BOUNDARY_ERRORS = (
    ADAPTERS_ROOT / "fastapi/app.py",
    ADAPTERS_ROOT / "gradio/app.py",
    ADAPTERS_ROOT / "cli/app.py",
)


def test_adapters_do_not_import_backend_internal_modules() -> None:
    """Adapters should depend on model contracts/errors, not backend internals."""
    violations: list[str] = []
    for file_path in sorted(ADAPTERS_ROOT.rglob("*.py")):
        if file_path.name == "__init__.py":
            continue
        source = file_path.read_text(encoding="utf-8")
        for snippet in DISALLOWED_IMPORT_SNIPPETS:
            if snippet in source:
                relative = file_path.relative_to(PROJECT_ROOT)
                violations.append(f"{relative}: {snippet}")

    assert not violations, (
        "Adapters must not import backend internal modules directly. "
        f"Violations: {violations}"
    )


def test_adapters_import_shared_boundary_errors() -> None:
    """Adapters handling boundary errors should import them from `model.errors`."""
    for file_path in ADAPTERS_USING_BOUNDARY_ERRORS:
        source = file_path.read_text(encoding="utf-8")
        assert "from ems_prepared.model.errors import" in source
