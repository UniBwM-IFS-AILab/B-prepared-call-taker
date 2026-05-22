"""Import-boundary checks for adapter modules."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ADAPTERS_ROOT = PROJECT_ROOT / "src/ems_prepared/adapters"
GRADIO_ROOT = ADAPTERS_ROOT / "gradio"

DISALLOWED_IMPORT_SNIPPETS = (
    "from ems_prepared.model.session_service import",
    "import ems_prepared.model.session_service",
    "from ems_prepared.model.policy_runtime",
    "import ems_prepared.model.policy_runtime",
)

ADAPTERS_USING_BOUNDARY_ERRORS = (
    ADAPTERS_ROOT / "fastapi/app.py",
    ADAPTERS_ROOT / "cli/app.py",
    ADAPTERS_ROOT / "gradio/flow.py",
    ADAPTERS_ROOT / "gradio/session_runtime.py",
    ADAPTERS_ROOT / "gradio/session_demo.py",
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


def test_adapters_use_builtin_boundary_errors() -> None:
    """Adapters should use builtin exceptions instead of model-specific custom errors."""
    for file_path in ADAPTERS_USING_BOUNDARY_ERRORS:
        source = file_path.read_text(encoding="utf-8")
        assert "from ems_prepared.model.errors import" not in source


def test_unified_gradio_ui_module_does_not_import_removed_shells_or_entrypoint() -> None:
    """Unified Gradio UI should depend only on shared helpers, not legacy shell modules."""
    session_demo_source = (GRADIO_ROOT / "session_demo.py").read_text(encoding="utf-8")

    assert "from ems_prepared.adapters.gradio.app import" not in session_demo_source
    assert "import ems_prepared.adapters.gradio.app" not in session_demo_source
    assert "from ems_prepared.adapters.gradio.standard import" not in session_demo_source
    assert "import ems_prepared.adapters.gradio.standard" not in session_demo_source
    assert "from ems_prepared.adapters.gradio.guided import" not in session_demo_source
    assert "import ems_prepared.adapters.gradio.guided" not in session_demo_source
