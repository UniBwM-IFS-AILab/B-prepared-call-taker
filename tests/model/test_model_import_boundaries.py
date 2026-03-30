"""Import-boundary checks for core model modules."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = PROJECT_ROOT / "src/ems_prepared/model"

DISALLOWED_IMPORT_SNIPPETS = (
    "from ems_prepared.adapters",
    "import ems_prepared.adapters",
    "from ems_prepared.policies",
    "import ems_prepared.policies",
)


def test_model_modules_do_not_import_adapters_or_policies() -> None:
    """Core model modules should remain concrete-adapter/policy independent."""
    violations: list[str] = []
    for file_path in sorted(MODEL_ROOT.rglob("*.py")):
        if file_path.name == "__init__.py":
            continue
        source = file_path.read_text(encoding="utf-8")
        for snippet in DISALLOWED_IMPORT_SNIPPETS:
            if snippet in source:
                relative = file_path.relative_to(PROJECT_ROOT)
                violations.append(f"{relative}: {snippet}")

    assert not violations, (
        "Model layer must not import concrete adapters or concrete policies. "
        f"Violations: {violations}"
    )
