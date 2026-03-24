#!/usr/bin/env python
"""
List leaf directories under 'logs' and print metadata.scenario from survey.json if present.
Usage: python list_surveys.py [path/to/logs]
"""

import json
import sys
from pathlib import Path
from typing import Optional


def find_leaf_dirs(root: Path):
    for dirpath, dirnames, filenames in __import__("os").walk(root):
        if not dirnames:  # leaf directory
            yield Path(dirpath)


def read_scenario(survey_file: Path) -> Optional[str]:
    try:
        data = json.loads(survey_file.read_text(encoding="utf-8"))
        return data.get("metadata", {}).get("scenario")
    except Exception:
        return None


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("logs")
    if not root.exists():
        print(f"Logs root not found: {root}", file=sys.stderr)
        raise SystemExit(1)
    results = []
    for session_dir in find_leaf_dirs(root):
        survey = session_dir / "survey.json"
        if survey.exists():
            scenario = read_scenario(survey)
            results.append((session_dir.relative_to(root), scenario))

    # Sort by scenario name (case-insensitive). Put entries with no scenario last.
    results.sort(key=lambda item: (item[1] is None, (item[1] or "").lower()))

    for relpath, scenario in results:
        print(f"{relpath} -> scenario: {scenario!r}")


if __name__ == "__main__":
    main()
