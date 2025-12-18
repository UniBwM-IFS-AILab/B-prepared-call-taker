"""Utilities for saving results."""

import json
from pathlib import Path

from pydantic_core import to_jsonable_python


# TODO: move other save functions here
def save_message_history_json(
    message_history: list,
    save_path: Path,
) -> None:
    """Save message history to JSON file."""
    message_history_file_path = save_path / "message_history.json"
    message_history_json = to_jsonable_python(message_history)
    _ = message_history_file_path.write_text(
        json.dumps(message_history_json), encoding="utf-8"
    )
