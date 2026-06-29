from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def record_identity(record: Mapping[str, Any]) -> tuple[object, object]:
    if "dialog_id" not in record:
        raise KeyError(
            "Input record is missing required key 'dialog_id'; "
            "cannot isolate context per conversation."
        )
    turn_key = next((key for key in ("turn_count", "turn_index") if key in record), None)
    if turn_key is None:
        raise KeyError(
            "Input record is missing required key 'turn_count' or 'turn_index'; "
            "cannot deduplicate output rows."
        )
    return record["dialog_id"], record[turn_key]


def load_resume_checkpoint(output_path: Path) -> tuple[object, object] | None:
    last_line = _read_last_nonempty_line(output_path)
    if last_line is None:
        return None
    return record_identity(json.loads(last_line))


def _read_last_nonempty_line(output_path: Path, *, block_size: int = 8192) -> bytes | None:
    if not output_path.exists():
        return None

    with output_path.open("rb") as output_file:
        output_file.seek(0, 2)
        position = output_file.tell()
        if position == 0:
            return None

        tail_fragment = b""
        while position > 0:
            read_size = min(block_size, position)
            position -= read_size
            output_file.seek(position)
            chunk = output_file.read(read_size)
            fragments = (chunk + tail_fragment).split(b"\n")
            tail_fragment = fragments[0]
            for line in reversed(fragments[1:]):
                if line.strip():
                    return line

        return tail_fragment if tail_fragment.strip() else None
