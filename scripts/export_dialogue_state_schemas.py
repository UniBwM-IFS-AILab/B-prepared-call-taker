from __future__ import annotations

import json
from pathlib import Path

from pydantic import RootModel

from ems_prepared.dialogue_state.emergency_call_state import EmergencyCall
from ems_prepared.dialogue_state.generate_strict_model import (
    StrictEmergencyCall,
    inline_json_schema_refs,
)
from ems_prepared.dialogue_state.schema_variants import (
    slot_entry_type_from_model,
    slot_name_type_from_model,
)

SCHEMA_DIR = Path("schemas/dialogue_state")


class EmergencyCallSlotNames(
    RootModel[list[slot_name_type_from_model(EmergencyCall)]]  # pyrefly: ignore [not-a-type] # ty: ignore [invalid-type-form]
):
    pass


class EmergencyCallSlotEntries(
    RootModel[list[slot_entry_type_from_model(EmergencyCall)]]  # pyrefly: ignore [not-a-type] # ty: ignore [invalid-type-form]
):
    pass


def _write_schema(path: Path, schema: dict) -> None:
    path.write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n")


def export_dialogue_state_schemas(output_dir: Path = SCHEMA_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_schema(
        output_dir / "emergency_call.schema.json",
        inline_json_schema_refs(EmergencyCall.model_json_schema()),
    )
    _write_schema(
        output_dir / "strict_emergency_call.schema.json",
        inline_json_schema_refs(StrictEmergencyCall.model_json_schema()),
    )
    _write_schema(
        output_dir / "emergency_call_slot_names.schema.json",
        inline_json_schema_refs(EmergencyCallSlotNames.model_json_schema()),
    )
    _write_schema(
        output_dir / "emergency_call_slot_entries.schema.json",
        inline_json_schema_refs(EmergencyCallSlotEntries.model_json_schema()),
    )


if __name__ == "__main__":
    export_dialogue_state_schemas()
