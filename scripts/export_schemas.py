from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.dialog_act_labelling.dialogue_act.data_row_jedison_schema_patch import (
    data_row_jedison_schema_patch_schema,
)
from ems_prepared.dialogue_state.emergency_call_state import EmergencyCall
from ems_prepared.dialogue_state.generate_strict_model import (
    StrictEmergencyCall,
    inline_json_schema_refs,
)
from ems_prepared.dialogue_state.schema_variants import (
    EmergencyCallSlotEntries,
    EmergencyCallSlotNames,
)

SCHEMA_DIR = Path("schemas/dialogue_state")
DIALOG_ACT_SCHEMA_DIR = Path("schemas/annotations")


def _write_schema(path: Path, schema: dict) -> None:
    path.write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n")


def export_dialogue_state_schemas(output_dir: Path = SCHEMA_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    DIALOG_ACT_SCHEMA_DIR.mkdir(parents=True, exist_ok=True)

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
    _write_schema(
        DIALOG_ACT_SCHEMA_DIR / "dialogue_acts.schema.json",
        data_row_jedison_schema_patch_schema(),
    )


if __name__ == "__main__":
    export_dialogue_state_schemas()
