from pathlib import Path
from ems_prepared.util.settings import Settings
from ems_prepared.dialogue_state.emergency_call_state import EmergencyCall
from pydantic_graph.graph import Graph
from pydantic_graph.persistence.file import FileStatePersistence
from pydantic_graph.persistence import StateT
from pydantic_graph.persistence import NodeSnapshot
from pydantic_graph.persistence import RunEndT

class ResumableFilePersistence(FileStatePersistence[StateT, RunEndT]):
    """A persistence layer that can resume from a partially completed state by duplicating last snapshot if it was faulty."""
    async def load_next(self):

        try:
            # First try the standard behavior (created → pending)
            snapshot = await super().load_next()
            if snapshot is not None:
                return snapshot
        except Exception as e:
            print(f"failed to resume, attempt to repair persistence file. Error: {e}")
        # No 'created' found; requeue the last pending/running as a fresh 'created'
        async with self._lock():  # uses the file lock as in base class
            snapshots = await self.load_all()
            for snapshot in reversed(snapshots):
                if isinstance(snapshot, NodeSnapshot) and snapshot.status in {"pending", "running", "error"}:
                    # Append a duplicate 'created' snapshot and return it
                    await self._append_save(NodeSnapshot(state=snapshot.state, node=snapshot.node), lock=False)
                    # Now the base logic will pick it on the next call:
                    break
        return await super().load_next()

async def setup_file_persistence(
    graph: Graph[EmergencyCall, Settings, EmergencyCall], save_path: Path
):
    save_path.mkdir(parents=True, exist_ok=True)
    persistence = ResumableFilePersistence[EmergencyCall, EmergencyCall](
        json_file=(save_path / "persistence.json")
    )
    if persistence.should_set_types():
        persistence.set_graph_types(graph)
    return persistence
