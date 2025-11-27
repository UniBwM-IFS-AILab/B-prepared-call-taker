from pathlib import Path
from typing import Any
from uuid import UUID

from pydantic.main import BaseModel
from pydantic_graph.graph import Graph
from pydantic_graph.persistence import NodeSnapshot, RunEndT, SnapshotStatus, StateT
from pydantic_graph.persistence.file import FileStatePersistence


class ResumableFilePersistence(FileStatePersistence[StateT, RunEndT]):
    """A persistence layer that can resume from a partially completed state by duplicating last snapshot if it was faulty.

    Also includes a fix for Python 3.7+ StopIteration/RuntimeError issue in _after_run_sync.
    """

    def _after_run_sync(
        self, snapshot_id: str, duration: float, status: SnapshotStatus
    ) -> None:
        """Override to fix StopIteration bug in pydantic_graph with Python 3.7+

        The base implementation uses next() which raises StopIteration when the snapshot
        is not found. In Python 3.7+, StopIteration raised inside a generator that's
        wrapped by run_in_executor causes a RuntimeError.
        """
        try:
            snapshots = self._load_sync()
            # Use a loop instead of next() to avoid StopIteration
            snapshot = None
            for s in snapshots:
                if s.id == snapshot_id:
                    snapshot = s
                    break

            if snapshot is None:
                # Snapshot not found - this shouldn't happen in normal operation
                print(f"Warning: Snapshot {snapshot_id} not found in persistence")
                return

            assert isinstance(snapshot, NodeSnapshot), (
                "Only NodeSnapshot can be recorded"
            )
            snapshot.duration = duration
            snapshot.status = status
            self._save_sync(snapshots)
        except Exception as e:
            # Log but don't raise to avoid breaking the graph execution
            print(f"Warning: Failed to update snapshot status: {e}")

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
                if isinstance(snapshot, NodeSnapshot) and snapshot.status in {
                    "pending",
                    "running",
                    "error",
                }:
                    # Append a duplicate 'created' snapshot and return it
                    await self._append_save(
                        NodeSnapshot(state=snapshot.state, node=snapshot.node),
                        lock=False,
                    )
                    # Now the base logic will pick it on the next call:
                    break
        return await super().load_next()


async def clear_old_run(user_id: UUID):
    if user_id.int == 0:
        import shutil

        print("Cleaning up old test runs for user_id=0")
        shutil.rmtree(
            Path(f"logs/{user_id.hex}"), ignore_errors=True
        )  # , ignore_errors=True


async def setup_resumable_file_persistence(
    graph: Graph[Any, Any, Any],
    save_path: Path,
    prefix: str = "",
) -> ResumableFilePersistence[Any, Any]:
    save_path.mkdir(parents=True, exist_ok=True)
    persistence = ResumableFilePersistence[BaseModel, BaseModel](
        json_file=(save_path / f"{prefix}persistence.json")
    )
    if persistence.should_set_types():
        persistence.set_graph_types(graph)
    return persistence


async def setup_file_persistence(
    graph: Graph[Any, Any, Any],
    save_path: Path,
    prefix: str = "",
) -> FileStatePersistence[Any, Any]:
    save_path.mkdir(parents=True, exist_ok=True)
    persistence = FileStatePersistence[BaseModel, BaseModel](
        json_file=(save_path / f"{prefix}persistence.json")
    )
    if persistence.should_set_types():
        persistence.set_graph_types(graph)
    return persistence
