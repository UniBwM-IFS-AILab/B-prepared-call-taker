"""Session-scoped runtime settings and aggregate dependency containers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from functools import cached_property
from pathlib import Path
from typing import Any, Awaitable, Callable
from uuid import UUID, uuid4

from pydantic import computed_field
from pydantic.fields import Field
from pydantic_settings import BaseSettings

from ems_prepared.util.logger import (
    setup_messages_logger,
    setup_session_logger,
    setup_state_logger,
)

type MaybeAwaitable = Awaitable[None] | None
type EmitCallable = Callable[["Settings", str], MaybeAwaitable]
type RequestInputCallable = Callable[["Settings", str], Awaitable[str]]
type RecordCompletionArtifactsCallable = Callable[[Any, list[Any]], None]
type LoadAgentSnapshotCallable = Callable[[], dict[str, Any] | None]
type SaveAgentSnapshotCallable = Callable[[dict[str, Any]], None]


class InputMode(Enum):
    """Enumeration for input modes in the emergency call workflow.

    Defines whether input is collected via CLI or as a request.
    """

    CLI = auto()
    API = auto()
    TEST = auto()


class Locale(str, Enum):
    """Supported operator/caller locales."""

    EN = "english"
    DE = "german"


async def _default_emit(deps: Settings, msg: str) -> None:
    """Fallback outbound transport used when no UI transport is configured."""
    deps.telemetry.logger.info(msg)


@dataclass(frozen=True)
class RunInfo:
    """Serializable session identity and naming metadata."""

    name: str
    timestamp: str
    user_id: UUID
    session_id: UUID
    experiment_name: str
    scenario_name: str | None
    policy_name: str | None

    @property
    def file_name(self) -> str:
        """Stable file stem used by older export paths."""
        return f"{self.name}_{self.timestamp}"


@dataclass(frozen=True)
class Transport:
    """Runtime-only outbound transport resources."""

    emit: EmitCallable
    request_input: RequestInputCallable | None = None


@dataclass(frozen=True)
class Storage:
    """Filesystem paths owned by a single session."""

    experiment_name: str
    user_id: UUID
    session_id: UUID

    @cached_property
    def logs_root(self) -> Path:
        """Base logs directory for this experiment context."""
        root: Path = (
            Path("experiments") / self.experiment_name / "logs"
            if self.experiment_name
            else Path("logs")
        )
        root.mkdir(parents=True, exist_ok=True)
        return root

    @cached_property
    def user_root(self) -> Path:
        """Directory containing all sessions for this user in one experiment."""
        root = self.logs_root / self.user_id.hex
        root.mkdir(parents=True, exist_ok=True)
        return root

    @cached_property
    def save_path(self) -> Path:
        """Directory used for all recorded outputs of a run."""
        save_path = self.user_root / self.session_id.hex
        save_path.mkdir(parents=True, exist_ok=True)
        return save_path


@dataclass(frozen=True)
class Telemetry:
    """Session-scoped loggers and derived telemetry handles."""

    save_path: Path
    session_id: UUID

    @cached_property
    def messages_logger(self):
        """Lazy conversation logger."""
        return setup_messages_logger(self.save_path)

    @cached_property
    def state_logger(self):
        """Lazy state transition logger."""
        return setup_state_logger(self.save_path)

    @cached_property
    def logger(self) -> logging.Logger:
        """Per-session stdout/file logger."""
        return setup_session_logger(
            save_path=self.save_path,
            session_id=self.session_id.hex,
        )


class Settings(BaseSettings):
    """Aggregate facade passed to graph nodes as the single deps object."""

    # Logging
    name: str
    timestamp: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    user_id: UUID = Field(default_factory=uuid4)
    session_id: UUID = Field(default_factory=uuid4)
    experiment_name: str = Field(
        default="",
        description="Optional experiment/run name to group logs into an experiment directory. "
        "When provided, logs are saved to experiments/{experiment_name}/logs/{user_id}/{session_id}/. "
        "When empty, logs are saved to logs/{user_id}/{session_id}/. "
        "Can also be set via EXPERIMENT_NAME environment variable.",
    )
    scenario_name: str | None = Field(
        default=None,
        description="Selected scenario filename for logging purposes.",
    )
    policy_name: str | None = Field(
        default=None,
        description="Resolved policy name ('graph' or 'agent') for this session.",
    )

    request_input: RequestInputCallable | None = Field(
        default=None,
        exclude=True,
        repr=False,
    )
    record_completion_artifacts: RecordCompletionArtifactsCallable | None = Field(
        default=None,
        exclude=True,
        repr=False,
    )
    load_agent_snapshot: LoadAgentSnapshotCallable | None = Field(
        default=None,
        exclude=True,
        repr=False,
    )
    save_agent_snapshot: SaveAgentSnapshotCallable | None = Field(
        default=None,
        exclude=True,
        repr=False,
    )

    # @computed_field
    # @cached_property
    # def log_dir(self) -> Path:
    #     return Path("logs") / self.user_id.hex / self.session_id.hex

    @computed_field(repr=False)
    def file_name(self) -> str:
        """Backward-compatible file stem used by existing exports."""
        return self.run_info.file_name

    # Runtime
    locale: Locale = Locale.EN
    call_origin: InputMode = Field(default=InputMode.CLI, frozen=True)

    async def async_print(self, msg: str):
        """Fallback helper for logging outbound messages during local runs."""
        self.telemetry.logger.info(msg)

    emit: EmitCallable = Field(  # type: ignore
        exclude=True, repr=False, default=_default_emit
    )

    @cached_property
    def run_info(self) -> RunInfo:
        """Structured access to serializable session identity data."""
        return RunInfo(
            name=self.name,
            timestamp=self.timestamp,
            user_id=self.user_id,
            session_id=self.session_id,
            experiment_name=self.experiment_name,
            scenario_name=self.scenario_name,
            policy_name=self.policy_name,
        )

    @cached_property
    def transport(self) -> Transport:
        """Structured access to outbound transport resources."""
        return Transport(emit=self.emit, request_input=self.request_input)

    @cached_property
    def storage(self) -> Storage:
        """Structured access to filesystem-backed storage paths."""
        return Storage(
            experiment_name=self.experiment_name,
            user_id=self.user_id,
            session_id=self.session_id,
        )

    @cached_property
    def telemetry(self) -> Telemetry:
        """Structured access to per-session loggers."""
        return Telemetry(save_path=self.storage.save_path, session_id=self.session_id)
