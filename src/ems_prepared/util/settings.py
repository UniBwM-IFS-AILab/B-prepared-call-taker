import os
from datetime import datetime
from enum import Enum, auto
from functools import cached_property
from pathlib import Path
from typing import Awaitable, Callable
from uuid import UUID, uuid4

from loguru import logger
from pydantic import computed_field
from pydantic.fields import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from starlette.websockets import WebSocket

from ems_prepared.util.logger import setup_messages_logger, setup_state_logger


class InputMode(Enum):
    """Enumeration for input modes in the emergency call workflow.

    Defines whether input is collected via CLI or as a request.
    """

    CLI = auto()
    API = auto()
    TEST = auto()


class Locale(str, Enum):
    EN = "english"
    DE = "german"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(populate_by_name=True)

    # Logging
    name: str
    timestamp: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    user_id: UUID = Field(default_factory=uuid4)
    session_id: UUID = Field(default_factory=uuid4)
    experiment_name: str = Field(
        default="_sessions",
        description="Optional experiment/run name to group logs into a subdirectory. "
        "When provided, logs are saved to logs/{experiment_name}/{user_id}/{session_id}/. "
        "Can also be set via EXPERIMENT_NAME environment variable.",
        validation_alias="EXPERIMENT_NAME",
    )
    scenario_name: str | None = Field(
        default=None,
        description="Selected scenario filename for logging purposes.",
    )
    policy_name: str | None = Field(
        default=None,
        description="Resolved policy name ('graph' or 'agent') for this session.",
    )

    @property
    def save_path(self) -> Path:
        # Use experiment_name if provided, otherwise use "_sessions" to keep logs organized
        experiment_dir = self.experiment_name if self.experiment_name else "_sessions"
        save_path: Path = (
            Path("logs") / experiment_dir / self.user_id.hex / self.session_id.hex
        )
        save_path.mkdir(parents=True, exist_ok=True)
        os.makedirs(save_path, exist_ok=True)

        return save_path

    websocket: WebSocket | None = Field(default=None, repr=False)  # exclude=True,

    # @computed_field
    # @cached_property
    # def log_dir(self) -> Path:
    #     return Path("logs") / self.user_id.hex / self.session_id.hex

    @computed_field(repr=False)
    @cached_property
    def file_name(self) -> str:
        return f"{self.name}_{self.timestamp}"

    # Runtime
    locale: Locale = Locale.EN
    call_origin: InputMode = Field(default=InputMode.CLI, frozen=True)
    emit: Callable[[str], Awaitable[None]] | Callable[[str], None] = Field(
        exclude=True, repr=False, default=print
    )

    @cached_property
    def messages_logger(self):
        """Lazy-initialized messages logger.

        Created on first access and cached for the lifetime of the Settings instance.
        Works with both Pydantic models and dataclasses.
        """
        return setup_messages_logger(self.save_path)

    @cached_property
    def state_logger(self):
        """Lazy-initialized state change logger.

        Tracks state extraction and changes during graph execution.
        Created on first access and cached for the lifetime of the Settings instance.
        """
        return setup_state_logger(self.save_path)

    def model_post_init(self, __context) -> None:
        """Initialize file logging with loguru after model creation."""
        log_file = self.save_path / "stdout.log"
        logger.add(
            log_file,
            # level="DEBUG",
            enqueue=True,
            backtrace=True,
            diagnose=True,
            # format="{time} | {level} | {message}",
        )
