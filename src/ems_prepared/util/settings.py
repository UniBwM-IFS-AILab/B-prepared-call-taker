from datetime import datetime
from enum import Enum, auto
from functools import cached_property
from pathlib import Path
from typing import Awaitable, Callable
from uuid import UUID, uuid4

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
    EN = "en"
    DE = "de"


class Settings(BaseSettings):
    # Logging
    name: str
    timestamp: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    user_id: UUID = Field(default_factory=uuid4)
    session_id: UUID = Field(default_factory=uuid4)

    @property
    def save_path(self) -> Path:
        save_path: Path = Path("logs") / self.user_id.hex / self.session_id.hex
        save_path.mkdir(parents=True, exist_ok=True)
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
