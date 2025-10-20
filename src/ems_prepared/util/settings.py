from datetime import datetime
from enum import Enum, auto
from functools import cached_property
from pathlib import Path
from uuid import UUID, uuid4

from pydantic import computed_field
from pydantic.fields import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from starlette.websockets import WebSocket

LOCALE: str = "en"  # "de"  # "en"


class InputMode(Enum):
    """Enumeration for input modes in the emergency call workflow.

    Defines whether input is collected via CLI or as a request.
    """

    CLI = auto()
    API = auto()
    TEST = auto()


class Settings(BaseSettings):
    # Logging
    name: str
    timestamp: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    user_id: UUID = uuid4()
    session_id: UUID = uuid4()
    save_path: Path = Path("logs") / user_id.hex / session_id.hex

    websocket: WebSocket | None = Field(default=None, exclude=True, repr=False)

    @computed_field
    @cached_property
    def log_dir(self) -> Path:
        return Path("logs") / self.user_id.hex / self.session_id.hex

    @computed_field
    @cached_property
    def file_name(self) -> str:
        return f"{self.name}_{self.timestamp}"

    # Runtime
    locale: str = "en"  # TODO: replace with enum & correct localization system
    call_origin: InputMode = InputMode.CLI
