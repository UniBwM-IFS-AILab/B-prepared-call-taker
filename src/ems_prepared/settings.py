from datetime import datetime
from enum import Enum, auto
from functools import cached_property
from pathlib import Path

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

LOCALE: str = "en"  # "de"  # "en"


class RunMode(Enum):
    """Enumeration for input modes in the emergency call workflow.

    Defines whether input is collected via CLI or as a request.
    """

    CLI = auto()
    MAIN = auto()
    TEST = auto()


class Settings(BaseSettings):
    # Logging
    graph_name: str
    timestamp: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    @computed_field
    @cached_property
    def log_dir(self) -> Path:
        return Path("logs") / self.graph_name / self.timestamp

    @computed_field
    @cached_property
    def file_name(self) -> str:
        return f"{self.graph_name}_{self.timestamp}"

    # Runtime
    locale: str = "en"  # TODO: replace with enum & correct localization system
    call_origin: RunMode = RunMode.MAIN
