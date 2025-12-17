"""Setup for specialized loggers using Python's standard logging library.

Clean implementation using custom Formatter classes to handle extra fields.
Each logger writes to its own file with no stdout pollution.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Set, TypedDict

import rich.traceback
from pydantic import BaseModel
from rich.console import Console
from rich.logging import RichHandler

# Install pretty tracebacks for uncaught exceptions (optional but nice)
rich.traceback.install(show_locals=False)


def build_rich_console_handler() -> logging.Handler:
    handler = RichHandler(
        console=Console(file=sys.stdout),
        rich_tracebacks=True,
    )
    handler.setFormatter(logging.Formatter("%(funcName)s() - %(message)s"))
    return handler


def build_file_handler(log_path: Path, level: int = logging.DEBUG) -> logging.Handler:
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(
        logging.Formatter(
            (
                "%(asctime)s.%(msecs)03d\t"
                "%(levelname)-8s\t"
                "%(filename)s:%(funcName)s():%(lineno)d - "
                "%(message)s"
            ),
            "%Y-%m-%d %H:%M:%S",
        )
    )
    return file_handler


class StateLogEntry(TypedDict, total=False):
    timestamp: str
    event: str
    state: dict[str, Any]
    question: str
    response: str
    result: Any
    session: dict[str, Any]


class PydanticSanitizingFilter(logging.Filter):
    def __init__(self, fields_to_exclude: Iterable[str] = ()):
        super().__init__()
        # store once on the instance
        self.fields_to_exclude: Set[str] = set(fields_to_exclude)

    def _sanitize(self, obj: Any) -> Any:
        if isinstance(obj, BaseModel):
            return obj.model_dump(exclude=self.fields_to_exclude)
        elif isinstance(obj, dict):
            return {
                k: self._sanitize(v)
                for k, v in obj.items()
                if k not in self.fields_to_exclude
            }
        elif isinstance(obj, list):
            return [self._sanitize(item) for item in obj]
        else:
            return obj

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if not isinstance(args, tuple):
            return True

        new_args = []
        for arg in args:
            new_args.append(self._sanitize(arg))

        record.args = tuple(new_args)
        record.msg = self._sanitize(record.msg)
        return True


class StateFormatter(logging.Formatter):
    """Formats state changes as JSONL with timestamp, event, and payload content."""

    def format(self, record):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        event = getattr(record, "event", "unknown")

        if not isinstance(record.msg, Mapping):
            raise TypeError("state_logger expects a mapping payload")

        entry: StateLogEntry = {"timestamp": timestamp, "event": event}
        entry.update(record.msg)
        return json.dumps(entry, default=str, ensure_ascii=True)


class MessageFormatter(logging.Formatter):
    """Formats conversation messages as TSV with speaker and message."""

    def format(self, record):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Extract extra fields with defaults
        speaker = getattr(record, "speaker", "unknown")
        # Use msg_text to avoid collision with record.message
        message = getattr(record, "msg_text", "")

        return f"{timestamp}\t{speaker}\t{message}"


def flush_logger(logger: logging.Logger) -> None:
    """Flush all handlers of a logger."""
    for handler in logger.handlers:
        handler.flush()


def setup_session_logger(
    *,
    save_path: Path,
    session_id: str,
    level: int = logging.DEBUG,
) -> logging.Logger:
    log_path = save_path / "stdout.log"

    logger = logging.getLogger(f"ems.session.{session_id}")
    logger.setLevel(level)
    logger.propagate = False
    logger.handlers.clear()

    console_handler = build_rich_console_handler()
    file_handler = build_file_handler(log_path, level=level)

    # Optional: add filters here
    # console_handler.addFilter(PydanticSanitizingFilter(["message_history"]))
    # file_handler.addFilter(PydanticSanitizingFilter(["message_history"]))

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.info(f"Log Path: {save_path}")
    return logger


def setup_state_logger(save_path: Path):
    """Setup independent logger for state changes.

    Logs state after merges and extractions in JSONL format.

    Args:
        save_path: Directory for state_changes.jsonl

    Returns:
        Logger that writes to file only, no stdout

    Usage:
        logger.info('"state": {...}', extra={"node": "MergeState", "event": "merged"})
    """
    log_path = save_path / "state_changes.jsonl"

    # Unique logger per save_path
    logger = logging.getLogger(f"ems.state.{id(save_path)}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.handlers.clear()

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(StateFormatter())
    logger.addHandler(handler)
    logger.addFilter(PydanticSanitizingFilter(["message_history"]))

    return logger


def setup_messages_logger(save_path: Path):
    """Setup independent logger for conversation messages.

    Logs all operator questions and caller responses in TSV format.

    Args:
        save_path: Directory for messages.tsv

    Returns:
        Logger that writes to file only, no stdout

    Usage:
        logger.info("", extra={"speaker": "operator", "msg_text": "What's your location?"})
    """
    log_path = save_path / "messages.tsv"

    # Write header if new file
    if not log_path.exists():
        log_path.write_text("timestamp\tspeaker\tmessage\n", encoding="utf-8")

    # Unique logger per save_path
    logger = logging.getLogger(f"ems.messages.{id(save_path)}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.handlers.clear()

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(MessageFormatter())
    logger.addHandler(handler)

    return logger


def setup_global_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure 'ems' top-level logger with Rich; keep root mostly silent."""
    app_logger = logging.getLogger("ems")
    app_logger.setLevel(level)
    app_logger.propagate = False  # don't bubble to root
    app_logger.handlers.clear()

    console_handler = build_rich_console_handler()
    app_logger.addHandler(console_handler)

    # Keep root quiet / high level
    root = logging.getLogger()
    root.setLevel(logging.WARNING)
    # you can clear root.handlers if you had added something there
    # root.handlers.clear()

    return app_logger


setup_global_logging()
