"""Setup for specialized loggers using Python's standard logging library.

Clean implementation using custom Formatter classes to handle extra fields.
Each logger writes to its own file with no stdout pollution.
"""

import logging
from datetime import datetime
from pathlib import Path


def flush_logger(logger: logging.Logger) -> None:
    """Flush all handlers of a logger."""
    for handler in logger.handlers:
        handler.flush()


def setup_console_logging(level: int = logging.INFO) -> None:
    """Setup console logging for application stdout messages.

    Configures logging for 'ems_prepared' logger only, filtering out third-party libraries.
    Safe to call multiple times - will skip if already configured.

    Args:
        level: Logging level (default: INFO)

    Usage:
        Call once at application startup:
        >>> setup_console_logging(logging.DEBUG)
        >>> logger = logging.getLogger(__name__)
        >>> logger.info("Application started")
    """
    # Use application-specific logger instead of root to avoid third-party noise
    app_logger = logging.getLogger("ems_prepared")

    # Check if already configured (avoid duplicate handlers)
    if any(isinstance(h, logging.StreamHandler) for h in app_logger.handlers):
        return

    # Configure app logger
    app_logger.setLevel(level)
    app_logger.propagate = False  # Don't propagate to root logger

    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Simple, readable format
    formatter = logging.Formatter("%(levelname)s - %(name)s - %(message)s")
    console_handler.setFormatter(formatter)

    app_logger.addHandler(console_handler)


class StateFormatter(logging.Formatter):
    """Formats state changes as JSONL with node, event, and message content."""

    def format(self, record):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Extract extra fields with defaults
        node = getattr(record, "node", "unknown")
        event = getattr(record, "event", "unknown")
        # The message contains the JSON content
        msg = record.getMessage()

        return f'{{"timestamp": "{timestamp}", "node": "{node}", "event": "{event}", {msg}}}'


class MessageFormatter(logging.Formatter):
    """Formats conversation messages as TSV with speaker and message."""

    def format(self, record):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Extract extra fields with defaults
        speaker = getattr(record, "speaker", "unknown")
        # Use msg_text to avoid collision with record.message
        message = getattr(record, "msg_text", "")

        return f"{timestamp}\t{speaker}\t{message}"


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
