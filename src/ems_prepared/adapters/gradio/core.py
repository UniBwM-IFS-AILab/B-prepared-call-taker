"""Shared Gradio helpers for chat typing and scenario file utilities."""

import os
import random
from pathlib import Path
from typing import Literal, TypedDict

from gradio import ChatMessage

try:
    from typing import NotRequired
except ImportError:
    from typing_extensions import NotRequired


# =============================================================================
# Type Definitions
# =============================================================================


class ChatMessageMetadata(TypedDict, total=False):
    """Metadata for ChatMessage."""

    title: str
    id: str | int
    parent_id: str | int
    log: str
    duration: float
    status: Literal["pending", "done"]


class ChatMessageOption(TypedDict):
    """Options for a ChatMessage."""

    key: str
    value: str


class ChatMessageDict(TypedDict):
    """TypedDict representation of a ChatMessage."""

    role: Literal["user", "assistant", "system"]
    content: str
    metadata: NotRequired[ChatMessageMetadata]
    options: NotRequired[ChatMessageOption]


# =============================================================================
# Constants
# =============================================================================

# Completion message shown when conversation ends
COMPLETION_MESSAGE = ChatMessage(
    content="The emergency call has been processed. Thank you.",
    metadata={"id": "completion_message"},
)


# =============================================================================
# Scenario Utilities
# =============================================================================


def get_scenario_directory(scenario_dir: str | None = None) -> Path:
    """Get the docs directory from environment variable or command line argument.

    Priority:
    1. `scenario_dir` argument
    2. Environment variable SCENARIO_DIR
    """
    if scenario_dir:
        path = Path(scenario_dir).resolve()
    # Fall back to environment variable
    elif env_scenario_dir := os.getenv("SCENARIO_DIR"):
        path = Path(env_scenario_dir).resolve()
    else:
        raise ValueError(
            "No scenario directory configured. Pass --scenario-dir or set SCENARIO_DIR."
        )

    if not path.exists() or not path.is_dir():
        raise ValueError(f"Scenario directory does not exist: {path}")
    return path


def list_md_files(scenario_dir: str | None = None) -> list[str]:
    """Return available Markdown filenames (base names)."""
    path = get_scenario_directory(scenario_dir=scenario_dir)
    return sorted(p.name for p in path.glob("*.md") if not p.name.startswith("_"))


def get_random_scenario(scenario_dir: str | None = None) -> str | None:
    """Get a random scenario filename from available scenarios."""
    scenarios = list_md_files(scenario_dir=scenario_dir)
    return random.choice(scenarios) if scenarios else None


def read_md(filename: str | None, scenario_dir: str | None = None) -> str:
    """Read a markdown file from the scenario directory.

    Args:
        filename: Name of the file to read (just the filename, not full path).
        scenario_dir: Optional scenario directory path override.

    Returns:
        Content of the file, or an error message if not found.

    """
    if not filename:
        return "### No file selected"
    path = get_scenario_directory(scenario_dir=scenario_dir) / filename
    if not path.exists():
        return f"### File not found: `{filename}`"
    return path.read_text(encoding="utf-8")


def construct_scenario_desc(
    filename: str | None,
    scenario_dir: str | None = None,
) -> str:
    """Merge the shared instructions with the selected scenario content."""
    scenario_path = get_scenario_directory(scenario_dir=scenario_dir)
    instructions_path = scenario_path / "_Instructions.md"
    instructions = (
        read_md("_Instructions.md", scenario_dir=scenario_dir)
        if instructions_path.exists()
        else ""
    )

    scenario_content = read_md(filename, scenario_dir=scenario_dir)
    return f"{instructions}\n\n{scenario_content}" if instructions else scenario_content
