"""Scenario and chat utility helpers for Gradio frontends."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Literal, NotRequired, TypedDict

from gradio import ChatMessage


class ChatMessageMetadata(TypedDict, total=False):
    title: str
    id: str | int
    parent_id: str | int
    log: str
    duration: float
    status: Literal["pending", "done"]


class ChatMessageOption(TypedDict):
    key: str
    value: str


class ChatMessageDict(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: str
    metadata: NotRequired[ChatMessageMetadata]
    options: NotRequired[ChatMessageOption]


COMPLETION_MESSAGE = ChatMessage(
    role="assistant",
    content="The emergency call has been processed. Thank you.",
    metadata={"id": "completion_message"},
)


def get_scenario_directory(scenario_dir: str | None = None) -> Path:
    """Resolve the scenario directory from arg or environment."""
    if scenario_dir:
        path = Path(scenario_dir).resolve()
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
    """Return available Markdown scenario filenames."""
    path = get_scenario_directory(scenario_dir=scenario_dir)
    return sorted(
        scenario_file.name
        for scenario_file in path.glob("*.md")
        if not scenario_file.name.startswith("_")
    )


def get_random_scenario(scenario_dir: str | None = None) -> str | None:
    """Return one random scenario filename."""
    scenarios = list_md_files(scenario_dir=scenario_dir)
    return random.choice(scenarios) if scenarios else None


def read_md(filename: str | None, scenario_dir: str | None = None) -> str:
    """Read one markdown file from the scenario directory."""
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
    """Merge shared instructions with selected scenario content."""
    scenario_path = get_scenario_directory(scenario_dir=scenario_dir)
    instructions_path = scenario_path / "_Instructions.md"
    instructions = (
        read_md("_Instructions.md", scenario_dir=scenario_dir)
        if instructions_path.exists()
        else ""
    )

    scenario_content = read_md(filename, scenario_dir=scenario_dir)
    return f"{instructions}\n\n{scenario_content}" if instructions else scenario_content
