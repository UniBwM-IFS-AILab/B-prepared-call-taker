"""Survey configuration and utilities for the Gradio emergency call simulator.

This module contains:
- Dataclasses for survey configuration (questions, scale labels)
- Default survey configuration based on Evaluation_Survey.md
- Survey response persistence (save to JSON file)
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# =============================================================================
# Survey Configuration Dataclasses
# =============================================================================


@dataclass(frozen=True)
class SurveyQuestion:
    """A single survey question definition.

    Attributes:
        category: category for the question (used in logging).
        label: Short label for display/reference.
        text: Full question text shown to the user.
    """

    category: str
    label: str
    text: str


@dataclass(frozen=True)
class SurveyConfig:
    """Configuration for a Likert scale survey.

    Attributes:
        labels: Tuple of scale labels from lowest to highest
            (e.g., ("Strongly Disagree", ..., "Strongly Agree")).
            The number of options is derived from len(labels).
        questions: Tuple of SurveyQuestion instances.
    """

    labels: tuple[str, ...]
    questions: tuple[SurveyQuestion, ...]

    @property
    def num_options(self) -> int:
        """Number of options in the Likert scale."""
        return len(self.labels)

    def get_choices(self) -> list[tuple[str, int]]:
        """Get choices formatted for gr.Radio component.

        Returns:
            List of (label, value) tuples where value is 1-indexed.
            Format: (display_label, numeric_value)
        """
        return [(label, i + 1) for i, label in enumerate(self.labels)]


# =============================================================================
# Default Survey Configuration
# =============================================================================

# 5-point Likert scale labels
DEFAULT_LIKERT_LABELS = (
    "Strongly Disagree",
    "Disagree",
    "Neutral",
    "Agree",
    "Strongly Agree",
)


# Survey questions from docs/Evaluation_Survey.md
DEFAULT_SURVEY_QUESTIONS = (
    SurveyQuestion(
        category="understanding",
        label="Scenario Understanding",
        text="I always knew what I could say at each point in the dialogue.",
    ),
    SurveyQuestion(
        category="understanding",
        label="Agents' Understanding",
        text="The agent understood what I said.",  # The system understands the user’s request and fulfils
    ),
    # SurveyQuestion(
    #     category="understanding",
    #     label="Clarity of Communication",
    #     text="The agent's messages were clear and easy to understand.",
    # ),
    SurveyQuestion(
        category="efficiency",
        label="Relevance of messages",
        text="The agent's messages were relevant and focused on the situation.",
    ),
    SurveyQuestion(
        category="efficiency",
        label="Response Time",
        text="The agent responded promptly and avoided unnecessary delays during the conversation.",
    ),
    # SurveyQuestion(
    #     category="efficiency",
    #     label="Coherence of the dialogue",
    #     text="I always knew what I could say at each point in the dialogue.",
    # ),
    SurveyQuestion(
        category="naturalness",
        label="Agent adjustment to context",
        text="The agent's behavior felt appropriate for this conversation.",
    ),
    SurveyQuestion(
        category="naturalness",
        label="Human-like responses",
        text="The agent's messages felt realistic and fluent.",
    ),
    SurveyQuestion(
        category="overall",
        label="Overall Satisfaction",
        text="Overall, this conversation went well.",  #  https://aclanthology.org/P07-1100/
    ),
)

DEFAULT_SURVEY = SurveyConfig(
    labels=DEFAULT_LIKERT_LABELS,
    questions=DEFAULT_SURVEY_QUESTIONS,
)


def save_survey_responses(
    save_path: Path,
    responses: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save survey responses to a JSON file.

    Creates a single survey.json file in the session directory.
    Overwrites any existing file (one survey per session).

    Args:
        save_path: Session-specific directory path.
        responses: Dictionary mapping question label to response data.
        metadata: Optional metadata (user_id, session_id, scenario, etc.).

    Returns:
        Path to the saved survey.json file.
    """
    survey_data = {
        "timestamp": datetime.now().isoformat(),
        "responses": responses,
    }

    if metadata:
        survey_data["metadata"] = metadata

    file_path = save_path / "survey.json"
    file_path.write_text(
        json.dumps(survey_data, indent=2, default=str), encoding="utf-8"
    )

    return file_path
