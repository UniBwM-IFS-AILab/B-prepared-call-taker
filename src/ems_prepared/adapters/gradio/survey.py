"""Survey configuration and utilities for the Gradio emergency call simulator.

This module contains:
- Dataclasses for survey configuration (questions, scale labels)
- Default survey question set used by the Gradio adapter
"""

from dataclasses import dataclass

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

    id: int
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
        """Get choices formatted.

        Returns:
            List of (label, value) tuples where value is 1-indexed.
            Format: (display_label, numeric_value)

        """
        return [(label, idx) for idx, label in enumerate(self.labels, 1)]

    def get_choices_dict(self) -> dict[int, str]:
        """Get choices formatted.

        Returns:
            List of (label, value) tuples where value is 1-indexed.
            Format: (display_label, numeric_value)

        """
        return {idx: label for idx, label in enumerate(self.labels, 1)}


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


# Survey questions used by the Gradio flow
DEFAULT_SURVEY_QUESTIONS = (
    SurveyQuestion(
        id=1,
        category="understanding",
        label="Scenario Understanding",
        text="I always knew what I could say at each point in the dialogue.",
    ),
    SurveyQuestion(
        id=2,
        category="understanding",
        label="Agents' Understanding",
        text="The agent understood what I said.",  # The system understands the user's request and fulfils
    ),
    SurveyQuestion(
        id=3,
        category="efficiency",
        label="Relevance of messages",
        text="The agent's messages were relevant and focused on the situation.",
    ),
    SurveyQuestion(
        id=4,
        category="efficiency",
        label="Response Time",
        text="The agent responded promptly and avoided unnecessary delays during the conversation.",
    ),
    SurveyQuestion(
        id=5,
        category="naturalness",
        label="Agent adjustment to context",
        text="The agent's behavior felt appropriate for this conversation.",
    ),
    SurveyQuestion(
        id=6,
        category="naturalness",
        label="Human-like responses",
        text="The agent's messages felt realistic and fluent.",
    ),
    SurveyQuestion(
        id=7,
        category="overall",
        label="Overall Satisfaction",
        text="Overall, this conversation went well.",  #  https://aclanthology.org/P07-1100/
    ),
)

DEFAULT_SURVEY = SurveyConfig(
    labels=DEFAULT_LIKERT_LABELS,
    questions=DEFAULT_SURVEY_QUESTIONS,
)


DEFAULT_BY_LABEL: dict[str, SurveyQuestion] = {
    survey_question.label: survey_question
    for survey_question in DEFAULT_SURVEY_QUESTIONS
}
DEFAULT_BY_ID: dict[int, SurveyQuestion] = {
    survey_question.id: survey_question for survey_question in DEFAULT_SURVEY_QUESTIONS
}
