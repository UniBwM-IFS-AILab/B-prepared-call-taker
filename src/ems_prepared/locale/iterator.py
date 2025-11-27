import os

import pyjson5 as json


def load_all_questions() -> dict[str, dict[str, list[str]]]:
    """Load data from JSON file."""
    current_dir = os.path.dirname(__file__)
    data_path = os.path.join(current_dir, "questions.jsonc")

    with open(data_path, "r", encoding="utf-8") as file:
        return json.load(file)  # type: ignore


def load_questions(locale: str, question_set: str) -> list[str]:
    """Load data from JSON file."""
    current_dir = os.path.dirname(__file__)
    data_path = os.path.join(current_dir, "questions.jsonc")

    with open(data_path, "r", encoding="utf-8") as file:
        data = json.load(file)  # type: ignore
        questions = data.get(locale, None).get(question_set, None)

        if questions is None:
            raise ValueError(
                f"Questions not found for locale '{locale}' and question set '{question_set}'"
            )

        return questions
