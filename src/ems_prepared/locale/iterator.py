import os
import typing as t
from collections.abc import Iterator, Mapping

JsonType: t.TypeAlias = list["JsonValue"] | Mapping[str, "JsonValue"]
JsonValue: t.TypeAlias = str | int | float | None | JsonType

import pycountry
import pyjson5 as json


class QuestionIterator:
    """Iterator that yields questions from data.json based on language and category."""

    def __init__(self, language: str, category: str):
        """Initialize the iterator.

        Args:
            language: Language code ('de', 'en') or language name ('German', 'English')
            category: Category name to iterate through

        """
        self.language_code: str = self._get_language_code(language)
        self.category: str = category
        self.data: JsonValue = self._load_data()
        self.questions: list[str] = self._get_questions()
        self.index: int = 0

    def _get_language_code(self, language: str) -> str:
        """Convert language input to ISO language code using pycountry."""
        language = language.strip()

        # Try to get language by alpha_2 code first
        try:
            lang = pycountry.languages.get(alpha_2=language.lower())
            if lang:
                return lang.alpha_2
        except KeyError:
            pass

        # Try to get language by name
        try:
            lang = pycountry.languages.get(name=language.title())
            if lang:
                return lang.alpha_2
        except KeyError:
            pass

        # Try common variations
        language_variations = {
            "german": "de",
            "deutsch": "de",
            "english": "en",
            "englisch": "en",
        }

        language_lower = language.lower()
        if language_lower in language_variations:
            return language_variations[language_lower]

        # If nothing works, raise an error
        raise ValueError(
            f"Language '{language}' not supported or not found in pycountry database"
        )

    def _load_data(self) -> JsonValue:
        """Load data from JSON file."""
        current_dir = os.path.dirname(__file__)
        data_path = os.path.join(current_dir, "questions.jsonc")

        with open(data_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _get_questions(self) -> list[str]:
        """Extract all questions for the specified language and category."""
        questions = []

        if self.language_code not in self.data:
            raise ValueError(f"Language '{self.language_code}' not available in data")

        language_data = self.data[self.language_code]

        if self.category not in language_data:
            raise ValueError(
                f"Category '{self.category}' not available for language '{self.language_code}'"
            )

        category_data: list[str] | dict[str, str] = language_data[self.category]

        if isinstance(category_data, list):
            # Direct list of questions
            questions.extend(category_data)
        elif isinstance(category_data, dict):
            # Dictionary with subcategories
            for subcategory, question_list in category_data.items():
                if isinstance(question_list, list):
                    questions.extend(question_list)

        return questions

    def __iter__(self) -> Iterator[str]:
        """Return iterator object."""
        self.index = 0
        return self

    def __next__(self) -> str:
        """Return next question."""
        if self.index >= len(self.questions):
            raise StopIteration

        question = self.questions[self.index]
        self.index += 1
        return question

    def __len__(self) -> int:
        """Return number of questions."""
        return len(self.questions)

    def get_available_categories(self) -> list[str]:
        """Get list of available categories for the current language."""
        if self.language_code in self.data:
            return list(self.data[self.language_code].keys())
        return []


def create_question_iterator(language: str, category: str) -> QuestionIterator:
    """
    Factory function to create a QuestionIterator.

    Args:
        language: Language code (e.g., 'de', 'en') or language name ('German', 'English')
        category: Category name to iterate through

    Returns:
        QuestionIterator instance
    """
    return QuestionIterator(language, category)
