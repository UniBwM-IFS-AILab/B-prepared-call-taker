from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "anonymize_script.py"


class FakeEntity:
    """Minimal spaCy-style entity stub."""

    def __init__(self, start_char: int, end_char: int, label_: str) -> None:
        self.start_char = start_char
        self.end_char = end_char
        self.label_ = label_


class FakeDoc:
    """Minimal spaCy-style doc stub."""

    def __init__(self, entities: list[FakeEntity]) -> None:
        self.ents = entities


class FakeSpaCyPipeline:
    """Return preconfigured entity docs for each input text."""

    def __init__(self, entities_by_text: dict[str, list[tuple[int, int, str]]]) -> None:
        self.entities_by_text = entities_by_text
        self.calls: list[list[str]] = []

    def pipe(self, texts: list[str], batch_size: int) -> list[FakeDoc]:
        del batch_size
        self.calls.append(list(texts))
        docs: list[FakeDoc] = []
        for text in texts:
            docs.append(
                FakeDoc(
                    [
                        FakeEntity(start, end, label)
                        for start, end, label in self.entities_by_text.get(text, [])
                    ]
                )
            )
        return docs


@pytest.fixture
def anonymize_script(monkeypatch: pytest.MonkeyPatch):
    """Load the script as a module with lightweight fake dependencies."""
    spec = importlib.util.spec_from_file_location(
        "tests.scripts.anonymize_script_under_test",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module,
        "load_common_name_gazetteer",
        lambda: frozenset({"emilia", "nico", "simone"}),
    )
    monkeypatch.setattr(
        module,
        "load_german_location_gazetteer",
        lambda: frozenset({"kapuzinerplatz", "munich", "münchen", "neubiberg"}),
    )
    return module


def test_extract_exact_entity_texts_ignores_existing_placeholder(
    anonymize_script,
) -> None:
    """A later model stage must not turn [LOCATION] into a fresh LOCATION candidate."""
    assert (
        anonymize_script.extract_exact_entity_texts(
            "[LOCATION]",
            0,
            len("[LOCATION]"),
            anonymize_script.LOCATION_REPLACEMENT,
        )
        == []
    )


def test_apply_exact_entities_does_not_double_redact_placeholder(
    anonymize_script,
) -> None:
    """Exact replacement should not match inside an existing placeholder."""
    text = "Caller already said [LOCATION]."
    entities = [
        anonymize_script.KnownEntity(
            text="LOCATION",
            replacement=anonymize_script.LOCATION_REPLACEMENT,
        )
    ]

    assert anonymize_script.apply_exact_entities(text, entities) == text


def test_build_entities_uses_spacy_on_caller_name_state_text(
    anonymize_script,
) -> None:
    """caller_name participates only as ordinary source text for spaCy discovery."""
    full_name = "Brigitte Weber"
    message = "Brigitte Weber needs help."
    pipeline = FakeSpaCyPipeline({full_name: [(0, len(full_name), "PERSON")]})
    turns = [
        anonymize_script.TranscriptTurn(
            index=0,
            speaker="caller",
            message=message,
            state={"caller_name": full_name},
        )
    ]
    anonymizer = anonymize_script.TranscriptAnonymizer(
        spacy_pipelines=(
            anonymize_script.SpaCyPipelineHandle(name="fake_en", nlp=pipeline),
        ),
    )

    person_entities, _ = anonymizer.build_entities(turns)
    anonymized_turns = anonymize_script.anonymize_turns(
        turns,
        person_entities=person_entities,
        location_entities=[],
    )

    assert full_name in [entity.text for entity in person_entities]
    assert anonymized_turns[0].message == "{{Brigitte Weber}} needs help."
    assert anonymized_turns[0].state == {"caller_name": "{{Brigitte Weber}}"}


def test_build_entities_uses_spacy_on_emergency_location_state_text(
    anonymize_script,
) -> None:
    """emergency_location participates only as ordinary source text for spaCy discovery."""
    exact_location = "BMW Welt, Am Olympiapark 1"
    pipeline = FakeSpaCyPipeline({exact_location: [(0, len(exact_location), "LOC")]})
    turns = [
        anonymize_script.TranscriptTurn(
            index=0,
            speaker="caller",
            message=(
                "The patient is at BMW Welt, Am Olympiapark 1 and "
                "we are waiting outside."
            ),
            state={"emergency_location": exact_location},
        )
    ]
    anonymizer = anonymize_script.TranscriptAnonymizer(
        spacy_pipelines=(
            anonymize_script.SpaCyPipelineHandle(name="fake_en", nlp=pipeline),
        ),
    )

    _, location_entities = anonymizer.build_entities(turns)
    anonymized_turns = anonymize_script.anonymize_turns(
        turns,
        person_entities=[],
        location_entities=location_entities,
    )

    assert [entity.text for entity in location_entities][:1] == [exact_location]
    assert anonymized_turns[0].message == (
        "The patient is at [[BMW Welt, Am Olympiapark 1]] and we are "
        "waiting outside."
    )
    assert anonymized_turns[0].state == {
        "emergency_location": "[[BMW Welt, Am Olympiapark 1]]"
    }


def test_build_entities_uses_name_gazetteer_anywhere_else(anonymize_script) -> None:
    """Common names outside caller_name should still be redacted via the gazetteer."""
    turns = [
        anonymize_script.TranscriptTurn(
            index=0,
            speaker="caller",
            message="My friend Emilia called and Emilia cannot walk.",
            state={"caller_name": "Nico Michel"},
        )
    ]
    anonymizer = anonymize_script.TranscriptAnonymizer()

    person_entities, _ = anonymizer.build_entities(turns)
    anonymized_turns = anonymize_script.anonymize_turns(
        turns,
        person_entities=person_entities,
        location_entities=[],
    )

    assert "Emilia" in [entity.text for entity in person_entities]
    assert anonymized_turns[0].message == (
        "My friend {{Emilia}} called and {{Emilia}} cannot walk."
    )


def test_build_entities_uses_location_gazetteer_anywhere_else(
    anonymize_script,
) -> None:
    """Known German locations outside emergency_location should be redacted via the gazetteer."""
    turns = [
        anonymize_script.TranscriptTurn(
            index=0,
            speaker="caller",
            message="She said she is somewhere around Neubiberg near the station.",
            state={"caller_name": "Nico Michel"},
        )
    ]
    anonymizer = anonymize_script.TranscriptAnonymizer()

    _, location_entities = anonymizer.build_entities(turns)
    anonymized_turns = anonymize_script.anonymize_turns(
        turns,
        person_entities=[],
        location_entities=location_entities,
    )

    assert "Neubiberg" in [entity.text for entity in location_entities]
    assert anonymized_turns[0].message == (
        "She said she is somewhere around [[Neubiberg]] near the station."
    )


def test_build_entities_uses_spacy_for_unknown_person(
    anonymize_script,
) -> None:
    """spaCy should resolve unknown person names after gazetteer masking."""
    message = "Patient Brigitte Weber cannot walk."
    pipeline = FakeSpaCyPipeline({message: [(8, 23, "PERSON")]})
    turns = [
        anonymize_script.TranscriptTurn(
            index=0,
            speaker="caller",
            message=message,
            state={"caller_name": "Nico Michel"},
        )
    ]
    anonymizer = anonymize_script.TranscriptAnonymizer(
        spacy_pipelines=(
            anonymize_script.SpaCyPipelineHandle(name="fake_en", nlp=pipeline),
        ),
    )

    person_entities, _ = anonymizer.build_entities(turns)
    anonymized_turns = anonymize_script.anonymize_turns(
        turns,
        person_entities=person_entities,
        location_entities=[],
    )

    assert "Brigitte Weber" in [entity.text for entity in person_entities]
    assert anonymized_turns[0].message == "Patient {{Brigitte Weber}} cannot walk."
    assert all(message in batch for batch in pipeline.calls)


def test_build_entities_does_not_redact_pronouns_from_spacy(
    anonymize_script,
) -> None:
    """spaCy PERSON mislabels like 'she' must be rejected by the post-filter."""
    message = "She cannot walk."
    pipeline = FakeSpaCyPipeline({message: [(0, 3, "PERSON")]})
    turns = [
        anonymize_script.TranscriptTurn(
            index=0,
            speaker="caller",
            message=message,
            state={"caller_name": "Nico Michel"},
        )
    ]
    anonymizer = anonymize_script.TranscriptAnonymizer(
        spacy_pipelines=(
            anonymize_script.SpaCyPipelineHandle(name="fake_en", nlp=pipeline),
        ),
    )

    person_entities, _ = anonymizer.build_entities(turns)
    anonymized_turns = anonymize_script.anonymize_turns(
        turns,
        person_entities=person_entities,
        location_entities=[],
    )

    assert "She" not in [entity.text for entity in person_entities]
    assert anonymized_turns[0].message == message


def test_build_entities_ignores_operator_questions_for_location_discovery(
    anonymize_script,
) -> None:
    """Location discovery should not inspect operator prompts like intake questions."""
    operator_message = "With whom am I speaking, please?"
    caller_message = "My name is Nico Michel."
    pipeline = FakeSpaCyPipeline({operator_message: [(0, 9, "LOC")]})
    turns = [
        anonymize_script.TranscriptTurn(
            index=0,
            speaker="operator",
            message=operator_message,
            state=None,
        ),
        anonymize_script.TranscriptTurn(
            index=1,
            speaker="caller",
            message=caller_message,
            state={"caller_name": "Nico Michel"},
        ),
    ]
    anonymizer = anonymize_script.TranscriptAnonymizer(
        spacy_pipelines=(
            anonymize_script.SpaCyPipelineHandle(name="fake_en", nlp=pipeline),
        ),
    )

    _, location_entities = anonymizer.build_entities(turns)
    anonymized_turns = anonymize_script.anonymize_turns(
        turns,
        person_entities=[],
        location_entities=location_entities,
    )

    assert not location_entities
    assert anonymized_turns[0].message == operator_message
    assert all(operator_message not in batch for batch in pipeline.calls)


def test_build_entities_does_not_keep_generic_possessive_place_location(
    anonymize_script,
) -> None:
    """Generic relational places like 'my mums place' are not specific locations."""
    message = "She's at my mums place."
    turns = [
        anonymize_script.TranscriptTurn(
            index=0,
            speaker="caller",
            message=message,
            state={"emergency_location": "my mums place"},
        )
    ]
    anonymizer = anonymize_script.TranscriptAnonymizer()

    _, location_entities = anonymizer.build_entities(turns)
    anonymized_turns = anonymize_script.anonymize_turns(
        turns,
        person_entities=[],
        location_entities=location_entities,
    )

    assert "my mums place" not in [entity.text for entity in location_entities]
    assert anonymized_turns[0].message == message
    assert anonymized_turns[0].state == {"emergency_location": "my mums place"}


def test_build_entities_does_not_redact_unknown_location_without_gazetteer_or_spacy(
    anonymize_script,
) -> None:
    """Unknown non-gazetteer locations should not be redacted without spaCy support."""
    message = "She fell somewhere in the Preacher first forest area in the south of munich."
    turns = [
        anonymize_script.TranscriptTurn(
            index=0,
            speaker="caller",
            message=message,
            state={"caller_name": "Nico Michel"},
        )
    ]
    anonymizer = anonymize_script.TranscriptAnonymizer()

    _, location_entities = anonymizer.build_entities(turns)
    anonymized_turns = anonymize_script.anonymize_turns(
        turns,
        person_entities=[],
        location_entities=location_entities,
    )

    assert "munich" in [entity.text for entity in location_entities]
    assert "Preacher first forest" not in [entity.text for entity in location_entities]
    assert anonymized_turns[0].message == (
        "She fell somewhere in the Preacher first forest area in the south of "
        "[[munich]]."
    )
