#!/usr/bin/env -S uv run --script

# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "Faker>=40.23.0",
#   "geonamescache>=3.0.1",
#   "spacy>=3.8.0,<4.0.0",
#   "en-core-web-trf @ https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.8.0/en_core_web_trf-3.8.0-py3-none-any.whl",
#   "de-core-news-lg @ https://github.com/explosion/spacy-models/releases/download/de_core_news_lg-3.8.0/de_core_news_lg-3.8.0-py3-none-any.whl",
# ]
# ///

"""Parse log files into anonymized transcript/survey export bundles.

For each dialogue, the script writes:

- ``call_N/transcript.json`` with one JSON array of transcript turns
- ``call_N/survey.json`` containing only the survey ``responses`` object,
  with timestamps and id-like fields removed, when a survey is present
- ``metadata.json`` at the output root linking each exported call directory
  to its scenario

Each transcript item is a JSON object with:

- ``index``: turn index in the transcript
- ``speaker``: ``operator`` or ``caller``
- ``message``: anonymized turn text
- ``state``: anonymized merged state attached to that turn, or ``null``
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import re
import sys
import traceback
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

INPUT_GLOBS: tuple[str, ...] = ("**/*.log",)
SPACY_BATCH_SIZE = 32

PERSON_REPLACEMENT = "[PERSON]"
LOCATION_REPLACEMENT = "[LOCATION]"
REPLACEMENT_PLACEHOLDERS = frozenset({PERSON_REPLACEMENT, LOCATION_REPLACEMENT})
PERSON_REVIEW_MARKER_PREFIX = "{{"
PERSON_REVIEW_MARKER_SUFFIX = "}}"
LOCATION_REVIEW_MARKER_PREFIX = "[["
LOCATION_REVIEW_MARKER_SUFFIX = "]]"
REVIEW_ANNOTATION_PATTERN = re.compile(r"\[\[[^\]]+\]\]|\{\{[^}]+\}\}")
SPACY_PERSON_LABELS = {"PER", "PERSON"}
SPACY_LOCATION_LABELS = {"GPE", "LOC"}
SURVEY_DROP_KEYS = {"timestamp", "id", "user_id", "session_id"}
POLICY_LABELS = {
    "agent": "LLM-only",
    "graph": "FSM+LLM",
}
TRANSIENT_STATE_KEYS = {
    "cpr_needed",
    "rd1",
    "rd2",
    "time_critical",
}

ENUM_REPR_PATTERN = re.compile(r"<[^:<>]+:\s*'([^']*)'>")
LOG_TIMESTAMP_PATTERN = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
)
GAZETTEER_TOKEN_PATTERN = re.compile(r"[A-Za-zÀ-ÿ0-9'.-]+")
MAX_GAZETTEER_NGRAM_TOKENS = 5
LOG_FIELD_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "operator",
        re.compile(r".*?Streaming (?:message|item):\s*(?P<text>.*)$"),
    ),
    (
        "caller",
        re.compile(r".*?User message:\s*(?P<text>.*)$"),
    ),
    (
        "merged_state",
        re.compile(r".*?Merged State:\s*(?P<text>.*)$"),
    ),
)


@dataclass(frozen=True)
class LogEvent:
    """A parsed event from the raw log stream."""

    kind: str
    text: str


@dataclass(frozen=True)
class TranscriptTurn:
    """One transcript turn that will be written into the transcript JSON."""

    index: int
    speaker: str
    message: str
    state: Any = None


@dataclass(frozen=True)
class KnownEntity:
    """One exact string that should be replaced everywhere it appears."""

    text: str
    replacement: str


@dataclass(frozen=True)
class ConversationSource:
    """The source files that belong to one exported dialogue."""

    log_path: Path
    survey_path: Path | None


@dataclass(frozen=True)
class ExportPaths:
    """The output paths for one exported dialogue."""

    call_dir: Path
    transcript_path: Path
    survey_path: Path


@dataclass(frozen=True)
class MetadataEntry:
    """One row in the exported metadata manifest."""

    index: int
    scenario: str
    policy: str
    call_directory: str
    transcript_file: str
    survey_file: str | None


@dataclass(frozen=True)
class RedactionSpan:
    """One direct model span replacement inside a single string."""

    start: int
    end: int
    replacement: str


@dataclass(frozen=True)
class SpaCyPipelineHandle:
    """One loaded spaCy pipeline used as an intermediate NER stage."""

    name: str
    nlp: Any


def read_text_file(path: Path) -> str:
    """Read UTF-8 log text, tolerating an optional BOM."""
    return path.read_text(encoding="utf-8-sig")


def iter_input_files(root: Path) -> Iterable[Path]:
    """Yield unique input log files matched by the configured globs in sorted order."""
    seen: set[Path] = set()
    matched: list[Path] = []
    for pattern in INPUT_GLOBS:
        for path in root.glob(pattern):
            if path.is_file() and path not in seen:
                seen.add(path)
                matched.append(path)
    for path in sorted(matched):
        yield path


def iter_conversation_sources(root: Path) -> Iterable[ConversationSource]:
    """Yield each log file together with its sibling survey file."""
    for log_path in iter_input_files(root):
        survey_path = log_path.with_name("survey.json")
        yield ConversationSource(
            log_path=log_path,
            survey_path=survey_path if survey_path.is_file() else None,
        )


def parse_log_events(lines: Sequence[str]) -> list[LogEvent]:
    """Parse relevant log lines into structured events."""
    events: list[LogEvent] = []
    for line in lines:
        if events and "Survey saved for session" in line:
            return events
        for kind, pattern in LOG_FIELD_PATTERNS:
            match = pattern.match(line)
            if match is None:
                continue
            text = match.group("text").strip()
            if kind == "operator" and text.startswith("End(data="):
                return events
            events.append(LogEvent(kind=kind, text=text))
            break
    return events


def parse_state_text(text: str) -> Any:
    """Parse a merged-state payload from Python repr into JSON-compatible data."""
    normalized = ENUM_REPR_PATTERN.sub(lambda match: repr(match.group(1)), text)
    try:
        value = ast.literal_eval(normalized)
    except (SyntaxError, ValueError):
        return text
    return normalize_json_value(value)


def normalize_json_value(value: Any) -> Any:
    """Convert parsed Python values into JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): normalize_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [normalize_json_value(item) for item in value]
    return value


def read_json_file(path: Path) -> Any:
    """Read a JSON file with UTF-8 encoding."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sanitize_survey_json(value: Any) -> Any:
    """Remove timestamp and id-like fields from a survey payload."""
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if should_drop_survey_key(str(key)):
                continue
            cleaned[str(key)] = sanitize_survey_json(item)
        return cleaned
    if isinstance(value, list):
        return [sanitize_survey_json(item) for item in value]
    return value


def build_survey_export_payload(value: Any) -> dict[str, Any]:
    """Export only the survey ``responses`` object as the top-level JSON value."""
    if not isinstance(value, dict):
        raise ValueError("survey.json root is not an object")
    responses = value.get("responses")
    if not isinstance(responses, dict):
        raise ValueError("survey.json is missing an object-valued 'responses' field")
    return sanitize_survey_json(responses)


def should_drop_survey_key(key: str) -> bool:
    """Return whether a survey key should be removed from the export."""
    normalized = key.lower()
    return normalized in SURVEY_DROP_KEYS or normalized.endswith("_id")


def read_state_changes_metadata(log_path: Path) -> dict[str, str]:
    """Read scenario/policy metadata from the first line of sibling ``state_changes.jsonl``."""
    state_changes_path = log_path.with_name("state_changes.jsonl")
    if not state_changes_path.is_file():
        return {}

    with state_changes_path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline().strip()
    if not first_line:
        return {}
    try:
        payload = json.loads(first_line)
    except json.JSONDecodeError:
        return {}
    if payload.get("event") != "metadata":
        return {}
    session = payload.get("session")
    if not isinstance(session, dict):
        return {}
    metadata: dict[str, str] = {}
    timestamp = payload.get("timestamp")
    if isinstance(timestamp, str) and timestamp:
        metadata["timestamp"] = timestamp[:19]
    scenario = session.get("scenario")
    if isinstance(scenario, str) and scenario:
        metadata["scenario"] = scenario
    policy = session.get("policy")
    if isinstance(policy, str) and policy:
        metadata["policy"] = policy
    return metadata


def iter_state_changes(log_path: Path) -> Iterable[dict[str, Any]]:
    """Yield JSON objects from sibling ``state_changes.jsonl``."""
    state_changes_path = log_path.with_name("state_changes.jsonl")
    if not state_changes_path.is_file():
        return

    with state_changes_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                yield payload


def filter_log_lines_from_timestamp(
    lines: Sequence[str], start_timestamp: str | None
) -> list[str]:
    """Return log lines starting from the first line at or after ``start_timestamp``."""
    if start_timestamp is None:
        return list(lines)
    if LOG_TIMESTAMP_PATTERN.fullmatch(start_timestamp) is None:
        return list(lines)

    started = False
    filtered: list[str] = []
    for line in lines:
        if not started:
            match = LOG_TIMESTAMP_PATTERN.match(line)
            if match is None:
                continue
            if match.group("timestamp") < start_timestamp:
                continue
            started = True
        filtered.append(line)
    return filtered


def map_policy_label(policy: str | None) -> str | None:
    """Map raw policy ids to export labels."""
    if policy is None:
        return None
    return POLICY_LABELS.get(policy, policy)


def normalize_scenario_name(scenario: str) -> str:
    """Drop a trailing Markdown suffix from a scenario name."""
    return scenario.removesuffix(".md")


def build_transcript(events: Sequence[LogEvent]) -> list[TranscriptTurn]:
    """Build a clean transcript with cumulative merged state on every later turn."""
    turns: list[TranscriptTurn] = []
    current_state: dict[str, Any] | None = None

    for event in events:
        if event.kind == "operator":
            turns.append(
                TranscriptTurn(
                    index=len(turns),
                    speaker="operator",
                    message=event.text,
                    state=copy.deepcopy(current_state),
                )
            )
            continue

        if event.kind == "caller":
            turns.append(
                TranscriptTurn(
                    index=len(turns),
                    speaker="caller",
                    message=event.text,
                    state=copy.deepcopy(current_state),
                )
            )
            continue

        if event.kind == "merged_state" and turns:
            parsed_state = parse_state_text(event.text)
            if not isinstance(parsed_state, dict):
                raise ValueError(f"Merged State is not a dictionary: {event.text}")
            ensure_state_superset(current_state, parsed_state)
            current_state = parsed_state
            turns[-1] = replace(turns[-1], state=copy.deepcopy(current_state))

    return turns


def build_transcript_from_state_changes(log_path: Path) -> list[TranscriptTurn]:
    """Build a transcript from ``state_changes.jsonl`` when caller log lines are absent."""
    turns: list[TranscriptTurn] = []
    current_state: dict[str, Any] | None = None
    last_caller_index: int | None = None

    for payload in iter_state_changes(log_path):
        event_type = payload.get("event")

        if event_type == "extraction":
            question = payload.get("question")
            if isinstance(question, str) and question.strip():
                turns.append(
                    TranscriptTurn(
                        index=len(turns),
                        speaker="operator",
                        message=question.strip(),
                        state=copy.deepcopy(current_state),
                    )
                )

            response = payload.get("response")
            if isinstance(response, str) and response.strip():
                turns.append(
                    TranscriptTurn(
                        index=len(turns),
                        speaker="caller",
                        message=response.strip(),
                        state=copy.deepcopy(current_state),
                    )
                )
                last_caller_index = len(turns) - 1
            continue

        if event_type != "state_merged" or last_caller_index is None:
            continue

        state = payload.get("state")
        if not isinstance(state, dict):
            continue
        call_state = state.get("call_state")
        if not isinstance(call_state, dict):
            continue

        ensure_state_superset(current_state, call_state)
        current_state = normalize_json_value(call_state)
        turns[last_caller_index] = replace(
            turns[last_caller_index],
            state=copy.deepcopy(current_state),
        )

    return turns


def load_spacy_entity_pipelines() -> tuple[SpaCyPipelineHandle, ...]:
    """Load spaCy NER pipelines used after exact and gazetteer matching."""
    import de_core_news_lg
    import en_core_web_trf

    return (
        SpaCyPipelineHandle(name="de_core_news_lg", nlp=de_core_news_lg.load()),
        SpaCyPipelineHandle(name="en_core_web_trf", nlp=en_core_web_trf.load()),
    )


class TranscriptAnonymizer:
    """Collect exact, gazetteer, and spaCy NER entities and reuse them."""

    def __init__(
        self,
        spacy_pipelines: Sequence[SpaCyPipelineHandle] = (),
    ) -> None:
        self.spacy_pipelines = tuple(spacy_pipelines)

    def build_entities(
        self,
        turns: Sequence[TranscriptTurn],
    ) -> tuple[list[KnownEntity], list[KnownEntity]]:
        """Use only gazetteer matches first, then spaCy NER over source texts."""
        source_texts = collect_entity_source_texts(turns)
        gazetteer_person_entities = extract_gazetteer_phrase_entities(
            source_texts,
            PERSON_REPLACEMENT,
        )
        gazetteer_location_entities = extract_gazetteer_phrase_entities(
            source_texts,
            LOCATION_REPLACEMENT,
        )
        texts_for_spacy = mask_known_entities_in_texts(
            source_texts,
            gazetteer_person_entities,
            gazetteer_location_entities,
        )
        spacy_person_entities = self.extract_spacy_entities(
            texts_for_spacy,
            PERSON_REPLACEMENT,
        )
        spacy_location_entities = self.extract_spacy_entities(
            texts_for_spacy,
            LOCATION_REPLACEMENT,
        )
        return (
            merge_entities(
                gazetteer_person_entities,
                spacy_person_entities,
            ),
            merge_entities(
                gazetteer_location_entities,
                spacy_location_entities,
            ),
        )

    def extract_spacy_entities(
        self,
        texts: Sequence[str],
        replacement: str,
    ) -> list[KnownEntity]:
        """Return entity strings from spaCy over partially redacted text."""
        unique_texts = [text for text in deduplicate_texts(texts) if text.strip()]
        if not unique_texts or not self.spacy_pipelines:
            return []

        entities: list[KnownEntity] = []
        seen: set[tuple[str, str]] = set()
        for pipeline in self.spacy_pipelines:
            for text, doc in zip(
                unique_texts,
                pipeline.nlp.pipe(unique_texts, batch_size=SPACY_BATCH_SIZE),
            ):
                for span in extract_spacy_spans(text, doc, replacement):
                    for candidate in extract_spacy_entity_texts(
                        text,
                        span.start,
                        span.end,
                        replacement,
                    ):
                        signature = (normalize_entity_text(candidate), replacement)
                        if signature in seen:
                            continue
                        seen.add(signature)
                        entities.append(
                            KnownEntity(text=candidate, replacement=replacement)
                        )
        return sort_entities_by_length(entities)


def extract_spacy_spans(
    text: str,
    doc: Any,
    replacement: str,
) -> list[RedactionSpan]:
    """Convert one spaCy doc into filtered entity spans."""
    labels = (
        SPACY_PERSON_LABELS
        if replacement == PERSON_REPLACEMENT
        else SPACY_LOCATION_LABELS
    )
    spans = [
        RedactionSpan(
            start=int(entity.start_char),
            end=int(entity.end_char),
            replacement=replacement,
        )
        for entity in getattr(doc, "ents", ())
        if str(getattr(entity, "label_", "")) in labels
    ]
    return merge_spans(text, spans)


def merge_spans(text: str, spans: Sequence[RedactionSpan]) -> list[RedactionSpan]:
    """Merge overlapping spans from one model pass."""
    merged: list[RedactionSpan] = []
    for span in sorted(spans, key=lambda item: (item.start, -(item.end - item.start))):
        if not merged:
            merged.append(span)
            continue
        previous = merged[-1]
        gap = text[previous.end : span.start]
        if span.start >= previous.end and gap and not _is_mergeable_gap(gap):
            merged.append(span)
            continue
        merged[-1] = RedactionSpan(
            start=previous.start,
            end=max(previous.end, span.end),
            replacement=previous.replacement,
        )
    return merged


def _is_mergeable_gap(text: str) -> bool:
    """Return whether two neighboring location spans should be merged."""
    return bool(re.fullmatch(r"[\s,./-]*", text))


def trim_entity_text(text: str, start: int, end: int) -> str | None:
    """Trim a model span down to the exact string that should be replaced."""
    start = max(0, start)
    end = min(len(text), end)
    while start < end and text[start] in " \t\r\n,.;:()[]{}":
        start += 1
    while end > start and text[end - 1] in " \t\r\n,.;:()[]{}":
        end -= 1
    if start >= end:
        return None
    return text[start:end]


def extract_exact_entity_texts(
    text: str,
    start: int,
    end: int,
    replacement: str,
) -> list[str]:
    """Trim one candidate span down to its exact text."""
    del replacement
    candidate = trim_entity_text(text, start, end)
    if candidate is None or is_replacement_placeholder_text(candidate):
        return []
    return [candidate]


def extract_spacy_entity_texts(
    text: str,
    start: int,
    end: int,
    replacement: str,
) -> list[str]:
    """Trim one spaCy span and keep only detector-backed candidates."""
    candidate = trim_entity_text(text, start, end)
    if candidate is None or is_replacement_placeholder_text(candidate):
        return []
    if replacement == PERSON_REPLACEMENT:
        return extract_spacy_person_candidates(candidate)
    return [candidate]


def extract_gazetteer_phrase_entities(
    texts: Sequence[str],
    replacement: str,
) -> list[KnownEntity]:
    """Return gazetteer phrase matches from source text."""
    gazetteer = gazetteer_for_replacement(replacement)
    if not gazetteer:
        return []

    entities: list[KnownEntity] = []
    for text in deduplicate_texts(texts):
        if not text.strip():
            continue
        candidates = extract_exact_gazetteer_phrases(text, gazetteer)
        for candidate in candidates:
            normalized_candidate = normalize_candidate_text(candidate)
            if not normalized_candidate:
                continue
            entities.append(
                KnownEntity(text=normalized_candidate, replacement=replacement)
            )
    return merge_entities(entities)


def extract_spacy_person_candidates(text: str) -> list[str]:
    """Return spaCy PERSON spans, using the name gazetteer for single-token names."""
    candidate = normalize_candidate_text(text)
    if not candidate:
        return []
    if len(candidate.split()) == 1:
        if normalize_entity_text(candidate) not in load_common_name_gazetteer():
            return []
    return [candidate]


def gazetteer_for_replacement(replacement: str) -> frozenset[str]:
    """Return the gazetteer corresponding to one replacement type."""
    if replacement == PERSON_REPLACEMENT:
        return load_common_name_gazetteer()
    if replacement == LOCATION_REPLACEMENT:
        return load_german_location_gazetteer()
    return frozenset()


def normalize_candidate_text(text: str) -> str:
    """Collapse whitespace and trim punctuation from a raw entity candidate."""
    return re.sub(r"\s+", " ", text).strip(" \t\r\n,.;:()[]{}!?\"'")


def is_replacement_placeholder_text(text: str) -> bool:
    """Return whether text is already one of the redaction placeholders."""
    normalized = normalize_candidate_text(text)
    return (
        text.strip() in REPLACEMENT_PLACEHOLDERS
        or normalized.casefold() in {"person", "location"}
    )


@lru_cache(maxsize=1)
def load_common_name_gazetteer() -> frozenset[str]:
    """Load common German first names from Faker's bundled de_DE person provider."""
    from faker.providers.person.de_DE import Provider as GermanPersonProvider

    entries: set[str] = set()
    for attribute in (
        "first_names",
        "first_names_female",
        "first_names_male",
        "first_names_nonbinary",
    ):
        for value in getattr(GermanPersonProvider, attribute, ()):
            normalized = normalize_entity_text(value)
            if normalized:
                entries.add(normalized)
    return frozenset(entries)


@lru_cache(maxsize=1)
def load_german_location_gazetteer() -> frozenset[str]:
    """Load German locations from geonamescache plus Faker's German states."""
    import geonamescache
    from faker.providers.address.de_DE import Provider as GermanAddressProvider

    entries: set[str] = set()
    geonames = geonamescache.GeonamesCache(min_city_population=5000)
    for city in geonames.get_cities().values():
        if city.get("countrycode") != "DE":
            continue
        for value in (city.get("name"), *city.get("alternatenames", ())):
            normalized = normalize_entity_text(value or "")
            if normalized:
                entries.add(normalized)
    for state in getattr(GermanAddressProvider, "states", ()):
        normalized = normalize_entity_text(state)
        if normalized:
            entries.add(normalized)
    return frozenset(entries)


def contains_gazetteer_entry(text: str, gazetteer: frozenset[str]) -> bool:
    """Return whether any gazetteer entry appears as a token-bounded substring."""
    for entry in gazetteer:
        if not entry:
            continue
        if re.search(rf"(?<!\w){re.escape(entry)}(?!\w)", text, flags=re.IGNORECASE):
            return True
    return False


def extract_exact_gazetteer_phrases(
    text: str,
    gazetteer: frozenset[str],
) -> list[str]:
    """Return exact gazetteer-backed phrases found in free-form text."""
    candidates: list[str] = []
    tokens = list(GAZETTEER_TOKEN_PATTERN.finditer(text))
    for start_index in range(len(tokens)):
        max_end = min(len(tokens), start_index + MAX_GAZETTEER_NGRAM_TOKENS)
        for end_index in range(max_end, start_index, -1):
            phrase = text[tokens[start_index].start() : tokens[end_index - 1].end()]
            if normalize_entity_text(phrase) not in gazetteer:
                continue
            append_unique_candidates(candidates, [phrase])
            break
    return sort_candidate_texts(candidates)


def mask_known_entities_in_texts(
    texts: Sequence[str],
    person_entities: Sequence[KnownEntity],
    location_entities: Sequence[KnownEntity],
) -> list[str]:
    """Mask already-known entities before a later extraction stage."""
    return [
        mask_known_entities_in_text(text, person_entities, location_entities)
        for text in deduplicate_texts(texts)
        if text.strip()
    ]


def mask_known_entities_in_text(
    text: str,
    person_entities: Sequence[KnownEntity],
    location_entities: Sequence[KnownEntity],
) -> str:
    """Mask exact person and location matches with typed placeholders."""
    person_masked = apply_exact_entities(text, person_entities)
    return apply_exact_entities(person_masked, location_entities)


def append_unique_candidates(target: list[str], candidates: Sequence[str]) -> None:
    """Append only new entity candidates, preserving first-seen order."""
    seen = {normalize_entity_text(item) for item in target}
    for candidate in candidates:
        signature = normalize_entity_text(candidate)
        if not signature or signature in seen:
            continue
        target.append(candidate)
        seen.add(signature)


def deduplicate_entity_candidates(candidates: Sequence[str]) -> list[str]:
    """Return entity candidate strings without normalized duplicates."""
    unique: list[str] = []
    append_unique_candidates(unique, candidates)
    return unique


def sort_candidate_texts(candidates: Sequence[str]) -> list[str]:
    """Sort raw candidate strings longest-first for stable replacement precedence."""
    return sorted(candidates, key=len, reverse=True)


def normalize_entity_text(text: str) -> str:
    """Normalize entity text for deduplication and precedence checks."""
    return re.sub(r"\s+", " ", text).strip(" \t\r\n,.;:()[]{}!?\"'").casefold()


def deduplicate_texts(texts: Sequence[str]) -> list[str]:
    """Return texts in first-seen order without duplicates."""
    unique: list[str] = []
    seen: set[str] = set()
    for text in texts:
        if text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return unique


def collect_entity_source_texts(
    turns: Sequence[TranscriptTurn],
) -> list[str]:
    """Collect caller text and state strings for gazetteer/spaCy passes."""
    texts: list[str] = []
    seen: set[str] = set()
    for turn in turns:
        if turn.speaker == "caller":
            text = turn.message.strip()
            if text and text not in seen:
                seen.add(text)
                texts.append(text)
        collect_state_strings(turn.state, texts, seen)
    return texts


def collect_state_strings(
    value: Any,
    texts: list[str],
    seen: set[str],
) -> None:
    """Recursively collect state strings from transcript state values."""
    if isinstance(value, str):
        text = value.strip()
        if text and text not in seen:
            seen.add(text)
            texts.append(text)
        return
    if isinstance(value, list):
        for item in value:
            collect_state_strings(item, texts, seen)
        return
    if not isinstance(value, dict):
        return
    for item in value.values():
        collect_state_strings(item, texts, seen)


def merge_entities(*groups: Sequence[KnownEntity]) -> list[KnownEntity]:
    """Merge entity groups while preserving one normalized text per replacement."""
    merged: list[KnownEntity] = []
    seen: set[tuple[str, str]] = set()
    for group in groups:
        for entity in group:
            signature = (normalize_entity_text(entity.text), entity.replacement)
            if signature in seen:
                continue
            seen.add(signature)
            merged.append(entity)
    return sort_entities_by_length(merged)


def sort_entities_by_length(entities: Sequence[KnownEntity]) -> list[KnownEntity]:
    """Sort entities longest-first for stable exact replacement."""
    return sorted(entities, key=lambda entity: len(entity.text), reverse=True)


def apply_exact_entities(
    text: str,
    entities: Sequence[KnownEntity],
    renderer: Callable[[re.Match[str], KnownEntity], str] | None = None,
) -> str:
    """Rewrite exact entity strings on token boundaries without overlap."""
    if renderer is None:
        renderer = render_masked_entity
    spans: list[tuple[int, int, re.Match[str], KnownEntity]] = []
    for entity in entities:
        if is_replacement_placeholder_text(entity.text):
            continue
        pattern = re.compile(
            rf"(?<![\w\[]){re.escape(entity.text)}(?![\w\]])",
            flags=re.IGNORECASE,
        )
        for match in pattern.finditer(text):
            start, end = match.span()
            if any(start < existing_end and end > existing_start for existing_start, existing_end, _, _ in spans):
                continue
            spans.append((start, end, match, entity))
    if not spans:
        return text

    parts: list[str] = []
    cursor = 0
    for start, end, match, entity in sorted(spans, key=lambda item: item[0]):
        parts.append(text[cursor:start])
        parts.append(renderer(match, entity))
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts)


def render_masked_entity(match: re.Match[str], entity: KnownEntity) -> str:
    """Render one internal stage-masking replacement."""
    del match
    return entity.replacement


def render_review_annotation(match: re.Match[str], entity: KnownEntity) -> str:
    """Render one exported review annotation while preserving matched text."""
    return wrap_review_annotation(match.group(0), entity.replacement)


def wrap_review_annotation(text: str, replacement: str) -> str:
    """Wrap one detected span for manual review with type-specific markers."""
    if replacement == PERSON_REPLACEMENT:
        return f"{PERSON_REVIEW_MARKER_PREFIX}{text}{PERSON_REVIEW_MARKER_SUFFIX}"
    if replacement == LOCATION_REPLACEMENT:
        return f"{LOCATION_REVIEW_MARKER_PREFIX}{text}{LOCATION_REVIEW_MARKER_SUFFIX}"
    return text


def ensure_state_superset(
    previous_state: dict[str, Any] | None, current_state: dict[str, Any]
) -> None:
    """Raise if the new merged state drops previously known keys or structure."""
    if previous_state is None:
        return
    mismatch = find_state_regression(previous_state, current_state, path="state")
    if mismatch is not None:
        raise ValueError(f"Merged state regressed: {mismatch}")


def find_state_regression(previous: Any, current: Any, path: str) -> str | None:
    """Return a human-readable key/shape regression description, or ``None``."""
    if isinstance(previous, dict):
        if not isinstance(current, dict):
            return f"{path} changed type from dict to {type(current).__name__}"
        for key, previous_value in previous.items():
            if key not in current:
                if str(key) in TRANSIENT_STATE_KEYS:
                    continue
                return f"{path}.{key} is missing"
            mismatch = find_state_regression(
                previous_value, current[key], path=f"{path}.{key}"
            )
            if mismatch is not None:
                return mismatch
        return None

    if isinstance(previous, list):
        if not isinstance(current, list):
            return f"{path} changed type from list to {type(current).__name__}"
        if len(current) < len(previous):
            return f"{path} shrank list length from {len(previous)} to {len(current)}"
        for index, previous_value in enumerate(previous):
            mismatch = find_state_regression(
                previous_value, current[index], path=f"{path}[{index}]"
            )
            if mismatch is not None:
                return mismatch
        return None

    return None


def anonymize_text(
    text: str,
    person_entities: Sequence[KnownEntity],
    location_entities: Sequence[KnownEntity],
) -> str:
    """Wrap exact person matches first, then exact location matches, for review."""
    person_annotated = apply_exact_entities(
        text,
        person_entities,
        render_review_annotation,
    )
    return apply_exact_entities(
        person_annotated,
        location_entities,
        render_review_annotation,
    )


def anonymize_json_value(
    value: Any,
    person_entities: Sequence[KnownEntity],
    location_entities: Sequence[KnownEntity],
) -> Any:
    """Apply exported review annotations recursively to a JSON-like value."""
    if isinstance(value, str):
        return anonymize_text(value, person_entities, location_entities)
    if isinstance(value, list):
        return [
            anonymize_json_value(item, person_entities, location_entities)
            for item in value
        ]
    if isinstance(value, dict):
        anonymized: dict[str, Any] = {}
        for key, item in value.items():
            anonymized[key] = anonymize_json_value(
                item,
                person_entities,
                location_entities,
            )
        return anonymized
    return value


def anonymize_turns(
    turns: Sequence[TranscriptTurn],
    person_entities: Sequence[KnownEntity],
    location_entities: Sequence[KnownEntity],
) -> list[TranscriptTurn]:
    """Apply exported review annotations to transcript messages and states."""
    anonymized: list[TranscriptTurn] = []
    for turn in turns:
        anonymized.append(
            TranscriptTurn(
                index=turn.index,
                speaker=turn.speaker,
                message=anonymize_text(
                    turn.message,
                    person_entities,
                    location_entities,
                ),
                state=anonymize_json_value(
                    turn.state,
                    person_entities,
                    location_entities,
                ),
            )
        )
    return anonymized


def count_review_annotations_in_value(value: Any) -> int:
    """Count review-annotation occurrences in a JSON-like value."""
    if isinstance(value, str):
        return len(REVIEW_ANNOTATION_PATTERN.findall(value))
    if isinstance(value, list):
        return sum(count_review_annotations_in_value(item) for item in value)
    if isinstance(value, dict):
        return sum(count_review_annotations_in_value(item) for item in value.values())
    return 0


def count_review_annotations(turns: Sequence[TranscriptTurn]) -> int:
    """Count review annotations in the exported transcript."""
    total = 0
    for turn in turns:
        total += len(REVIEW_ANNOTATION_PATTERN.findall(turn.message))
        total += count_review_annotations_in_value(turn.state)
    return total


def write_transcript_json(turns: Sequence[TranscriptTurn], outfile: Path) -> None:
    """Write the transcript as one JSON array."""
    payload = [
        {
            "index": turn.index,
            "speaker": turn.speaker,
            "message": turn.message,
            "state": turn.state,
        }
        for turn in turns
    ]
    write_json_file(payload, outfile)


def write_json_file(value: Any, outfile: Path) -> None:
    """Write one JSON document with UTF-8 encoding."""
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with outfile.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_report(report_lines: Sequence[str], output_dir: Path) -> None:
    """Write a plain-text export report next to ``metadata.json``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.txt"
    lines = list(report_lines) if report_lines else ["No issues detected."]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_export_paths(output_dir: Path, index: int) -> ExportPaths:
    """Return the output paths for one exported dialogue."""
    call_dir = output_dir / f"call_{index}"
    return ExportPaths(
        call_dir=call_dir,
        transcript_path=call_dir / "transcript.json",
        survey_path=call_dir / "survey.json",
    )


def output_dir_root(export_paths: ExportPaths) -> Path:
    """Return the export root directory for one dialogue path bundle."""
    return export_paths.call_dir.parent


def format_debug_context(
    source: ConversationSource, export_paths: ExportPaths, index: int
) -> str:
    """Return a readable debug context string for one dialogue export."""
    survey_path = source.survey_path if source.survey_path is not None else "<missing>"
    state_changes_path = source.log_path.with_name("state_changes.jsonl")
    return (
        f"index={index} "
        f"log={source.log_path} "
        f"survey={survey_path} "
        f"state_changes={state_changes_path} "
        f"call_out={export_paths.transcript_path} "
        f"survey_out={export_paths.survey_path}"
    )


def process_conversation(
    index: int,
    source: ConversationSource,
    export_paths: ExportPaths,
    anonymizer: TranscriptAnonymizer,
    report_lines: list[str],
) -> tuple[int, int, MetadataEntry]:
    """Export one dialogue transcript, its survey, and a metadata row."""
    call_id = export_paths.call_dir.name
    state_changes_metadata = read_state_changes_metadata(source.log_path)
    lines = filter_log_lines_from_timestamp(
        read_text_file(source.log_path).splitlines(),
        state_changes_metadata.get("timestamp"),
    )
    events = parse_log_events(lines)
    if any(event.kind == "caller" for event in events):
        turns = build_transcript(events)
    else:
        report_lines.append(
            f"{call_id}: Reconstructed transcript from state_changes.jsonl for "
            f"{source.log_path} because stdout.log had no caller events."
        )
        turns = build_transcript_from_state_changes(source.log_path)
    person_entities, location_entities = anonymizer.build_entities(turns)
    anonymized_turns = anonymize_turns(
        turns,
        person_entities,
        location_entities,
    )
    write_transcript_json(anonymized_turns, export_paths.transcript_path)

    survey_file: str | None = None
    if source.survey_path is None:
        message = (
            f"{call_id}: Missing survey.json for {source.log_path}; "
            "metadata still comes from state_changes.jsonl."
        )
        report_lines.append(message)
        print(message, file=sys.stderr)
    else:
        survey_payload = build_survey_export_payload(read_json_file(source.survey_path))
        write_json_file(survey_payload, export_paths.survey_path)
        survey_file = export_paths.survey_path.relative_to(
            output_dir_root(export_paths)
        ).as_posix()

    scenario = state_changes_metadata.get("scenario")
    if scenario is None:
        raise ValueError(f"Could not determine scenario for {source.log_path}")
    policy = map_policy_label(state_changes_metadata.get("policy"))
    if policy is None:
        raise ValueError(f"Could not determine policy for {source.log_path}")

    metadata_entry = MetadataEntry(
        index=index,
        scenario=normalize_scenario_name(scenario),
        policy=policy,
        call_directory=export_paths.call_dir.relative_to(
            output_dir_root(export_paths)
        ).as_posix(),
        transcript_file=export_paths.transcript_path.relative_to(
            output_dir_root(export_paths)
        ).as_posix(),
        survey_file=survey_file,
    )
    annotation_count = count_review_annotations(anonymized_turns)
    return len(turns), annotation_count, metadata_entry


def write_metadata(entries: Sequence[MetadataEntry], output_dir: Path) -> None:
    """Write the dataset manifest that links each dialogue to its scenario."""
    payload = {
        "entries": [
            {
                "index": entry.index,
                "scenario": entry.scenario,
                "policy": entry.policy,
                "call_directory": entry.call_directory,
                "transcript_file": entry.transcript_file,
                "survey_file": entry.survey_file,
            }
            for entry in entries
        ]
    }
    write_json_file(payload, output_dir / "metadata.json")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Parse log files into anonymized transcript JSON files, sanitized "
            "survey JSON files, and a metadata manifest."
        )
    )
    parser.add_argument(
        "--input-dir", "-i", required=True, help="Directory containing input log files."
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        required=True,
        help="Directory for anonymized JSON output.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print progress messages."
    )
    args = parser.parse_args(argv)

    anonymizer = TranscriptAnonymizer(load_spacy_entity_pipelines())

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    metadata_entries: list[MetadataEntry] = []
    report_lines: list[str] = []

    for index, source in enumerate(iter_conversation_sources(input_dir), start=1):
        export_paths = build_export_paths(output_dir, index)
        try:
            turn_count, annotation_count, metadata_entry = process_conversation(
                index, source, export_paths, anonymizer, report_lines
            )
        except Exception as exc:
            context = format_debug_context(source, export_paths, index)
            error_message = f"Export failed: {exc.__class__.__name__}: {exc}"
            report_lines.extend(
                [
                    error_message,
                    context,
                    traceback.format_exc().rstrip(),
                ]
            )
            write_report(report_lines, output_dir)
            print(error_message, file=sys.stderr)
            print(context, file=sys.stderr)
            raise RuntimeError(f"{error_message} | {context}") from exc
        metadata_entries.append(metadata_entry)
        if args.verbose:
            print(
                f"Processed {source.log_path} -> {export_paths.transcript_path} "
                f"({turn_count} turns, {annotation_count} annotations)"
            )

    write_metadata(metadata_entries, output_dir)
    write_report(report_lines, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
