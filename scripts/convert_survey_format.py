# TODO: use this to convert old data

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ems_prepared.adapters.gradio.survey import (
    DEFAULT_BY_LABEL,
    DEFAULT_SURVEY_QUESTIONS,
)


def _iter_json_files(inputs: List[str]) -> Iterable[Path]:
    for input_path_text in inputs:
        path = Path(input_path_text)
        if path.is_dir():
            yield from sorted(path.glob("*.json"))
        else:
            yield path


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def _deterministic_extra_id(category: str, label: str, text: str) -> int:
    """
    Stable integer id for non-default questions (in case you add more later).
    """
    key = f"{category}||{label}||{text}".encode("utf-8")
    hash_digest = hashlib.sha1(key).hexdigest()
    return 1000 + (int(hash_digest[:8], 16) % 900_000_000)


def _build_normalized_payload(
    *,
    metadata: Dict[str, Any],
    timestamp: str,
    responses: List[Dict[str, Any]],
    feedback: str | None,
) -> Dict[str, Any]:
    payload = {
        "metadata": metadata,
        "timestamp": timestamp,
        "responses": responses,
    }
    if feedback is not None:
        payload["feedback"] = feedback
    return payload


def normalize_survey_json(
    src: Dict[str, Any],
    include_all_default_questions: bool = True,
) -> Dict[str, Any]:
    """
    Returns a single JSON object with exactly:
      - metadata: object
      - timestamp: string
      - responses: list[object]
      - feedback: string (optional)
    """

    timestamp = src.get("timestamp")
    if not isinstance(timestamp, str):
        raise ValueError("Missing/invalid 'timestamp' (must be a string).")

    metadata = src.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    else:
        metadata = dict(metadata)

    feedback = src.get("feedback")
    if not isinstance(feedback, str):
        legacy_feedback = metadata.pop("feedback", None)
        feedback = legacy_feedback if isinstance(legacy_feedback, str) else None
    else:
        metadata.pop("feedback", None)

    responses_in = src.get("responses")
    if responses_in is None:
        raise ValueError("Missing 'responses'.")

    # If already in the new format, just enforce shape + ordering.
    if isinstance(responses_in, list):
        out_list: List[Dict[str, Any]] = []
        for resp in responses_in:
            if not isinstance(resp, dict):
                continue
            out_list.append(
                {
                    "id": resp.get("id"),
                    "category": resp.get("category"),
                    "label": resp.get("label"),
                    "text": resp.get("text"),
                    "score": resp.get("score"),
                }
            )
        out_list.sort(
            key=lambda response_item: (
                response_item.get("id") is None,
                response_item.get("id", 10**18),
            )
        )
        return _build_normalized_payload(
            metadata=metadata,
            timestamp=timestamp,
            responses=out_list,
            feedback=feedback,
        )

    # Old format: dict keyed by label
    if not isinstance(responses_in, dict):
        raise ValueError("'responses' must be a list (new) or a dict/object (old).")

    old: Dict[str, Any] = responses_in

    def score_for_label(label: str) -> Optional[int]:
        item = old.get(label)
        if isinstance(item, dict) and isinstance(item.get("score"), int):
            return item["score"]
        return None

    out_responses: List[Dict[str, Any]] = []

    if include_all_default_questions:
        # Always emit canonical 1..7
        for survey_question in DEFAULT_SURVEY_QUESTIONS:
            out_responses.append(
                {
                    "id": survey_question.id,
                    "category": survey_question.category,
                    "label": survey_question.label,
                    "text": survey_question.text,
                    "score": score_for_label(survey_question.label),
                }
            )

        # Append any extras found in old format
        for label, item in old.items():
            if label in DEFAULT_BY_LABEL or not isinstance(item, dict):
                continue
            category = item.get("category")
            text = item.get("question") or item.get("text")
            score = item.get("score")
            if not (isinstance(category, str) and isinstance(text, str)):
                continue
            out_responses.append(
                {
                    "id": _deterministic_extra_id(category, str(label), text),
                    "category": category,
                    "label": str(label),
                    "text": text,
                    "score": score if isinstance(score, int) else None,
                }
            )

        out_responses.sort(key=lambda response_item: response_item["id"])
        return _build_normalized_payload(
            metadata=metadata,
            timestamp=timestamp,
            responses=out_responses,
            feedback=feedback,
        )

    # Only answered items
    for label, item in old.items():
        if not isinstance(item, dict):
            continue
        category = item.get("category")
        text = item.get("question") or item.get("text")
        score = item.get("score")
        if not (isinstance(category, str) and isinstance(text, str)):
            continue

        if label in DEFAULT_BY_LABEL:
            survey_question = DEFAULT_BY_LABEL[label]
            out_responses.append(
                {
                    "id": survey_question.id,
                    "category": survey_question.category,
                    "label": survey_question.label,
                    "text": survey_question.text,
                    "score": score,
                }
            )
        else:
            out_responses.append(
                {
                    "id": _deterministic_extra_id(category, str(label), text),
                    "category": category,
                    "label": str(label),
                    "text": text,
                    "score": score if isinstance(score, int) else None,
                }
            )

    out_responses.sort(key=lambda response_item: response_item["id"])
    return _build_normalized_payload(
        metadata=metadata,
        timestamp=timestamp,
        responses=out_responses,
        feedback=feedback,
    )


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputs", nargs="+", help="JSON file(s) or directory(ies) containing *.json"
    )
    parser.add_argument("--outdir", default="out", help="Output directory")
    parser.add_argument(
        "--suffix", default=".normalized.json", help="Output filename suffix"
    )
    parser.add_argument(
        "--only-answered",
        action="store_true",
        help="Only include questions present in the input (default: include all 7 defaults).",
    )
    args = parser.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for fpath in _iter_json_files(args.inputs):
        src = _load_json(fpath)
        out = normalize_survey_json(
            src, include_all_default_questions=not args.only_answered
        )

        out_path = outdir / (fpath.stem + args.suffix)
        with out_path.open("w", encoding="utf-8") as output_file:
            json.dump(out, output_file, ensure_ascii=False, indent=2)

        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
