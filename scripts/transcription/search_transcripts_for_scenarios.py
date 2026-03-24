#!/usr/bin/env python3
"""
Scan tabular transcript files for keyword/number combinations.

Example:
    python scripts/transcription/search_transcripts_for_scenarios.py \
        --data-dir data/transcripts \
        --keywords CPR AED \
        --output matches.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

try:
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "pandas is required. Install it with: pip install pandas openpyxl odfpy"
    ) from exc


SUPPORTED_SUFFIXES = {".csv", ".tsv", ".xlsx", ".xlsm", ".ods"}
TOKEN_PATTERN = re.compile(r"\d+|\w+", re.UNICODE)


def tokenize(text: str) -> List[str]:
    """Return cleaned tokens (words or numbers) extracted from text."""
    return TOKEN_PATTERN.findall(text)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search CSV/Excel files for occurrences of keywords paired with numbers. "
            "By default numbers are integers/decimals and matches within 5 tokens of a keyword are reported."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing transcript files to scan (searches recursively).",
    )
    parser.add_argument(
        "--keywords",
        nargs="+",
        default=[],
        help="Keywords (single words or abbreviations) to look for.",
    )
    parser.add_argument(
        "--keywords-file",
        type=Path,
        help="Optional newline-delimited list of keywords.",
    )
    parser.add_argument(
        "--numeric-pattern",
        default=r"\d+",
        help="Regular expression that defines a number (default: integers only).",
    )
    parser.add_argument(
        "--max-distance",
        type=int,
        default=5,
        help="Maximum token distance between a keyword and number (default: 5).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write matches to this CSV file; print to stdout when omitted.",
    )
    return parser.parse_args(argv)


def load_keywords(args: argparse.Namespace) -> List[str]:
    keywords = {kw.strip() for kw in args.keywords if kw.strip()}
    if args.keywords_file:
        text = args.keywords_file.read_text(encoding="utf-8")
        for line in text.splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                keywords.add(line)
    if not keywords:
        raise SystemExit(
            "At least one keyword must be supplied via --keywords or --keywords-file."
        )
    return sorted(keywords)


def iter_tabular_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            print(f"[INFO] Found file: {path}", file=sys.stderr)
            yield path


def iter_frames(path: Path) -> Iterable[Tuple[str, pd.DataFrame]]:
    try:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path, low_memory=False)
            yield "-", df
        elif path.suffix.lower() == ".tsv":
            df = pd.read_csv(path, sep="\t", low_memory=False)
            yield "-", df
        elif path.suffix.lower() in {".xlsx", ".xlsm"}:
            # Load all sheets in an Excel workbook
            frames: Dict[str, pd.DataFrame] = pd.read_excel(path, sheet_name=None)
            for sheet_name, df in frames.items():
                yield sheet_name, df
        else:
            # Use odfpy via pandas for OpenDocument spreadsheets
            frames: Dict[str, pd.DataFrame] = pd.read_excel(
                path, sheet_name=None, engine="odf"
            )
            for sheet_name, df in frames.items():
                yield sheet_name, df
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[WARN] Failed to read {path}: {exc}", file=sys.stderr)


def find_keyword_number_pairs(
    entries: Sequence[Tuple[str, str, str, str]],
    keyword_lookup: Dict[str, str],
    number_re: re.Pattern,
    max_distance: int,
) -> List[Dict[str, object]]:
    matches: List[Dict[str, object]] = []
    for idx, entry in enumerate(entries):
        normalized_token, raw_token, column, cell_text = entry
        if normalized_token not in keyword_lookup:
            continue
        keyword = keyword_lookup[normalized_token]
        for neighbor in range(
            max(0, idx - max_distance), min(len(entries), idx + max_distance + 1)
        ):
            _, neighbor_raw, neighbor_column, neighbor_text = entries[neighbor]
            if number_re.fullmatch(neighbor_raw):
                matches.append(
                    {
                        "keyword": keyword,
                        "number": neighbor_raw,
                        "keyword_column": column,
                        "number_column": neighbor_column,
                        "keyword_cell_value": cell_text,
                        "number_cell_value": neighbor_text,
                    }
                )
    return matches


def scan_file(
    file_path: Path,
    keywords: Sequence[str],
    numeric_pattern: str,
    max_distance: int,
) -> List[Dict[str, object]]:
    keyword_lookup = {kw.lower(): kw for kw in keywords}
    number_re = re.compile(rf"(?:{numeric_pattern})", re.IGNORECASE)
    results: List[Dict[str, object]] = []
    for sheet_name, frame in iter_frames(file_path):
        if frame.empty:
            continue
        for row_idx, row in frame.iterrows():
            row_entries: List[Tuple[str, str, str, str]] = []
            for column in frame.columns:
                value = row[column]
                if pd.isna(value):
                    continue
                text = str(value)
                tokens = tokenize(text)
                if not tokens:
                    continue
                for token in tokens:
                    row_entries.append((token.lower(), token, column, text))
            if not row_entries:
                continue
            matches = find_keyword_number_pairs(
                row_entries, keyword_lookup, number_re, max_distance
            )
            if not matches:
                continue
            row_number = row_idx + 2 if isinstance(row_idx, int) else row_idx
            for match in matches:
                results.append(
                    {
                        "file": str(file_path),
                        "sheet": sheet_name,
                        "row": row_number,
                        "column": match["keyword_column"],
                        "keyword": match["keyword"],
                        "number": match["number"],
                        "cell_value": match["keyword_cell_value"],
                        "number_column": match["number_column"],
                        "number_cell_value": match["number_cell_value"],
                    }
                )
    return results


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    keywords = load_keywords(args)

    if not args.data_dir.is_dir():
        print(f"[ERROR] {args.data_dir} is not a directory.", file=sys.stderr)
        return 1

    all_matches: List[Dict[str, object]] = []
    for path in iter_tabular_files(args.data_dir):
        all_matches.extend(
            scan_file(path, keywords, args.numeric_pattern, args.max_distance)
        )

    if not all_matches:
        print("No keyword/number combinations found.")
        return 0

    result_df = pd.DataFrame(all_matches)
    if args.output:
        result_df.to_csv(args.output, index=False)
        print(f"Wrote {len(result_df)} matches to {args.output}")
    else:
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            print(result_df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
