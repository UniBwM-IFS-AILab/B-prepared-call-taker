#!/usr/bin/env python3
"""
Copy survey.json and messages.tsv from *leaf directories* under SRC_ROOT
into DST_ROOT, preserving the directory hierarchy relative to SRC_ROOT.

Leaf directory = a directory that contains no subdirectories (after pruning).
If DST_ROOT is inside SRC_ROOT, the walk prunes DST_ROOT so it won't recurse into it.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

WANTED = ("survey.json", "messages.tsv")


def _relative_if_under(child: Path, parent: Path) -> Path | None:
    """Return child relative to parent if child is under parent, else None."""
    try:
        return child.relative_to(parent)
    except ValueError:
        return None


def copy_from_leaf_dirs(src_root: Path, dst_root: Path, overwrite: bool, dry_run: bool, verbose: bool) -> int:
    src_root = src_root.resolve()
    dst_root = dst_root.resolve()

    if not src_root.is_dir():
        raise NotADirectoryError(f"Source root is not a directory: {src_root}")

    if src_root == dst_root:
        raise ValueError("Destination root must be different from source root.")

    # We will prune DST if it's inside SRC to avoid copying the destination into itself.
    dst_rel_to_src = _relative_if_under(dst_root, src_root)  # None if not nested

    if not dry_run:
        dst_root.mkdir(parents=True, exist_ok=True)

    copied = 0

    # topdown=True lets us modify dirnames to prune traversal :contentReference[oaicite:1]{index=1}
    for dirpath, dirnames, filenames in os.walk(src_root, topdown=True):
        current = Path(dirpath).resolve()

        # If DST_ROOT is inside SRC_ROOT, prune the subtree by removing
        # the next path component from dirnames at each level.
        if dst_rel_to_src is not None:
            rel = _relative_if_under(dst_root, current)
            if rel is not None and rel.parts:
                next_component = rel.parts[0]
                if next_component in dirnames:
                    dirnames.remove(next_component)

        # Leaf directory (after pruning)
        if dirnames:
            continue

        fileset = set(filenames)
        for name in WANTED:
            if name not in fileset:
                continue

            src_file = current / name
            rel_dir = current.relative_to(src_root)
            dst_dir = dst_root / rel_dir
            dst_file = dst_dir / name

            # Extra guard against pathological cases (e.g., symlinks)
            try:
                if src_file.resolve() == dst_file.resolve():
                    if verbose:
                        print(f"SKIP (same file): {src_file}")
                    continue
            except FileNotFoundError:
                pass

            if dst_file.exists() and not overwrite:
                if verbose:
                    print(f"SKIP (exists): {dst_file}")
                continue

            if verbose or dry_run:
                print(f"{'DRY-RUN copy' if dry_run else 'Copy'}: {src_file} -> {dst_file}")

            if not dry_run:
                dst_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dst_file)  # preserves more metadata than copy() :contentReference[oaicite:2]{index=2}

            copied += 1

    return copied


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Copy survey.json and messages.tsv from leaf dirs, preserving hierarchy, and skipping DST if nested."
    )
    parser.add_argument("src_root", type=Path, help="Root directory to scan (source).")
    parser.add_argument("dst_root", type=Path, help="Destination root directory.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite files if they already exist in DST.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be copied, but do nothing.")
    parser.add_argument("--verbose", action="store_true", help="Print each file decision.")
    args = parser.parse_args(argv)

    try:
        copied = copy_from_leaf_dirs(
            src_root=args.src_root,
            dst_root=args.dst_root,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    print(f"Done. Copied {copied} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
