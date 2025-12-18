"""Custom merge strategies for deepmerge operations."""

from collections.abc import Sized
from enum import Enum

from deepmerge.merger import Merger
from deepmerge.strategy.core import STRATEGY_END
from pydantic.types import T

from ems_prepared.dialogue_state.type_defs import EmergencyType

EMPTY_VALUES: list[dict[None, None] | list[None] | str | None] = [
    None,
    "",
    {},  # dict(),
    [],  # list(),
]


def strategy_keep_if_not_none(config: Merger, path, base: T, nxt: T) -> T:
    """Keep if old value not Empty, else override. False is kept."""
    # If new value is empty, always keep the base (even if base is also empty)
    # if nxt in EMPTY_VALUES:
    #     return base
    # if base in EMPTY_VALUES:
    #     return nxt
    # # If new value is not empty, use it
    # return base
    return base if base not in EMPTY_VALUES else nxt


def strategy_override_if_not_none(config: Merger, path, base: T, nxt: T) -> T:
    """Override if new value not Empty, else keep. False is kept."""
    return nxt if nxt not in EMPTY_VALUES else base


def strategy_override_if_not_false(config: Merger, path, base: T, nxt: T) -> T:
    """Override if new value not False, else keep."""
    return nxt if nxt else base


def strategy_keep_if_not_false(config: Merger, path, base: T, nxt: T) -> T:
    """Keep if old value not False, else override."""
    return base if base else nxt


def strategy_length_nonzero(config: Merger, path, base: T, nxt: T) -> T:
    if isinstance(base, Sized) and len(base) == 0:
        return nxt
    return base


def strategy_replace_intersectless(config: Merger, path, base: list, nxt: list) -> list:
    """Replace list if it does not contain any items from the current list."""
    for item in nxt:
        if item in nxt:
            return base
    else:
        return nxt


def concat_strings(config: Merger, path, base: str, nxt: str) -> str | None | object:
    """Concatenate two strings with a space in between."""

    # If either is an Enum (like EmergencyType), do not treat it as a string here
    if isinstance(base, Enum) or isinstance(nxt, Enum):
        return STRATEGY_END  # fall through to other strategies (e.g. EmergencyType, fallback)

    if not base and not nxt:
        return None
    if not base:
        return nxt
    if not nxt:
        return base

    return f"{base} | {nxt}"


# TODO: use this function for string concat, first double-check
def concat_strings_similarity(
    config: Merger, path, base: str, nxt: str
) -> str | None | object:
    """Concatenate two strings only when they are sufficiently dissimilar.

    Uses `rapidfuzz` (fast and popular) to compute a similarity ratio. If
    similarity is above a threshold, keep the original `base` value; otherwise
    return the concatenation. If `rapidfuzz` is not available, falls back to
    the simpler `concat_strings` behavior.
    """

    # If either is an Enum (like EmergencyType), do not treat it as a string here
    if isinstance(base, Enum) or isinstance(nxt, Enum):
        return STRATEGY_END

    # Basic empty handling (same semantics as the simple concat)
    if not base and not nxt:
        return None
    if not base:
        return nxt
    if not nxt:
        return base

    # Attempt to use rapidfuzz for similarity; fall back if unavailable
    try:
        from rapidfuzz import fuzz

        # rapidfuzz.fuzz.ratio returns 0..100; higher means more similar
        similarity = fuzz.ratio(str(base), str(nxt))
    except Exception:
        # rapidfuzz not installed or import failed — fall back
        return concat_strings(config, path, base, nxt)

    # Threshold (percent). If the strings are very similar, keep the original.
    SIMILARITY_KEEP_THRESHOLD = 85

    if similarity >= SIMILARITY_KEEP_THRESHOLD:
        return base

    return f"{base} | {nxt}"


"""Keeps the old value unless the new value does not represent some kind of emptiness."""
ignore_empty_merger: Merger = Merger(
    type_strategies=[
        (dict, ["merge"]),
        (list, [strategy_replace_intersectless]),  #  "append"
        (EmergencyType, strategy_override_if_not_none),
        # (str, concat_strings_similarity),
    ],
    fallback_strategies=[strategy_override_if_not_none],
    type_conflict_strategies=[strategy_keep_if_not_none],
)
