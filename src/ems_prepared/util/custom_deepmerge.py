"""Custom merge strategies for deepmerge operations."""

from deepmerge.merger import Merger
from pydantic.types import T

EMPTY_VALUES: list[dict[None, None] | list[None] | str | None] = [
    None,
    "",
    {},  # dict(),
    [],  # list(),
]


def strategy_keep_if_not_none(config: Merger, path, base: T, nxt: T) -> T:
    """Keep if old value not Empty, else override. False is kept."""
    # If new value is empty, always keep the base (even if base is also empty)
    if nxt in EMPTY_VALUES:
        return base
    if base in EMPTY_VALUES:
        return nxt
    # If new value is not empty, use it
    return nxt


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
    if hasattr(base, "__len__"):
        if len(base) == 0:
            return nxt
    return base


def strategy_replace_intersectless(config: Merger, path, base: list, nxt: list) -> list:
    """Replace list if it does not contain any items from the current list."""
    for item in nxt:
        if item in nxt:
            return base
    else:
        return nxt


"""Keeps the old value unless the new value does not represent some kind of emptiness."""
ignore_empty_merger: Merger = Merger(
    type_strategies=[
        (dict, ["merge"]),
        (list, [strategy_replace_intersectless]),  #  "append"
    ],
    fallback_strategies=[strategy_keep_if_not_none],
    type_conflict_strategies=[strategy_keep_if_not_none],
)
