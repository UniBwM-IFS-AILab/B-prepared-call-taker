"""Type definitions for EMS Prepared state model.

This module provides type aliases used throughout the state model.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum, StrEnum, auto
from typing import Literal, NewType, TypeAlias

Unknown: TypeAlias = None
# type Unknown = Literal[None]
# type Unknown = None

# type Known[T] = T | Unknown

type KnownBoolean = bool | Unknown
type KnownString = str | Unknown
type KnownInteger = int | Unknown

type RD1_Boolean = KnownBoolean
# type RD2_Boolean = KnownBoolean
type RD2_Boolean = RD1_Boolean
type CPR_Boolean = RD2_Boolean
type Urgency_Boolean = RD2_Boolean


def tristate(values: Iterable[KnownBoolean]) -> KnownBoolean:
    from functools import reduce

    def tri_combine(acc: KnownBoolean, curr: KnownBoolean) -> KnownBoolean:
        # single True is sufficient
        if acc is True or curr is True:
            return True
        # All needs to be False
        if acc is False and curr is False:
            return False
        # at least one None, nothing True yet
        return None

    return reduce(tri_combine, values, False)  # type: ignore


class EmergencyType(StrEnum):
    """Enumeration for emergency types and dialogue phases."""

    INTRO = auto()
    MEDICAL = auto()
    FIRE = auto()

    # TCPR = auto()
    # FIRE_MEDICAL = auto()
    # NON_EMERGENCY = auto()


class DispoType(Enum):
    """Enumeration for different types of necessary Disposition."""

    RD1 = auto()  # Notfall
    RD2 = auto()  # Notarzteinsatz
