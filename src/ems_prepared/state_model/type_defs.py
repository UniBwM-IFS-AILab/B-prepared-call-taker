"""Type definitions for EMS Prepared state model.

This module provides type aliases used throughout the state model.
"""

from enum import Enum, StrEnum, auto
from typing import Iterable, TypeAlias

Unknown: TypeAlias = None
# type Unknown = None  # this does not work

# Unknown = NewType("Unknown", tp=None)

type KnownBoolean = Unknown | bool
# KnownBoolean: TypeAlias = Unknown | bool
type KnownString = str | Unknown


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

    return reduce(tri_combine, values, None)


class EmergencyType(StrEnum):
    """Enumeration for different types of emergencies."""

    MEDICAL = auto()
    FIRE = auto()
    NON_EMERGENCY = auto()
    FIRE_MEDICAL = auto()


class DispoType(Enum):
    """Enumeration for different types of necessary Disposition."""

    RD1 = auto()  # Notfall
    RD2 = auto()  # Notarzteinsatz


class KnownBoolean2(Enum):
    YES = True
    NO = False
    UNKNOWN = None
