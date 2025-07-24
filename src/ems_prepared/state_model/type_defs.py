"""Type definitions for EMS Prepared state model.

This module provides type aliases used throughout the state model.
"""

from enum import Enum, StrEnum, auto
from typing import TypeAlias

Unknown: TypeAlias = None
# type Unknown = None  # this does not work
# Unknown = NewType("Unknown", tp=None)


type KnownBoolean = bool | Unknown
type KnownString = str | Unknown


def tristate(values) -> KnownBoolean:
    from functools import reduce

    def tri_combine(acc, curr):
        # single True is sufficient
        if acc is True or curr is True:
            return True
        # All needs to be False
        if acc is False and curr is False:
            return False
        # at least one None, nothing True yet
        return Unknown

    return reduce(tri_combine, values, Unknown)


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
