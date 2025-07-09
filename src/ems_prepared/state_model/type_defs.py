"""Type definitions for EMS Prepared state model.

This module provides type aliases used throughout the state model.
"""

# FIXME: ensure that this is not interpreted as an optional type in pydantic models
from enum import Enum, StrEnum, auto
from typing import TypeAlias

Unknown: TypeAlias = None
# type Unknown = None  # this does not work
# Unknown = NewType("Unknown", tp=None)


type KnownBoolean = bool | Unknown
type KnownString = str | Unknown


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
