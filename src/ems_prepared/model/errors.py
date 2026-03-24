"""Shared backend error types exposed to adapters and callers."""

from __future__ import annotations


class SessionNotFoundError(KeyError):
    """Raised when a session lookup references an unknown session id."""


class UnsupportedPolicyError(ValueError):
    """Raised when no runtime can be built for a policy name."""
