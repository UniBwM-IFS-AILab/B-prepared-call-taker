"""FastAPI composition root."""

from __future__ import annotations

from ems_prepared.adapters.fastapi.app import create_app
from ems_prepared.model.session_service import create_session_manager

app = create_app(create_session_manager())
