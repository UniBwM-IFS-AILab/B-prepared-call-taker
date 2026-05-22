"""Shared session bootstrap, cleanup, and logging helpers for Gradio."""

from __future__ import annotations

import logging
from uuid import UUID, uuid4

import gradio as gr

from ems_prepared.adapters.gradio.args import GradioAppArgs
from ems_prepared.adapters.gradio.chat_stream import events_include_completion
from ems_prepared.adapters.gradio.state import SessionRef, StartedSession
from ems_prepared.model.context import InputMode, Locale
from ems_prepared.model.contracts import (
    BackendEvent,
    BackendEventKind,
    MessageFeedback,
    MessageRole,
    SessionManager,
    SessionParameters,
)

logger = logging.getLogger(__name__)

DEBUG_STATIC_REPLY = (
    "🧪 Debug mode is active. Policy calls are skipped in the Gradio UI."
)


def format_session_info(
    user_id: str | None = None,
    session_id: str | None = None,
    scenario: str | None = None,
    experiment_name: str | None = None,
    policy: str | None = None,
    *,
    default_experiment_name: str | None = None,
    default_policy: str | None = None,
) -> str:
    """Format session debug info string."""
    experiment = (
        experiment_name if experiment_name is not None else default_experiment_name
    )
    policy_name = policy if policy is not None else default_policy

    user_id_str = f"`{user_id}`" if user_id else "_initializing..._"
    session_id_str = f"`{session_id}`" if session_id else "_pending session_"
    scenario_str = f"`{scenario or 'None'}`"

    return (
        f"**Experiment:** `{experiment or 'None'}`  \n"
        f"**User ID:** {user_id_str}  \n"
        f"**Session ID:** {session_id_str}  \n"
        f"**Policy:** `{policy_name or 'None'}`  \n"
        f"**Scenario:** {scenario_str}"
    )


def session_info_text(args: GradioAppArgs, **kwargs: str | None) -> str:
    """Bind `format_session_info` defaults to the active Gradio args."""
    return format_session_info(
        **kwargs,
        default_experiment_name=args.experiment_name,
        default_policy=args.policy,
    )


def resolve_or_create_user_id(stored_user_id: str) -> UUID:
    """Resolve a persisted user ID or create a fresh UUID."""
    if stored_user_id:
        try:
            return UUID(stored_user_id)
        except ValueError:
            logger.warning("Invalid stored user_id: %s, generating new", stored_user_id)
    return uuid4()


def resolve_session_user_id(stored_user_id: str, forced_user_id: int) -> UUID | None:
    """Resolve the user ID used for session creation."""
    if forced_user_id:
        return UUID(int=forced_user_id)
    if stored_user_id:
        try:
            return UUID(stored_user_id)
        except ValueError:
            logger.warning("Invalid stored user_id: %s, generating new one", stored_user_id)
    return None


def resolve_requested_session_id(session_id: str | None) -> UUID | None:
    """Resolve an optional requested session ID string."""
    if session_id is None:
        return None
    try:
        return UUID(session_id)
    except ValueError:
        logger.warning("Invalid requested session_id: %s, creating new", session_id)
        return None


async def log_consent_acceptance(
    *,
    session_manager: SessionManager,
    session_id: str,
    ui_mode: str,
    scenario_name: str | None,
) -> None:
    """Persist one consent acceptance event for auditability."""
    try:
        await session_manager.submit_feedback(
            UUID(session_id),
            MessageFeedback(
                role=MessageRole.SYSTEM,
                content="Consent accepted",
                value="consent_accept",
                metadata={
                    "ui": "gradio",
                    "ui_mode": ui_mode,
                    "source": "consent.checkbox",
                    "scenario_name": scenario_name,
                },
            ),
        )
    except LookupError:
        logger.warning("Consent log dropped for missing session %s", session_id)


async def end_active_session(
    active_session: SessionRef | None,
    *,
    session_manager: SessionManager,
    skip_policy_calls: bool = False,
) -> None:
    """Best-effort cleanup for one active session."""
    if active_session is None or skip_policy_calls:
        return

    try:
        _ = await session_manager.end_session(UUID(active_session.session_id))
    except Exception:
        logger.exception("Failed ending session %s", active_session.session_id)


async def start_session_for_gradio(
    scenario_name: str | None,
    stored_user_id: str,
    previous_session: SessionRef | None,
    requested_session_id: str | None = None,
    *,
    session_manager: SessionManager,
    args: GradioAppArgs,
) -> StartedSession:
    """Start one backend session using shared Gradio behavior."""
    await end_active_session(
        previous_session,
        session_manager=session_manager,
        skip_policy_calls=args.should_skip_policy_calls(),
    )
    policy_setting = args.resolve_policy()
    experiment_name = args.experiment_name or ""

    if args.should_skip_policy_calls():
        user_id = resolve_or_create_user_id(stored_user_id)
        resolved_session_id = requested_session_id or str(uuid4())
        session = SessionRef(
            user_id=str(user_id),
            session_id=resolved_session_id,
            policy_name=policy_setting,
            scenario_name=scenario_name,
            experiment_name=experiment_name,
        )
        events = [BackendEvent(kind=BackendEventKind.MESSAGE, text=DEBUG_STATIC_REPLY)]
        return StartedSession(
            session=session,
            events=events,
            user_id=session.user_id,
            is_complete=events_include_completion(events),
        )

    user_id = resolve_session_user_id(stored_user_id, args.user_id)
    session_id = resolve_requested_session_id(requested_session_id)

    try:
        handle, events = await session_manager.start_session(
            SessionParameters(
                frontend_name="gradio",
                policy_name=policy_setting,
                scenario_name=scenario_name,
                user_id=user_id,
                session_id=session_id,
                locale=Locale(args.locale),
                call_origin=InputMode.API,
                experiment_name=experiment_name,
            )
        )
    except ValueError as ex:
        gr.Error(str(ex), duration=None)
        raise
    except Exception as ex:
        logger.exception("Failed to initialize session: %s", ex)
        gr.Error(
            f"❌ **Fatal Error**\n\n{ex}\n\nPlease refresh the page.",
            duration=None,
        )
        raise

    session = SessionRef(
        user_id=str(handle.user_id),
        session_id=str(handle.session_id),
        policy_name=handle.policy_name or policy_setting,
        scenario_name=handle.scenario_name,
        experiment_name=experiment_name,
    )
    return StartedSession(
        session=session,
        events=events,
        user_id=session.user_id,
        is_complete=events_include_completion(events),
    )
