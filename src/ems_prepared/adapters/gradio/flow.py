"""Shared non-visual Gradio flow helpers used by the unified session UI."""

from __future__ import annotations

import logging
import re
from dataclasses import replace
from pathlib import Path
from uuid import UUID

import gradio as gr
from gradio import ChatMessage

from ems_prepared.adapters.gradio.args import GradioAppArgs
from ems_prepared.adapters.gradio.chat_stream import (
    events_include_completion,
    stream_backend_events,
)
from ems_prepared.adapters.gradio.consent import DEFAULT_CONSENT_CONTEXT_MARKDOWN
from ems_prepared.adapters.gradio.scenarios import construct_scenario_desc, list_md_files
from ems_prepared.adapters.gradio.session_runtime import (
    DEBUG_STATIC_REPLY,
    end_active_session,
    session_info_text,
    start_session_for_gradio,
)
from ems_prepared.adapters.gradio.state import CompletedScenario, GuidedState, SessionRef
from ems_prepared.adapters.gradio.survey import DEFAULT_SURVEY, submit_survey_for_gradio
from ems_prepared.model.contracts import (
    BackendEvent,
    BackendEventKind,
    SessionManager,
)

logger = logging.getLogger(__name__)

GUIDED_RUNTIME_ARTIFACT_PATTERNS = (
    "main_persistence.json",
    "*_persistence.json",
    "agent_state.json",
    "deps.json",
    "final_state.json",
    "state_schema.json",
    "message_history.json",
    "graph.md",
    "graph.jpg",
)


def _slugify_scenario_label(value: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())
    return sanitized.strip("._-") or "scenario"


def archive_guided_runtime_artifacts(
    *,
    session_manager: SessionManager,
    run_session_id: str | None,
    scenario_index: int,
    scenario_name: str | None,
) -> None:
    """Move scenario runtime artifacts into a per-scenario archive folder."""
    if run_session_id is None:
        return

    view_state = session_manager.get_view_state(UUID(run_session_id))
    if view_state is None:
        return

    save_path = view_state.save_path
    archive_root = save_path / "scenarios"
    scenario_label = _slugify_scenario_label(
        scenario_name or f"scenario_{scenario_index + 1:02d}"
    )
    archive_dir = archive_root / f"{scenario_index + 1:02d}_{scenario_label}"
    archive_dir.mkdir(parents=True, exist_ok=True)

    candidates: set[Path] = set()
    for pattern in GUIDED_RUNTIME_ARTIFACT_PATTERNS:
        candidates.update(save_path.glob(pattern))

    for source in sorted(candidates):
        if not source.is_file():
            continue
        destination = archive_dir / source.name
        if destination.exists():
            destination.unlink()
        source.replace(destination)


def resolve_guided_scenarios(args: GradioAppArgs) -> tuple[str, ...]:
    """Resolve the guided scenario order from CLI args or directory files."""
    if args.guided_scenarios:
        return args.guided_scenarios
    scenario_names = tuple(list_md_files(scenario_dir=args.scenario_dir))
    if scenario_names:
        return scenario_names
    raise ValueError("Guided mode requires at least one scenario file.")


def build_initial_guided_state(
    scenario_names: tuple[str, ...],
    *,
    consent_required: bool = False,
) -> GuidedState:
    """Create the initial shared run state for multi-scenario sessions."""
    return GuidedState(
        scenario_names=scenario_names,
        phase="consent" if consent_required else "brief",
    )


def current_guided_scenario(state: GuidedState) -> str:
    """Return the currently active scenario name."""
    return state.scenario_names[state.active_index]


def build_scenario_intro_markdown(index: int, total_scenarios: int) -> str:
    """Render brief copy shown before each scenario."""
    _ = (index, total_scenarios)
    return (
        "### Ready to Start\n\n"
        "Read the scenario brief on the right. When you are ready, start the call and stay in character until it is complete."
    )


def current_scenario_markdown(state: GuidedState, *, args: GradioAppArgs) -> str:
    """Render the active scenario markdown."""
    return construct_scenario_desc(
        current_guided_scenario(state),
        scenario_dir=args.scenario_dir,
    )


def current_context_markdown(
    state: GuidedState,
    *,
    args: GradioAppArgs,
    consent_required: bool = False,
) -> str:
    """Render sidebar markdown based on the current run phase."""
    if consent_required and state.phase == "consent":
        return DEFAULT_CONSENT_CONTEXT_MARKDOWN
    if state.phase == "survey":
        total_scenarios = len(state.scenario_names)
        if args.ui == "guided":
            if state.survey_submitted:
                return (
                    "### Guided Session Complete\n\n"
                    f"You completed all {total_scenarios} scenarios and submitted the final survey."
                )
            return (
                "### Final Survey\n\n"
                f"Please answer the survey for the full guided run across all {total_scenarios} scenarios."
            )
        return "### Final Survey\n\nPlease answer the survey for this session."
    return current_scenario_markdown(state, args=args)


def begin_guided_start(
    state: GuidedState,
    *,
    scenario_index: int,
    consent_checked: bool = False,
) -> GuidedState:
    """Enter starting phase for the selected scenario."""
    return replace(
        state,
        active_index=scenario_index,
        phase="starting",
        survey_submitted=False,
        consent_accepted=state.consent_accepted or consent_checked,
    )


def start_failure_state(state: GuidedState) -> GuidedState:
    """Restore current scenario brief after failed start."""
    return replace(state, phase="brief", active_session=None)


def prepare_guided_survey_state(state: GuidedState) -> GuidedState:
    """Append final session metadata once before survey phase."""
    active_session = state.active_session
    if active_session is None:
        return state

    completed_scenario = active_session.scenario_name or current_guided_scenario(state)
    if any(item.scenario_name == completed_scenario for item in state.completed_sessions):
        return state

    completed = CompletedScenario(scenario_name=completed_scenario)
    return replace(state, completed_sessions=[*state.completed_sessions, completed])


def advance_to_next_scenario(state: GuidedState) -> GuidedState:
    """Record current scenario completion and move to the next one."""
    updated_state = prepare_guided_survey_state(state)
    next_index = min(updated_state.active_index + 1, len(updated_state.scenario_names) - 1)
    return replace(
        updated_state,
        active_index=next_index,
        phase="brief",
        active_session=None,
    )


def enter_guided_survey(state: GuidedState) -> GuidedState:
    """Transition from final scenario completion to survey phase."""
    return replace(prepare_guided_survey_state(state), phase="survey")


def restart_current_call(state: GuidedState) -> GuidedState:
    """Clear current session while preserving run progress."""
    return replace(state, phase="brief", active_session=None)


def reset_guided_flow(
    state: GuidedState,
    user_id: str,
    *,
    consent_required: bool,
) -> GuidedState:
    """Reset full run back to the first scenario."""
    return GuidedState(
        scenario_names=state.scenario_names,
        user_id=user_id or state.user_id,
        phase="consent" if consent_required else "brief",
    )


def finalize_started_call(state: GuidedState, *, is_complete: bool) -> GuidedState:
    """Finalize start transition after greeting stream."""
    return replace(state, phase="complete" if is_complete else "call")


def finalize_guided_turn(state: GuidedState, *, is_complete: bool) -> GuidedState:
    """Finalize one chat turn after streaming finishes."""
    return replace(state, phase="complete" if is_complete else "call")


async def start_guided_session(
    state: GuidedState,
    stored_user_id: str,
    *,
    scenario_index: int,
    session_manager: SessionManager,
    args: GradioAppArgs,
) -> tuple[GuidedState, list[BackendEvent], bool, str]:
    """Start selected scenario and prepare call phase."""
    scenario_name = state.scenario_names[scenario_index]
    started = await start_session_for_gradio(
        scenario_name=scenario_name,
        stored_user_id=stored_user_id,
        previous_session=state.active_session,
        requested_session_id=state.run_session_id,
        session_manager=session_manager,
        args=args,
    )
    run_session_id = state.run_session_id or started.session.session_id

    updated_state = replace(
        state,
        user_id=started.user_id,
        run_session_id=run_session_id,
        active_index=scenario_index,
        phase="starting",
        active_session=started.session,
    )
    return updated_state, started.events, started.is_complete, started.user_id


def capture_guided_user_submit(
    user_message: str,
    history: list[ChatMessage],
):
    """Append user message and disable controls before backend turn."""
    if not user_message or not user_message.strip():
        return (
            "",
            history,
            gr.update(interactive=True, value="", autofocus=True),
            gr.update(interactive=False),
            gr.update(interactive=True),
            gr.update(visible=False, value=""),
        )

    normalized_message = user_message.strip()
    updated_history = [*history, ChatMessage(role="user", content=normalized_message)]
    return (
        normalized_message,
        updated_history,
        gr.update(interactive=False, value="", autofocus=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(visible=False, value=""),
    )


async def stream_guided_turn(
    history: list[ChatMessage],
    user_message: str,
    guided_state: GuidedState,
    *,
    scenario_index: int,
    session_manager: SessionManager,
    skip_policy_calls: bool,
):
    """Stream one chat turn for a scenario."""
    if not user_message.strip():
        yield history, False
        return

    if guided_state.active_index != scenario_index or guided_state.phase != "call":
        yield history, False
        return

    active_session = guided_state.active_session
    if active_session is None:
        error_history = [
            *history,
            ChatMessage(
                role="assistant",
                content="❌ Session not initialized. Please restart the current call.",
            ),
        ]
        yield error_history, False
        return

    if skip_policy_calls:
        events = [
            BackendEvent(kind=BackendEventKind.MESSAGE, text=DEBUG_STATIC_REPLY),
            BackendEvent(kind=BackendEventKind.COMPLETED),
        ]
        completion = True
    else:
        try:
            events = await session_manager.handle_input(
                UUID(active_session.session_id),
                user_message,
            )
        except LookupError:
            error_history = [
                *history,
                ChatMessage(
                    role="assistant",
                    content="❌ Session expired. Please restart the current call.",
                ),
            ]
            yield error_history, False
            return
        completion = events_include_completion(events)

    async for next_history in stream_backend_events(history=history, events=events):
        yield next_history, completion


async def save_guided_survey(
    guided_state: GuidedState,
    feedback: str,
    *responses,
    session_manager: SessionManager,
    skip_policy_calls: bool,
):
    """Save run survey against the final active session."""
    active_session = guided_state.active_session
    if active_session is None:
        logger.warning("Guided survey submitted but no session is active")
        return (
            guided_state,
            gr.update(),
            gr.update(),
            *[gr.update() for _ in responses],
            gr.update(),
            gr.update(),
            gr.update(visible=True),
            gr.update(
                value="### ⚠️ Session expired. Please start a new guided session.",
                visible=True,
            ),
            gr.update(
                value="Start New Guided Session",
                interactive=True,
                visible=True,
            ),
        )

    if skip_policy_calls:
        updated_state = replace(guided_state, survey_submitted=True, phase="survey")
        return (
            updated_state,
            gr.update(),
            gr.update(),
            *[gr.update(interactive=False) for _ in responses],
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(visible=True),
            gr.update(
                value="### ✓ Thank you for completing the guided session!",
                visible=True,
            ),
            gr.update(
                value="Start New Guided Session",
                interactive=True,
                visible=True,
            ),
        )

    metadata = {
        "ui_mode": "guided",
        "scenario_names": list(guided_state.scenario_names),
        "completed_scenarios": [item.scenario_name for item in guided_state.completed_sessions],
        "completed_count": len(guided_state.completed_sessions),
        "scenario_count": len(guided_state.scenario_names),
    }
    status = await submit_survey_for_gradio(
        session_manager=session_manager,
        session_id=active_session.session_id,
        survey=DEFAULT_SURVEY,
        feedback=feedback,
        responses=responses,
        metadata=metadata,
    )
    if status == "expired":
        logger.warning("Guided survey submitted but final session no longer exists")
        return (
            guided_state,
            gr.update(),
            gr.update(),
            *[gr.update() for _ in responses],
            gr.update(),
            gr.update(),
            gr.update(visible=True),
            gr.update(
                value="### ⚠️ Session expired. Please start a new guided session.",
                visible=True,
            ),
            gr.update(
                value="Start New Guided Session",
                interactive=True,
                visible=True,
            ),
        )

    updated_state = replace(guided_state, survey_submitted=True, phase="survey")
    return (
        updated_state,
        gr.update(),
        gr.update(),
        *[gr.update(interactive=False) for _ in responses],
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(visible=True),
        gr.update(
            value="### ✓ Thank you for completing the guided session!",
            visible=True,
        ),
        gr.update(
            value="Start New Guided Session",
            interactive=True,
            visible=True,
        ),
    )


async def bot_respond(
    history: list[ChatMessage],
    session_data: SessionRef | None,
    user_msg: str,
    *,
    session_manager: SessionManager,
    skip_policy_calls: bool,
):
    """Handle one standard-mode user message and stream backend responses."""
    if not user_msg.strip():
        yield history, False
        return

    if session_data is None:
        history.append(
            ChatMessage(
                role="assistant",
                content="❌ Session not initialized. Please start a new session.",
            )
        )
        yield history, False
        return

    if skip_policy_calls:
        events = [
            BackendEvent(kind=BackendEventKind.MESSAGE, text=DEBUG_STATIC_REPLY),
            BackendEvent(kind=BackendEventKind.COMPLETED),
        ]
        completed = True
    else:
        try:
            events = await session_manager.handle_input(
                UUID(session_data.session_id),
                user_msg,
            )
        except LookupError:
            history.append(
                ChatMessage(
                    role="assistant",
                    content="❌ Session expired. Please reset and start again.",
                )
            )
            yield history, False
            return
        completed = events_include_completion(events)

    async for updated_history in stream_backend_events(history=history, events=events):
        yield updated_history, completed


def restore_standard_controls_after_response(
    is_complete: bool,
    *,
    picker_interactive: bool,
):
    """Enable standard controls after response unless the conversation is complete."""
    return (
        gr.update(interactive=not is_complete, value="", autofocus=not is_complete),
        gr.update(interactive=False),
        gr.update(interactive=not is_complete),
        gr.update(interactive=picker_interactive if not is_complete else False),
        gr.update(visible=is_complete, interactive=is_complete),
    )


async def reset_standard_session(
    old_session: SessionRef | None,
    user_id: str,
    scenario_name: str | None,
    *,
    session_manager: SessionManager,
    args: GradioAppArgs,
):
    """Reset the active single-scenario session and restore initial UI state."""
    if old_session is not None:
        await end_active_session(
            old_session,
            session_manager=session_manager,
            skip_policy_calls=args.should_skip_policy_calls(),
        )

    info_update = gr.update(
        value=session_info_text(
            args,
            user_id=user_id,
            scenario=scenario_name,
        ),
        visible=args.is_debug_enabled(),
    )

    return (
        None,
        [],
        False,
        [],
        *[gr.update(value=None, interactive=True) for _ in DEFAULT_SURVEY.questions],
        gr.update(value="", interactive=True),
        gr.update(interactive=False),
        gr.Column(visible=True),
        gr.update(value="### ✓ Thank you for your feedback!", visible="hidden"),
        gr.update(value="Start New Session ->", interactive=True, visible="hidden"),
        info_update,
    )


async def save_standard_survey(
    session_data: SessionRef | None,
    feedback: str,
    *responses,
    session_manager: SessionManager,
    skip_policy_calls: bool,
):
    """Save single-session survey responses and show thank-you message."""
    if session_data is not None and skip_policy_calls:
        return (
            *[gr.update(interactive=False) for _ in responses],
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.skip(),
            gr.update(value="### ✓ Thank you for your feedback!", visible=True),
            gr.update(value="Start New Session ->", interactive=True, visible=True),
        )

    if session_data is None:
        logger.warning("Survey submitted but no session active")
        return (
            *[gr.update() for _ in responses],
            gr.update(),
            gr.update(),
            gr.skip(),
            gr.update(
                value="### ⚠️ Session expired. Please start a new session.",
                visible=True,
            ),
            gr.update(
                value="Start New Session ->",
                interactive=True,
                visible=True,
            ),
        )

    status = await submit_survey_for_gradio(
        session_manager=session_manager,
        session_id=session_data.session_id,
        survey=DEFAULT_SURVEY,
        feedback=feedback,
        responses=responses,
    )
    if status == "expired":
        logger.warning("Survey submitted but session no longer exists")
        return (
            *[gr.update() for _ in responses],
            gr.update(),
            gr.update(),
            gr.skip(),
            gr.update(
                value="### ⚠️ Session expired. Please start a new session.",
                visible=True,
            ),
            gr.update(
                value="Start New Session ->",
                interactive=True,
                visible=True,
            ),
        )

    return (
        *[gr.update(interactive=False) for _ in responses],
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.skip(),
        gr.update(value="### ✓ Thank you for your feedback!", visible=True),
        gr.update(value="Start New Session ->", interactive=True, visible=True),
    )
