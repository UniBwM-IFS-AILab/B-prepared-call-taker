"""Guided multi-scenario Gradio flow."""

from __future__ import annotations
import logging
from dataclasses import dataclass
from functools import partial
from typing import Literal, TypedDict
from uuid import UUID

import gradio as gr
from gradio import ChatMessage

from ems_prepared.adapters.gradio.app import (
    ActiveSession,
    check_all_survey_answers,
    resolve_or_create_user_id,
    resolve_session_user_id,
    session_info_text,
    stream_backend_events,
)
from ems_prepared.adapters.gradio.args import GradioAppArgs
from ems_prepared.adapters.gradio.components import (
    ContextPanelView,
    ScenarioFlowView,
    SurveyStepView,
    build_context_panel,
    build_scenario_flow,
    build_survey_section,
)
from ems_prepared.adapters.gradio.core import (
    ChatMessageDict,
    construct_scenario_desc,
    list_md_files,
)
from ems_prepared.adapters.gradio.survey import DEFAULT_SURVEY
from ems_prepared.model.context import InputMode, Locale
from ems_prepared.model.contracts import BackendEvent, SessionManager, SessionParameters
from ems_prepared.model.errors import SessionNotFoundError, UnsupportedPolicyError

logger = logging.getLogger(__name__)

GuidedPhase = Literal["brief", "starting", "call", "complete", "survey"]
TurnStatus = Literal["idle", "skip", "stream", "error"]


class CompletedScenario(TypedDict):
    """Completed guided scenario metadata."""

    scenario_name: str
    session_id: str


class GuidedRunState(TypedDict):
    """Session-backed state for the guided flow."""

    user_id: str
    scenario_names: tuple[str, ...]
    active_index: int
    phase: GuidedPhase
    survey_submitted: bool
    active_session: ActiveSession | None
    completed_sessions: list[CompletedScenario]


@dataclass(slots=True)
class GuidedDemoUI:
    """Guided-mode component handles used for event wiring."""

    demo: gr.Blocks
    walkthrough: gr.Walkthrough
    steps: list[gr.Step]
    guided_state: gr.State
    user_id_state: gr.BrowserState
    session_info_display: gr.Markdown
    context: ContextPanelView
    flows: list[ScenarioFlowView]
    survey: SurveyStepView


def resolve_guided_scenarios(args: GradioAppArgs) -> tuple[str, ...]:
    """Resolve the guided scenario order from CLI args or the directory."""
    if args.guided_scenarios:
        return args.guided_scenarios

    scenario_names = tuple(list_md_files(scenario_dir=args.scenario_dir))
    if scenario_names:
        return scenario_names
    raise ValueError("Guided mode requires at least one scenario file.")


def build_initial_guided_state(
    scenario_names: tuple[str, ...],
) -> GuidedRunState:
    """Create the initial run state for guided mode."""
    return {
        "user_id": "",
        "scenario_names": scenario_names,
        "active_index": 0,
        "phase": "brief",
        "survey_submitted": False,
        "active_session": None,
        "completed_sessions": [],
    }


def current_guided_scenario(guided_state: GuidedRunState) -> str:
    """Return the currently active guided scenario name."""
    return guided_state["scenario_names"][guided_state["active_index"]]


def is_final_guided_scenario(guided_state: GuidedRunState) -> bool:
    """Return whether the active scenario is the final guided scenario."""
    return guided_state["active_index"] == len(guided_state["scenario_names"]) - 1


def build_scenario_intro_markdown(index: int, total_scenarios: int) -> str:
    """Render the reusable per-scenario briefing copy."""
    _ = (index, total_scenarios)
    return (
        "### Ready to Start\n\n"
        "Read the scenario brief on the right. When you're ready, start the call and stay in character until it is complete."
    )


def build_scenario_complete_markdown(*, is_final: bool) -> str:
    """Render the completion copy for one scenario."""
    if is_final:
        return (
            "### Call Complete\n\n"
            "You have finished the final scenario. Continue to the survey or restart this call."
        )
    return (
        "### Call Complete\n\n"
        "You can continue to the next scenario or restart this call."
    )


def build_survey_intro_markdown(total_scenarios: int) -> str:
    """Render the final survey copy."""
    scenario_word = "scenario" if total_scenarios == 1 else "scenarios"
    return (
        "### Final Survey\n\n"
        f"Please rate the overall guided session across all {total_scenarios} {scenario_word}. "
        "All questions are required."
    )


def build_survey_context_markdown(guided_state: GuidedRunState) -> str:
    """Render the sidebar context shown during the final survey."""
    total_scenarios = len(guided_state["scenario_names"])
    if guided_state["survey_submitted"]:
        return (
            "### Guided Session Complete\n\n"
            f"You completed all {total_scenarios} scenarios and submitted the final survey."
        )
    return (
        "### Final Survey\n\n"
        f"Please answer the survey for the full guided run across all {total_scenarios} scenarios."
    )


def current_scenario_markdown(guided_state: GuidedRunState, *, args: GradioAppArgs) -> str:
    """Render the active scenario markdown."""
    return construct_scenario_desc(
        current_guided_scenario(guided_state),
        scenario_dir=args.scenario_dir,
    )


def current_context_markdown(guided_state: GuidedRunState, *, args: GradioAppArgs) -> str:
    """Render the context panel markdown for the active phase."""
    if guided_state["phase"] == "survey":
        return build_survey_context_markdown(guided_state)
    return current_scenario_markdown(guided_state, args=args)


def events_include_completion(events: list[BackendEvent]) -> bool:
    """Return whether the backend emitted the completion event."""
    return any(event.kind == "completed" for event in events)


def guided_session_info_update(
    guided_state: GuidedRunState,
    *,
    args: GradioAppArgs,
):
    """Build the debug session info update for guided mode."""
    active_session = (
        guided_state["active_session"]
        if guided_state["phase"] in {"call", "complete", "survey"}
        else None
    )
    scenario_name = (
        active_session["scenario_name"]
        if active_session is not None
        else current_guided_scenario(guided_state)
    )
    return gr.update(
        value=session_info_text(
            args,
            user_id=guided_state["user_id"] or None,
            session_id=active_session["session_id"] if active_session else None,
            scenario=scenario_name,
            experiment_name=(
                active_session["experiment_name"] if active_session else None
            ),
            policy=active_session["policy_name"] if active_session else None,
        ),
        visible=args.is_debug_enabled(),
    )


def guided_context_updates(
    guided_state: GuidedRunState,
    *,
    context: ContextPanelView,
    session_info_display: gr.Markdown,
    args: GradioAppArgs,
) -> dict[object, object]:
    """Refresh the persistent guided context panel from session state."""
    updates = context.updates(
        scenario_markdown=current_context_markdown(guided_state, args=args),
    )
    updates[session_info_display] = guided_session_info_update(guided_state, args=args)
    return updates


def guided_step_interactivity_updates(
    steps: list[gr.Step],
    *,
    selected_step_id: int,
) -> dict[object, object]:
    """Enable only the active walkthrough step button."""
    return {
        step: gr.Step(interactive=index == selected_step_id)
        for index, step in enumerate(steps, start=1)
    }


def initialize_guided_user_id(
    stored_user_id: str,
    guided_state: GuidedRunState,
):
    """Initialize browser user ID and guided state on first page load."""
    user_id = str(resolve_or_create_user_id(stored_user_id))
    return {
        **guided_state,
        "user_id": user_id,
    }, user_id


def begin_guided_start(
    guided_state: GuidedRunState,
    *,
    scenario_index: int,
) -> GuidedRunState:
    """Enter the starting phase for the selected scenario."""
    return {
        **guided_state,
        "active_index": scenario_index,
        "phase": "starting",
        "survey_submitted": False,
    }


def start_failure_state(guided_state: GuidedRunState) -> GuidedRunState:
    """Restore the current scenario brief after a failed start attempt."""
    return {
        **guided_state,
        "phase": "brief",
        "active_session": None,
    }


def prepare_guided_survey_state(guided_state: GuidedRunState) -> GuidedRunState:
    """Record the final completed scenario before entering the survey."""
    active_session = guided_state["active_session"]
    if active_session is None:
        return guided_state

    session_id = active_session["session_id"]
    if any(
        completed_session["session_id"] == session_id
        for completed_session in guided_state["completed_sessions"]
    ):
        return guided_state

    return {
        **guided_state,
        "completed_sessions": [
            *guided_state["completed_sessions"],
            {
                "scenario_name": active_session["scenario_name"]
                or current_guided_scenario(guided_state),
                "session_id": session_id,
            },
        ],
    }


def advance_to_next_scenario(guided_state: GuidedRunState) -> GuidedRunState:
    """Record the completed scenario and move the active index forward."""
    updated_state = prepare_guided_survey_state(guided_state)
    next_index = min(
        updated_state["active_index"] + 1,
        len(updated_state["scenario_names"]) - 1,
    )
    return {
        **updated_state,
        "active_index": next_index,
        "phase": "brief",
        "active_session": None,
    }


def enter_guided_survey(guided_state: GuidedRunState) -> GuidedRunState:
    """Transition from the final completed scenario into the survey."""
    updated_state = prepare_guided_survey_state(guided_state)
    return {
        **updated_state,
        "phase": "survey",
    }


def restart_current_call(guided_state: GuidedRunState) -> GuidedRunState:
    """Clear the active session while preserving guided progress."""
    return {
        **guided_state,
        "phase": "brief",
        "active_session": None,
    }


def reset_guided_flow(
    guided_state: GuidedRunState,
    user_id: str,
) -> GuidedRunState:
    """Reset the guided flow back to the first scenario."""
    return {
        **build_initial_guided_state(guided_state["scenario_names"]),
        "user_id": user_id or guided_state["user_id"],
    }


def finalize_started_call(
    guided_state: GuidedRunState,
    *,
    is_complete: bool,
) -> GuidedRunState:
    """Finalize the start flow after greeting streaming completes."""
    return {
        **guided_state,
        "phase": "complete" if is_complete else "call",
    }


def finalize_guided_turn_state(
    guided_state: GuidedRunState,
    *,
    turn_status: TurnStatus,
    is_complete: bool,
) -> GuidedRunState:
    """Finalize one user turn after backend events finish streaming."""
    if turn_status == "error":
        return {
            **guided_state,
            "phase": "call",
            "active_session": None,
        }
    if turn_status == "complete" or is_complete:
        return {
            **guided_state,
            "phase": "complete",
        }
    return {
        **guided_state,
        "phase": "call",
    }


async def close_guided_session(
    active_session: ActiveSession | None,
    *,
    session_manager: SessionManager,
) -> None:
    """Best-effort cleanup for one guided session."""
    if active_session is None:
        return

    try:
        _ = await session_manager.end_session(UUID(active_session["session_id"]))
    except Exception:
        logger.exception("Failed ending guided session %s", active_session["session_id"])


async def start_guided_session(
    guided_state: GuidedRunState,
    stored_user_id: str,
    *,
    scenario_index: int,
    session_manager: SessionManager,
    args: GradioAppArgs,
) -> tuple[GuidedRunState, list[BackendEvent], bool, str]:
    """Start the active guided scenario and prepare the chat phase."""
    await close_guided_session(guided_state["active_session"], session_manager=session_manager)

    scenario_name = guided_state["scenario_names"][scenario_index]
    user_id = resolve_session_user_id(stored_user_id, args.user_id)
    policy_setting = args.resolve_policy()
    experiment_name = args.experiment_name or ""

    try:
        handle, events = await session_manager.start_session(
            SessionParameters(
                frontend_name="gradio",
                policy_name=policy_setting,
                scenario_name=scenario_name,
                user_id=user_id,
                locale=Locale(args.locale),
                call_origin=InputMode.API,
                experiment_name=experiment_name,
            )
        )
    except UnsupportedPolicyError as ex:
        gr.Error(str(ex), duration=None)
        raise
    except Exception as ex:
        import traceback

        error_msg = str(ex)
        logger.error(
            "Failed to initialize guided session: %s\n%s",
            error_msg,
            traceback.format_exc(),
        )
        gr.Error(
            f"❌ **Fatal Error**\n\n{error_msg}\n\nPlease refresh the page.",
            duration=None,
        )
        raise

    active_session: ActiveSession = {
        "user_id": str(handle.user_id),
        "session_id": str(handle.session_id),
        "policy_name": handle.policy_name or policy_setting,
        "scenario_name": handle.scenario_name,
        "experiment_name": experiment_name,
    }
    updated_state: GuidedRunState = {
        **guided_state,
        "user_id": active_session["user_id"],
        "active_index": scenario_index,
        "phase": "starting",
        "active_session": active_session,
    }
    completion = events_include_completion(events)
    return updated_state, events, completion, active_session["user_id"]


def capture_guided_user_submit(
    user_message: str,
    history: list[ChatMessageDict | ChatMessage],
):
    """Capture the user message immediately and disable controls before streaming."""
    if not user_message or not user_message.strip():
        return (
            "",
            history,
            "skip",
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
        "stream",
        gr.update(interactive=False, value="", autofocus=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(visible=False, value=""),
    )


async def stream_guided_turn(
    history: list[ChatMessageDict | ChatMessage],
    turn_status: TurnStatus,
    user_message: str,
    guided_state: GuidedRunState,
    *,
    scenario_index: int,
    session_manager: SessionManager,
):
    """Stream one guided chat turn using the official append-and-mutate chatbot pattern."""
    if turn_status == "skip":
        yield history, False, "skip"
        return

    if guided_state["active_index"] != scenario_index or guided_state["phase"] != "call":
        yield history, False, "skip"
        return

    active_session = guided_state["active_session"]
    if active_session is None:
        error_history = [
            *history,
            ChatMessage(
                role="assistant",
                content="❌ Session not initialized. Please restart the current call.",
            ),
        ]
        yield error_history, False, "error"
        return

    try:
        events = await session_manager.handle_input(
            UUID(active_session["session_id"]),
            user_message,
        )
    except SessionNotFoundError:
        events = [
            BackendEvent(
                kind="error",
                text="❌ Session expired. Please restart the current call.",
            )
        ]
        completion = False
        resolved_turn_status: TurnStatus = "error"
    except Exception as ex:
        logger.exception("Guided session turn failed")
        events = [
            BackendEvent(
                kind="error",
                text=f"❌ Fatal error: {ex}",
            )
        ]
        completion = False
        resolved_turn_status = "error"
    else:
        if not events:
            events = [
                BackendEvent(
                    kind="error",
                    text="❌ No response received. Please restart the current call.",
                )
            ]
            completion = False
            resolved_turn_status = "error"
        else:
            completion = events_include_completion(events)
            resolved_turn_status = "stream"

    async for next_history in stream_backend_events(
        history=history,
        events=events,
    ):
        yield next_history, completion, resolved_turn_status


async def save_guided_survey(
    guided_state: GuidedRunState,
    feedback: str,
    *responses,
    session_manager: SessionManager,
):
    """Save the single guided survey against the final active session."""
    active_session = guided_state["active_session"]
    if active_session is None:
        logger.warning("Guided survey submitted but no session is active")
        return (
            guided_state,
            *[gr.update() for _ in responses],
            gr.update(),
            gr.update(),
            gr.update(visible=True),
            gr.update(
                value="### ⚠️ Session expired. Please start a new guided session.",
            ),
            gr.update(
                value="Start New Guided Session",
                interactive=True,
            ),
        )

    response = [
        {
            "label": question.label,
            "category": question.category,
            "question": question.text,
            "score": score,
        }
        for question, score in zip(DEFAULT_SURVEY.questions, responses)
    ]
    normalized_feedback = feedback.strip() if feedback and feedback.strip() else None
    metadata = {
        "ui_mode": "guided",
        "scenario_names": list(guided_state["scenario_names"]),
        "session_ids": [
            completed_session["session_id"]
            for completed_session in guided_state["completed_sessions"]
        ],
        "scenario_count": len(guided_state["scenario_names"]),
    }

    try:
        _ = await session_manager.submit_survey(
            UUID(active_session["session_id"]),
            responses=response,
            feedback=normalized_feedback,
            metadata=metadata,
        )
    except SessionNotFoundError:
        logger.warning("Guided survey submitted but final session no longer exists")
        return (
            guided_state,
            *[gr.update() for _ in responses],
            gr.update(),
            gr.update(),
            gr.update(visible=True),
            gr.update(
                value="### ⚠️ Session expired. Please start a new guided session.",
            ),
            gr.update(
                value="Start New Guided Session",
                interactive=True,
            ),
        )

    updated_state = {
        **guided_state,
        "survey_submitted": True,
        "phase": "survey",
    }
    return (
        updated_state,
        *[gr.update(interactive=False) for _ in responses],
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(visible=True),
        gr.update(
            value="### ✓ Thank you for completing the guided session!",
        ),
        gr.update(
            value="Start New Guided Session",
            interactive=True,
        ),
    )


def build_guided_demo(session_manager: SessionManager, args: GradioAppArgs) -> gr.Blocks:
    """Build the guided scenario Gradio UI."""
    available_scenarios = set(list_md_files(scenario_dir=args.scenario_dir))
    scenario_names = resolve_guided_scenarios(args)
    missing_scenarios = [
        scenario_name
        for scenario_name in scenario_names
        if scenario_name not in available_scenarios
    ]
    if missing_scenarios:
        missing_list = ", ".join(missing_scenarios)
        raise ValueError(f"Unknown guided scenarios: {missing_list}")

    initial_state = build_initial_guided_state(scenario_names)
    initial_scenario = current_guided_scenario(initial_state)

    demo = gr.Blocks(title="Emergency Call Simulator", fill_height=True)
    with demo:
        gr.Markdown(
            """
            ## Emergency Call Simulator
            """
        )
        guided_state = gr.State(value=initial_state)
        user_id_state = gr.BrowserState(default_value="", storage_key="ems_user_id")
        initial_events_state = gr.State(value=[])
        initial_completion_state = gr.State(value=False)
        turn_message_state = gr.State(value="")
        turn_events_state = gr.State(value=[])
        turn_completion_state = gr.State(value=False)
        turn_status_state = gr.State(value="idle")

        with gr.Row(equal_height=False):
            with gr.Column(scale=8):
                with gr.Walkthrough(
                    selected=1,
                    elem_id="guided_session_walkthrough",
                    elem_classes=["guided-outer-walkthrough"],
                ) as walkthrough:
                    flows: list[ScenarioFlowView] = []
                    steps: list[gr.Step] = []
                    total_scenarios = len(scenario_names)
                    for index in range(total_scenarios):
                        with gr.Step(
                            f"Scenario {index + 1}",
                            id=index + 1,
                            interactive=index == 0,
                        ) as scenario_step:
                            steps.append(scenario_step)
                            flows.append(
                                build_scenario_flow(
                                    intro_markdown=build_scenario_intro_markdown(
                                        index,
                                        total_scenarios,
                                    ),
                                    start_label="Start Scenario",
                                    reset_label="Restart Current Call",
                                    completion_markdown=build_scenario_complete_markdown(
                                        is_final=index == total_scenarios - 1,
                                    ),
                                    advance_label=(
                                        "Continue to Survey"
                                        if index == total_scenarios - 1
                                        else "Continue to Next Scenario"
                                    ),
                                )
                            )

                    with gr.Step(
                        "Survey",
                        id=total_scenarios + 1,
                        interactive=False,
                    ) as survey_step:
                        steps.append(survey_step)
                        survey = build_survey_section(
                            survey=DEFAULT_SURVEY,
                            intro_markdown=build_survey_intro_markdown(total_scenarios),
                            feedback_label="Additional feedback (optional)",
                            feedback_placeholder="Share any comments about the full guided session...",
                            submit_label="Submit Survey",
                            thanks_markdown="### ✓ Thank you for completing the guided session!",
                            restart_label="Start New Guided Session",
                        )

            with gr.Column(scale=4):
                context = build_context_panel(
                    scenario_markdown=current_context_markdown(initial_state, args=args),
                )

        session_info_display = gr.Markdown(
            session_info_text(args, scenario=initial_scenario),
            visible=args.is_debug_enabled(),
            elem_id="session_info_display",
        )

    ui = GuidedDemoUI(
        demo=demo,
        walkthrough=walkthrough,
        steps=steps,
        guided_state=guided_state,
        user_id_state=user_id_state,
        session_info_display=session_info_display,
        context=context,
        flows=flows,
        survey=survey,
    )

    survey_step_id = len(ui.flows) + 1
    step_outputs = ui.steps
    all_flow_outputs = [component for flow in ui.flows for component in flow.outputs()]
    context_outputs = [*ui.context.outputs(), ui.session_info_display]
    survey_outputs = ui.survey.outputs()

    on_context_update = partial(
        guided_context_updates,
        context=ui.context,
        session_info_display=ui.session_info_display,
        args=args,
    )

    with ui.demo:
        ui.demo.load(
            initialize_guided_user_id,
            inputs=[ui.user_id_state, ui.guided_state],
            outputs=[ui.guided_state, ui.user_id_state],
        ).then(
            on_context_update,
            inputs=[ui.guided_state],
            outputs=context_outputs,
            queue=False,
        )

        for index, flow in enumerate(ui.flows):
            flow.call.bind_send_interactivity()

            prepare_start_outputs = [ui.guided_state, *context_outputs, *flow.outputs()]
            start_session_outputs = [
                ui.guided_state,
                ui.user_id_state,
                initial_events_state,
                initial_completion_state,
                *context_outputs,
            ]
            finalize_start_outputs = [ui.guided_state, *context_outputs, *flow.outputs()]
            restart_outputs = [ui.guided_state, *context_outputs, *flow.outputs()]
            non_final_advance_outputs = [
                ui.guided_state,
                *step_outputs,
                *context_outputs,
                *ui.flows[index + 1].outputs(),
            ] if index < len(ui.flows) - 1 else [
                ui.guided_state,
                *step_outputs,
                *context_outputs,
                *survey_outputs,
            ]
            capture_turn_outputs = [
                turn_message_state,
                flow.call.chatbot,
                turn_status_state,
                flow.call.input_box,
                flow.call.send,
                flow.call.restart_button,
                flow.call.status_markdown,
            ]
            begin_turn_outputs = [ui.guided_state, *context_outputs, *flow.outputs()]
            finalize_turn_outputs = [ui.guided_state, *context_outputs, *flow.outputs()]

            def on_prepare_start(
                state: GuidedRunState,
                *,
                scenario_index: int = index,
                scenario_flow: ScenarioFlowView = flow,
            ) -> dict[object, object]:
                updated_state = begin_guided_start(state, scenario_index=scenario_index)
                updates = {ui.guided_state: updated_state}
                updates |= scenario_flow.starting_updates()
                updates |= on_context_update(updated_state)
                return updates

            async def on_start_failure(
                state: GuidedRunState,
                *,
                scenario_flow: ScenarioFlowView = flow,
            ) -> dict[object, object]:
                updated_state = start_failure_state(state)
                updates = {ui.guided_state: updated_state}
                updates |= scenario_flow.brief_updates()
                updates |= on_context_update(updated_state)
                return updates

            async def on_start_scenario(
                state: GuidedRunState,
                stored_user_id: str,
                *,
                scenario_index: int = index,
            ):
                updated_state, events, completion, resolved_user_id = await start_guided_session(
                    state,
                    stored_user_id,
                    scenario_index=scenario_index,
                    session_manager=session_manager,
                    args=args,
                )
                updates = {
                    ui.guided_state: updated_state,
                    ui.user_id_state: resolved_user_id,
                    initial_events_state: events,
                    initial_completion_state: completion,
                }
                updates |= on_context_update(updated_state)
                return updates

            def on_finalize_start(
                state: GuidedRunState,
                is_complete: bool,
                *,
                scenario_flow: ScenarioFlowView = flow,
            ) -> dict[object, object]:
                updated_state = finalize_started_call(state, is_complete=is_complete)
                updates = {ui.guided_state: updated_state}
                updates |= (
                    scenario_flow.complete_updates()
                    if is_complete
                    else scenario_flow.call_updates()
                )
                updates |= on_context_update(updated_state)
                return updates

            async def on_restart_scenario(
                state: GuidedRunState,
                *,
                scenario_index: int = index,
                scenario_flow: ScenarioFlowView = flow,
            ) -> dict[object, object]:
                if state["active_index"] != scenario_index:
                    return {ui.guided_state: state} | on_context_update(state)
                await close_guided_session(state["active_session"], session_manager=session_manager)
                updated_state = restart_current_call(state)
                updates = {ui.guided_state: updated_state}
                updates |= scenario_flow.brief_updates()
                updates |= on_context_update(updated_state)
                return updates

            async def on_advance_scenario(
                state: GuidedRunState,
                *,
                scenario_index: int = index,
            ) -> dict[object, object]:
                if state["active_index"] != scenario_index:
                    selected_step_id = (
                        survey_step_id if state["phase"] == "survey" else state["active_index"] + 1
                    )
                    return {
                        ui.guided_state: state,
                    } | guided_step_interactivity_updates(
                        ui.steps,
                        selected_step_id=selected_step_id,
                    ) | on_context_update(state)
                next_step_id = (
                    survey_step_id if scenario_index == len(ui.flows) - 1 else scenario_index + 2
                )
                if scenario_index == len(ui.flows) - 1:
                    updated_state = enter_guided_survey(state)
                    updates = {
                        ui.guided_state: updated_state,
                    }
                    updates |= guided_step_interactivity_updates(
                        ui.steps,
                        selected_step_id=next_step_id,
                    )
                    updates |= ui.survey.reset_updates()
                    updates |= on_context_update(updated_state)
                    return updates

                await close_guided_session(state["active_session"], session_manager=session_manager)
                updated_state = advance_to_next_scenario(state)
                updates = {
                    ui.guided_state: updated_state,
                }
                updates |= guided_step_interactivity_updates(
                    ui.steps,
                    selected_step_id=next_step_id,
                )
                updates |= ui.flows[scenario_index + 1].brief_updates()
                updates |= on_context_update(updated_state)
                return updates

            def on_begin_turn_processing(
                state: GuidedRunState,
                turn_status: TurnStatus,
                *,
                scenario_index: int = index,
                scenario_flow: ScenarioFlowView = flow,
            ) -> dict[object, object]:
                if (
                    state["active_index"] != scenario_index
                    or state["phase"] != "call"
                    or turn_status != "stream"
                ):
                    return {ui.guided_state: state} | on_context_update(state)
                updates = {ui.guided_state: state}
                updates |= scenario_flow.processing_updates()
                updates |= on_context_update(state)
                return updates

            def on_finalize_turn(
                state: GuidedRunState,
                turn_status: TurnStatus,
                is_complete: bool,
                *,
                scenario_index: int = index,
                scenario_flow: ScenarioFlowView = flow,
            ) -> dict[object, object]:
                if state["active_index"] != scenario_index:
                    return {ui.guided_state: state} | on_context_update(state)
                if turn_status == "skip":
                    return {ui.guided_state: state} | on_context_update(state)
                updated_state = finalize_guided_turn_state(
                    state,
                    turn_status=turn_status,
                    is_complete=is_complete,
                )
                updates = {ui.guided_state: updated_state}
                updates |= (
                    scenario_flow.call_error_updates()
                    if turn_status == "error"
                    else scenario_flow.complete_updates()
                    if is_complete
                    else scenario_flow.call_updates()
                )
                updates |= on_context_update(updated_state)
                return updates

            start_prepare_event = flow.intro.start_button.click(
                on_prepare_start,
                inputs=[ui.guided_state],
                outputs=prepare_start_outputs,
                show_progress="hidden",
                queue=False,
            )
            start_session_event = start_prepare_event.then(
                on_start_scenario,
                inputs=[ui.guided_state, ui.user_id_state],
                outputs=start_session_outputs,
                show_progress="hidden",
            )
            start_session_event.failure(
                on_start_failure,
                inputs=[ui.guided_state],
                outputs=finalize_start_outputs,
                show_progress="hidden",
                queue=False,
            )
            start_greeting_event = start_session_event.then(
                stream_backend_events,
                inputs=[flow.call.chatbot, initial_events_state],
                outputs=[flow.call.chatbot],
                show_progress="hidden",
            )
            start_greeting_event.then(
                lambda: [],
                outputs=[initial_events_state],
                show_progress="hidden",
                queue=False,
            )
            start_greeting_event.then(
                on_finalize_start,
                inputs=[ui.guided_state, initial_completion_state],
                outputs=finalize_start_outputs,
                show_progress="hidden",
                queue=False,
            ).then(
                lambda: False,
                outputs=[initial_completion_state],
                show_progress="hidden",
                queue=False,
            )

            submit_event = gr.on(
                triggers=flow.call.submit_triggers(),
                fn=capture_guided_user_submit,
                inputs=[flow.call.input_box, flow.call.chatbot],
                outputs=capture_turn_outputs,
                show_progress="hidden",
                queue=False,
            )
            begin_turn_event = submit_event.then(
                on_begin_turn_processing,
                inputs=[ui.guided_state, turn_status_state],
                outputs=begin_turn_outputs,
                show_progress="hidden",
                queue=False,
            )
            stream_turn_event = begin_turn_event.then(
                partial(
                    stream_guided_turn,
                    scenario_index=index,
                    session_manager=session_manager,
                ),
                inputs=[flow.call.chatbot, turn_status_state, turn_message_state, ui.guided_state],
                outputs=[flow.call.chatbot, turn_completion_state, turn_status_state],
            )
            stream_turn_event.then(
                on_finalize_turn,
                inputs=[ui.guided_state, turn_status_state, turn_completion_state],
                outputs=finalize_turn_outputs,
                show_progress="hidden",
                queue=False,
            ).then(
                lambda: ("", False, "idle"),
                outputs=[turn_message_state, turn_completion_state, turn_status_state],
                show_progress="hidden",
                queue=False,
            )

            flow.call.restart_button.click(
                on_restart_scenario,
                inputs=[ui.guided_state],
                outputs=restart_outputs,
                show_progress="hidden",
                queue=False,
            )

            advance_event = flow.call.continue_button.click(
                on_advance_scenario,
                inputs=[ui.guided_state],
                outputs=non_final_advance_outputs,
                show_progress="hidden",
                queue=False,
            )
            advance_event.then(
                lambda step_id=(survey_step_id if index == len(ui.flows) - 1 else index + 2): gr.Walkthrough(selected=step_id),
                outputs=ui.walkthrough,
                show_progress="hidden",
                queue=False,
            )

        ui.survey.bind_validation(
            fn=check_all_survey_answers,
            concurrency_id="guided_survey_validation",
        )

        ui.survey.survey_submit.click(
            partial(save_guided_survey, session_manager=session_manager),
            inputs=[
                ui.guided_state,
                ui.survey.survey_feedback,
                *ui.survey.survey_radios,
            ],
            outputs=[ui.guided_state, *survey_outputs],
            show_progress="hidden",
        ).then(
            on_context_update,
            inputs=[ui.guided_state],
            outputs=context_outputs,
            show_progress="hidden",
            queue=False,
        )

        async def on_restart_guided_run(
            state: GuidedRunState,
            stored_user_id: str,
        ) -> dict[object, object]:
            await close_guided_session(state["active_session"], session_manager=session_manager)
            updated_state = reset_guided_flow(state, stored_user_id)
            updates = {
                ui.guided_state: updated_state,
            }
            updates |= guided_step_interactivity_updates(ui.steps, selected_step_id=1)
            for scenario_flow in ui.flows:
                updates |= scenario_flow.brief_updates()
            updates |= ui.survey.reset_updates()
            updates |= on_context_update(updated_state)
            return updates

        assert ui.survey.restart_button is not None
        ui.survey.restart_button.click(
            on_restart_guided_run,
            inputs=[ui.guided_state, ui.user_id_state],
            outputs=[
                ui.guided_state,
                *step_outputs,
                *context_outputs,
                *all_flow_outputs,
                *survey_outputs,
            ],
            show_progress="hidden",
            queue=False,
        ).then(
            lambda: gr.Walkthrough(selected=1),
            outputs=ui.walkthrough,
            show_progress="hidden",
            queue=False,
        )

    return ui.demo
