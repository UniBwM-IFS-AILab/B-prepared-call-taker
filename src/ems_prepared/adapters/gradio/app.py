"""Gradio UI for the Emergency Call Simulator."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from functools import partial
from typing import TypedDict
from uuid import UUID, uuid4

import gradio as gr
import requests
from gradio import ChatMessage

from ems_prepared.adapters.gradio.args import GradioAppArgs
from ems_prepared.adapters.gradio.args import from_namespace as gradio_args_from_namespace
from ems_prepared.adapters.gradio.args import (
    register_arguments as register_gradio_arguments,
)
from ems_prepared.adapters.gradio.components import (
    CallStepView,
    ContextPanelView,
    IntroStepView,
    SurveyStepView,
    build_chat_section,
    build_context_panel,
    build_intro_section,
    build_survey_section,
)
from ems_prepared.adapters.gradio.core import (
    COMPLETION_MESSAGE,
    ChatMessageDict,
    construct_scenario_desc,
    get_random_scenario,
    list_md_files,
)
from ems_prepared.adapters.gradio.survey import DEFAULT_SURVEY
from ems_prepared.model.context import InputMode, Locale
from ems_prepared.model.contracts import (
    BackendEvent,
    FrontendPlugin,
    SessionManager,
    SessionParameters,
)
from ems_prepared.model.errors import SessionNotFoundError, UnsupportedPolicyError

logger = logging.getLogger(__name__)

GRADIO_CSS = """
    .icon-button-wrapper.top-panel { display: none !important; }

    .guided-outer-walkthrough {
        margin-bottom: 0.75rem;
    }

    .guided-phase > .guidance {
        margin-bottom: 0.75rem;
    }

    .guided-call-status {
        margin-bottom: 0.5rem;
    }

    .guided-continue-button[disabled],
    .guided-continue-button:disabled {
        display: none !important;
    }

    #session_info_display {
        position: fixed !important;
        bottom: 8px !important;
        right: 8px !important;
        left: auto !important;
        background: transparent !important;
        padding: 6px 10px !important;
        border: none !important;
        z-index: 1000 !important;
        pointer-events: none;
        color: var(--text-color-primary) !important;
        opacity: 0.95 !important;
        text-align: right !important;
        max-width: 40vw;
    }

    #session_info_display * {
        user-select: text;
    }
"""


class ActiveSession(TypedDict):
    """UI state fields needed to address an active backend session."""

    user_id: str
    session_id: str
    policy_name: str
    scenario_name: str | None
    experiment_name: str


@dataclass(slots=True)
class StandardDemoUI:
    """Collection of standard-mode components used for event wiring."""

    demo: gr.Blocks
    walkthrough: gr.Walkthrough
    md_picker: gr.Dropdown
    session_info_display: gr.Markdown
    active_session_state: gr.State
    initial_events_state: gr.State
    user_msg_state: gr.State
    user_id_state: gr.BrowserState
    intro: IntroStepView
    context: ContextPanelView
    chat: CallStepView
    survey: SurveyStepView


async def stream_message_to_history(
    history: list[ChatMessageDict | ChatMessage],
    message: ChatMessage,
):
    """Stream a message character-by-character into the history."""
    content = message.content if isinstance(message.content, str) else ""
    current_message = ChatMessage(
        role=message.role, content="", metadata=message.metadata
    )
    history.append(current_message)

    for char in content:
        if isinstance(current_message.content, str):
            current_message.content += char
        yield history
        await asyncio.sleep(0.01)


async def stream_backend_events(
    history: list[ChatMessageDict | ChatMessage],
    events: list[BackendEvent],
):
    """Stream backend events into chat history."""
    try:
        if not events:
            logger.error("Session manager emitted no events for this turn.")
            return

        for event in events:
            if event.kind == "completed":
                async for updated_history in stream_message_to_history(
                    history,
                    COMPLETION_MESSAGE,
                ):
                    yield updated_history
                continue

            if event.kind in {"message", "question"}:
                text = event.text or ""
            elif event.kind == "error":
                text = event.text or "❌ Internal error. Please reset the session."
            else:
                text = event.text or ""

            if not text.strip():
                continue
            async for updated_history in stream_message_to_history(
                history,
                ChatMessage(role="assistant", content=text),
            ):
                yield updated_history
    except Exception as ex:
        import traceback

        error_msg = str(ex)
        logger.error(
            "Error streaming backend events: %s\n%s",
            error_msg,
            traceback.format_exc(),
        )
        fatal_message = ChatMessage(
            role="assistant",
            content="❌ **Fatal Error**\n\nPlease click 'Reset session' to recover.",
        )
        history.append(fatal_message)
        gr.Error(error_msg, duration=None)
        yield history


async def bot_respond(
    history: list[ChatMessageDict | ChatMessage],
    session_data: ActiveSession | None,
    user_msg: str,
    *,
    session_manager: SessionManager,
):
    """Handle one user message and stream backend responses."""
    if not user_msg or not user_msg.strip():
        yield history
        return
    if session_data is None:
        history.append(
            ChatMessage(
                role="assistant",
                content="❌ Session not initialized. Please start a new session.",
            )
        )
        yield history
        return

    try:
        events = await session_manager.handle_input(
            UUID(session_data["session_id"]),
            user_msg,
        )
    except SessionNotFoundError:
        history.append(
            ChatMessage(
                role="assistant",
                content="❌ Session expired. Please reset and start again.",
            )
        )
        yield history
        return

    async for updated_history in stream_backend_events(
        history=history,
        events=events,
    ):
        yield updated_history


def set_input_interactive(enabled: bool):
    """Return updates for input box and send button."""
    return (
        gr.update(interactive=enabled, value="", autofocus=enabled),
        gr.update(interactive=False),
    )


def set_controls_interactive(enabled: bool, *, picker_interactive: bool):
    """Return updates for reset button and scenario picker."""
    return (
        gr.update(interactive=enabled),
        gr.update(interactive=picker_interactive if enabled else False),
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


def history_is_complete(history) -> bool:
    """Check whether the chat history already contains the completion marker."""
    items = getattr(history, "root", history)
    for item in reversed(list(items)):
        metadata = None
        content = None
        if isinstance(item, dict):
            maybe_metadata = item.get("metadata")
            if isinstance(maybe_metadata, dict):
                metadata = maybe_metadata
            maybe_content = item.get("content")
            if isinstance(maybe_content, str):
                content = maybe_content
        elif isinstance(item, ChatMessage):
            if isinstance(item.metadata, dict):
                metadata = item.metadata
            if isinstance(item.content, str):
                content = item.content
        else:
            maybe_metadata = getattr(item, "metadata", None)
            if isinstance(maybe_metadata, dict):
                metadata = maybe_metadata
            maybe_content = getattr(item, "content", None)
            if isinstance(maybe_content, str):
                content = maybe_content
            elif isinstance(maybe_content, list):
                text_parts = [
                    part.text
                    for part in maybe_content
                    if isinstance(getattr(part, "text", None), str)
                ]
                if text_parts:
                    content = "\n".join(text_parts)

        if metadata and metadata.get("id") == "completion_message":
            return True
        if content and content.strip() == COMPLETION_MESSAGE.content:
            return True
    return False


def resolve_or_create_user_id(stored_user_id: str) -> UUID:
    """Resolve a persisted browser user ID or create a fresh UUID."""
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
            logger.warning(
                "Invalid stored user_id: %s, generating new one", stored_user_id
            )
    return None


def initialize_user_id(
    stored_user_id: str,
    scenario_name: str | None,
    *,
    args: GradioAppArgs,
):
    """Initialize browser user ID and debug session info."""
    user_id = resolve_or_create_user_id(stored_user_id)
    return str(user_id), gr.update(
        value=session_info_text(
            args,
            user_id=str(user_id),
            scenario=scenario_name,
        ),
        visible=args.is_debug_enabled(),
    )


def set_input_and_controls(
    *,
    input_enabled: bool,
    controls_enabled: bool,
    picker_interactive: bool,
):
    """Return updates for input controls and scenario controls together."""
    return (
        *set_input_interactive(input_enabled),
        *set_controls_interactive(
            controls_enabled,
            picker_interactive=picker_interactive,
        ),
    )


async def start_standard_session(
    scenario_name: str | None,
    stored_user_id: str,
    old_session: ActiveSession | None,
    *,
    session_manager: SessionManager,
    args: GradioAppArgs,
):
    """Initialize one standard-mode session and switch to chat."""
    if old_session is not None:
        _ = await session_manager.end_session(UUID(old_session["session_id"]))

    selected_scenario = scenario_name
    if args.random_scenario:
        selected_scenario = get_random_scenario(scenario_dir=args.scenario_dir)
        logger.info("Random scenario selection: chose '%s'", selected_scenario)

    user_id = resolve_session_user_id(stored_user_id, args.user_id)
    policy_setting = args.resolve_policy()
    experiment_name = args.experiment_name or ""
    try:
        handle, events = await session_manager.start_session(
            SessionParameters(
                frontend_name="gradio",
                policy_name=policy_setting,
                scenario_name=selected_scenario,
                user_id=user_id,
                locale=Locale(args.locale),
                call_origin=InputMode.API,
                experiment_name=experiment_name,
            )
        )
        active: ActiveSession = {
            "user_id": str(handle.user_id),
            "session_id": str(handle.session_id),
            "policy_name": handle.policy_name or policy_setting,
            "scenario_name": handle.scenario_name,
            "experiment_name": experiment_name,
        }

        info_update = gr.update(
            value=session_info_text(
                args,
                user_id=active["user_id"],
                session_id=active["session_id"],
                scenario=active["scenario_name"],
                experiment_name=active["experiment_name"],
                policy=active["policy_name"],
            ),
            visible=args.is_debug_enabled(),
        )

        return active, events, active["user_id"], [], info_update
    except UnsupportedPolicyError as ex:
        gr.Error(str(ex), duration=None)
        raise
    except Exception as ex:
        import traceback

        error_msg = str(ex)
        logger.error(
            "Failed to initialize session: %s\n%s", error_msg, traceback.format_exc()
        )
        gr.Error(
            f"❌ **Fatal Error**\n\n{error_msg}\n\nPlease refresh the page.",
            duration=None,
        )
        raise


async def stream_initial_events(history: list, events: list[BackendEvent]):
    """Stream initial greeting events from the session manager."""
    async for updated in stream_backend_events(history=history, events=events):
        yield updated


async def reset_standard_session(
    old_session: ActiveSession | None,
    user_id: str,
    scenario_name: str | None,
    *,
    session_manager: SessionManager,
    args: GradioAppArgs,
):
    """Reset the active session and restore initial standard-mode UI state."""
    num_radios = len(DEFAULT_SURVEY.questions)
    if old_session is None:
        return (
            gr.skip(),
            gr.skip(),
            gr.skip(),
            *[gr.skip() for _ in range(num_radios)],
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
        )

    _ = await session_manager.end_session(UUID(old_session["session_id"]))
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
        [],
        *[gr.update(value=None, interactive=True) for _ in range(num_radios)],
        gr.update(value="", interactive=True),
        gr.update(interactive=False),
        gr.update(visible=False),
        gr.update(visible=False),
        info_update,
    )


def update_scenario_view(
    scenario_name: str | None,
    user_id: str,
    *,
    args: GradioAppArgs,
):
    """Update scenario description and debug info with selected scenario."""
    return (
        construct_scenario_desc(
            scenario_name,
            scenario_dir=args.scenario_dir,
        ),
        gr.update(
            value=session_info_text(
                args,
                user_id=user_id,
                scenario=scenario_name,
            ),
            visible=args.is_debug_enabled(),
        ),
    )


def capture_standard_user_submit(
    user_message: str,
    history: list,
    *,
    picker_interactive: bool,
):
    """Append a caller message and disable controls while processing."""
    if not user_message or not user_message.strip():
        return None, history, gr.skip(), gr.skip(), gr.skip(), gr.skip()
    history.append(ChatMessage(role="user", content=user_message))
    return (
        user_message,
        history,
        *set_input_and_controls(
            input_enabled=False,
            controls_enabled=False,
            picker_interactive=picker_interactive,
        ),
    )


def restore_standard_controls_after_response(
    history: list[ChatMessageDict | ChatMessage],
    *,
    picker_interactive: bool,
):
    """Enable controls after response unless the conversation is complete."""
    is_complete = history_is_complete(history)
    return (
        *set_input_interactive(not is_complete),
        gr.update(visible=not is_complete, interactive=not is_complete),
        gr.update(interactive=picker_interactive if not is_complete else False),
        gr.update(visible=is_complete),
    )


def check_all_survey_answers(*values):
    """Enable submit only when all survey questions are answered."""
    return gr.update(interactive=all(value is not None for value in values))


async def save_standard_survey(
    session_data: ActiveSession | None,
    feedback: str,
    *responses,
    session_manager: SessionManager,
):
    """Save survey responses and show thank-you message."""
    if session_data is None:
        logger.warning("Survey submitted but no session active")
        return (
            *[gr.update() for _ in responses],
            gr.update(),
            gr.update(),
            gr.update(visible=True),
            gr.update(
                value="### ⚠️ Session expired. Please start a new session.",
            ),
            gr.update(value="Start New Session ->", interactive=True),
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

    try:
        _ = await session_manager.submit_survey(
            UUID(session_data["session_id"]),
            responses=response,
            feedback=normalized_feedback,
        )
    except SessionNotFoundError:
        logger.warning("Survey submitted but session no longer exists")
        return (
            *[gr.update() for _ in responses],
            gr.update(),
            gr.update(),
            gr.update(visible=True),
            gr.update(
                value="### ⚠️ Session expired. Please start a new session.",
            ),
            gr.update(value="Start New Session ->", interactive=True),
        )
    logger.info("Survey saved for session %s", session_data["session_id"])

    return (
        *[gr.update(interactive=False) for _ in responses],
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(visible=True),
        gr.update(value="### ✓ Thank you for your feedback!"),
        gr.update(value="Start New Session ->", interactive=True),
    )


def build_standard_demo(session_manager: SessionManager, args: GradioAppArgs) -> gr.Blocks:
    """Build the existing single-scenario Gradio demo."""
    scenario_choices = list_md_files(scenario_dir=args.scenario_dir)
    initial_scenario = scenario_choices[0] if scenario_choices else None

    demo = gr.Blocks(title="Emergency Call Simulator", fill_height=True)
    with demo:
        gr.Markdown(
            """
            ## Emergency Call Simulator
            """
        )

        active_session_state = gr.State(value=None)
        initial_events_state = gr.State(value=[])
        user_msg_state = gr.State(value="")
        user_id_state = gr.BrowserState(default_value="", storage_key="ems_user_id")

        with gr.Row(equal_height=False):
            with gr.Column(scale=8):
                with gr.Walkthrough(selected=1, elem_id="chat_walkthrough") as walkthrough:
                    with gr.Step("Scenario Selection", id=1):
                        md_picker = gr.Dropdown(
                            choices=scenario_choices,
                            value=initial_scenario,
                            label="Scenario selection",
                            interactive=args.picker_interactive,
                            allow_custom_value=False,
                            filterable=False,
                        )
                        intro = build_intro_section(
                            intro_markdown="""
                            ### ⚠️ Before You Begin

                            #### Please read the scenario description on the right carefully.

                            Once you understand the scenario, click the button below to start the emergency call simulation.
                            """,
                            start_label="Start Emergency Call",
                            button_variant="secondary",
                        )

                    with gr.Step("Emergency Call", id=2):
                        chat = build_chat_section(
                            reset_label="Reset session",
                            continue_label="Continue to Survey",
                        )

                    with gr.Step("Survey", id=3):
                        survey = build_survey_section(
                            survey=DEFAULT_SURVEY,
                            intro_markdown="""
                            ### Please rate your experience

                            Your feedback helps us improve the emergency call system.
                            All questions are required.
                            """,
                            feedback_label="Additional feedback (optional)",
                            feedback_placeholder="Share any additional comments about your experience...",
                            submit_label="Submit Survey",
                            thanks_markdown="### ✓ Thank you for your feedback!",
                            restart_label="Start New Session ->",
                        )

            with gr.Column(scale=4):
                context = build_context_panel(
                    scenario_markdown=construct_scenario_desc(
                        initial_scenario,
                        scenario_dir=args.scenario_dir,
                    )
                )

        session_info_display = gr.Markdown(
            session_info_text(args),
            visible=args.is_debug_enabled(),
            elem_id="session_info_display",
        )

    ui = StandardDemoUI(
        demo=demo,
        walkthrough=walkthrough,
        md_picker=md_picker,
        session_info_display=session_info_display,
        active_session_state=active_session_state,
        initial_events_state=initial_events_state,
        user_msg_state=user_msg_state,
        user_id_state=user_id_state,
        intro=intro,
        context=context,
        chat=chat,
        survey=survey,
    )

    assert ui.chat.continue_button is not None
    assert ui.survey.restart_button is not None

    respond_with_manager = partial(bot_respond, session_manager=session_manager)
    on_initialize_user_id = partial(initialize_user_id, args=args)
    on_start_session = partial(
        start_standard_session,
        session_manager=session_manager,
        args=args,
    )
    on_reset_session = partial(
        reset_standard_session,
        session_manager=session_manager,
        args=args,
    )
    on_update_scenario_view = partial(update_scenario_view, args=args)
    on_capture_user_submit = partial(
        capture_standard_user_submit,
        picker_interactive=args.picker_interactive,
    )
    on_restore_controls = partial(
        restore_standard_controls_after_response,
        picker_interactive=args.picker_interactive,
    )
    on_save_survey = partial(save_standard_survey, session_manager=session_manager)

    with ui.demo:
        ui.demo.load(
            on_initialize_user_id,
            inputs=[ui.user_id_state, ui.md_picker],
            outputs=[ui.user_id_state, ui.session_info_display],
        )

        confirm_event = (
            ui.intro.start_button.click(
                on_start_session,
                inputs=[ui.md_picker, ui.user_id_state, ui.active_session_state],
                outputs=[
                    ui.active_session_state,
                    ui.initial_events_state,
                    ui.user_id_state,
                    ui.chat.chatbot,
                    ui.session_info_display,
                ],
            )
            .then(
                lambda: set_input_and_controls(
                    input_enabled=False,
                    controls_enabled=False,
                    picker_interactive=args.picker_interactive,
                ),
                outputs=[ui.chat.input_box, ui.chat.send, ui.chat.reset, ui.md_picker],
            )
            .then(
                lambda: gr.Walkthrough(selected=2),
                outputs=ui.walkthrough,
            )
        )

        greeting_event = confirm_event.then(
            stream_initial_events,
            inputs=[ui.chat.chatbot, ui.initial_events_state],
            outputs=[ui.chat.chatbot],
        ).then(
            lambda: [],
            outputs=[ui.initial_events_state],
        )

        greeting_event.success(
            lambda: set_input_and_controls(
                input_enabled=True,
                controls_enabled=True,
                picker_interactive=args.picker_interactive,
            ),
            outputs=[ui.chat.input_box, ui.chat.send, ui.chat.reset, ui.md_picker],
        )

        greeting_event.failure(
            lambda: set_input_and_controls(
                input_enabled=False,
                controls_enabled=True,
                picker_interactive=args.picker_interactive,
            ),
            outputs=[ui.chat.input_box, ui.chat.send, ui.chat.reset, ui.md_picker],
        )

        gr.on(
            triggers=[ui.chat.reset.click],
            fn=on_reset_session,
            inputs=[ui.active_session_state, ui.user_id_state, ui.md_picker],
            outputs=[
                ui.active_session_state,
                ui.initial_events_state,
                ui.chat.chatbot,
                *ui.survey.survey_radios,
                ui.survey.survey_feedback,
                ui.survey.survey_submit,
                ui.survey.completion_group,
                ui.survey.survey_thanks,
                ui.survey.restart_button,
                ui.session_info_display,
            ],
            queue=False,
        ).then(
            lambda: (
                gr.update(visible=True),
                gr.update(visible=False),
                gr.Walkthrough(selected=1),
            ),
            outputs=[ui.chat.reset, ui.chat.continue_button, ui.walkthrough],
        )

        ui.md_picker.input(
            on_update_scenario_view,
            inputs=[ui.md_picker, ui.user_id_state],
            outputs=[ui.context.scenario_markdown, ui.session_info_display],
        )

        ui.chat.bind_send_interactivity()

        submit_event = (
            gr.on(
                triggers=ui.chat.submit_triggers(),
                fn=lambda: None,
            )
            .then(
                on_capture_user_submit,
                inputs=[ui.chat.input_box, ui.chat.chatbot],
                outputs=[
                    ui.user_msg_state,
                    ui.chat.chatbot,
                    ui.chat.input_box,
                    ui.chat.send,
                    ui.chat.reset,
                    ui.md_picker,
                ],
            )
            .then(
                respond_with_manager,
                inputs=[ui.chat.chatbot, ui.active_session_state, ui.user_msg_state],
                outputs=[ui.chat.chatbot],
            )
        )

        submit_event.then(
            on_restore_controls,
            inputs=[ui.chat.chatbot],
            outputs=[
                ui.chat.input_box,
                ui.chat.send,
                ui.chat.reset,
                ui.md_picker,
                ui.chat.continue_button,
            ],
        ).failure(
            lambda: set_input_and_controls(
                input_enabled=False,
                controls_enabled=True,
                picker_interactive=args.picker_interactive,
            ),
            outputs=[ui.chat.input_box, ui.chat.send, ui.chat.reset, ui.md_picker],
        )

        ui.survey.bind_validation(
            fn=check_all_survey_answers,
            concurrency_id="survey_validation",
        )

        ui.survey.survey_submit.click(
            on_save_survey,
            inputs=[
                ui.active_session_state,
                ui.survey.survey_feedback,
                *ui.survey.survey_radios,
            ],
            outputs=[
                *ui.survey.survey_radios,
                ui.survey.survey_feedback,
                ui.survey.survey_submit,
                ui.survey.completion_group,
                ui.survey.survey_thanks,
                ui.survey.restart_button,
            ],
        )

        ui.chat.continue_button.click(
            lambda: gr.Walkthrough(selected=3),
            outputs=ui.walkthrough,
        )

        ui.survey.restart_button.click(
            on_reset_session,
            inputs=[ui.active_session_state, ui.user_id_state, ui.md_picker],
            outputs=[
                ui.active_session_state,
                ui.initial_events_state,
                ui.chat.chatbot,
                *ui.survey.survey_radios,
                ui.survey.survey_feedback,
                ui.survey.survey_submit,
                ui.survey.completion_group,
                ui.survey.survey_thanks,
                ui.survey.restart_button,
                ui.session_info_display,
            ],
        ).then(
            lambda: (
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(interactive=args.picker_interactive),
                gr.Walkthrough(selected=1),
            ),
            outputs=[ui.chat.reset, ui.chat.continue_button, ui.md_picker, ui.walkthrough],
        )

    return ui.demo


def build_demo(session_manager: SessionManager, args: GradioAppArgs) -> gr.Blocks:
    """Build the selected Gradio demo for the configured UI mode."""
    if args.ui == "guided":
        from ems_prepared.adapters.gradio.guided import build_guided_demo

        return build_guided_demo(session_manager=session_manager, args=args)
    return build_standard_demo(session_manager=session_manager, args=args)


def _notify_share_url(share_url: str) -> None:
    """Post the new share URL to Slack via incoming webhook."""
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook_url:
        logger.warning("SLACK_WEBHOOK_URL not set; skipping Slack notification")
        return

    payload = {"text": f"New *Emergency Call Simulator* URL:\n{share_url}"}

    try:
        resp = requests.post(webhook_url, json=payload, timeout=5)
        resp.raise_for_status()
        logger.info("Posted share URL to Slack")
    except requests.RequestException:
        logger.exception("Failed posting share URL to Slack")


def _launch_gradio_app(
    demo: gr.Blocks,
    args: GradioAppArgs,
) -> tuple[object, str, str | None]:
    """Launch Gradio with queueing and return launch metadata."""
    return demo.queue(default_concurrency_limit=16).launch(
        pwa=True,
        share=not args.debug,
        css=GRADIO_CSS,
        footer_links=["settings"],
        debug=args.debug,
        inbrowser=args.debug,
        show_error=True,
        prevent_thread_lock=True,
    )


def _log_launch_info(local_url: str, share_url: str | None) -> None:
    """Log launch URLs and send share URL notification when available."""
    logger.info("Gradio local URL: %s", local_url)
    logger.info("Gradio share URL: %s", share_url)
    if share_url:
        _notify_share_url(share_url)
    else:
        logger.warning("No share URL returned from Gradio")


def _block_forever() -> None:
    """Keep process attached for terminal log visibility."""
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


def run_gradio_app(session_manager: SessionManager, args: GradioAppArgs) -> None:
    """Build and launch the Gradio app with injected dependencies."""
    demo = build_demo(session_manager=session_manager, args=args)
    _app, local_url, share_url = _launch_gradio_app(demo=demo, args=args)
    _ = _app
    _log_launch_info(local_url, share_url)
    if not args.exit_on_launch:
        _block_forever()


class GradioFrontend(FrontendPlugin):
    """Frontend plugin that launches the Gradio UI."""

    def register_arguments(self, subparser: ArgumentParser, /) -> None:
        """Register Gradio frontend specific arguments."""
        register_gradio_arguments(subparser)

    def run(
        self,
        session_manager: SessionManager,
        parsed_args: Namespace,
        /,
    ) -> int | None:
        """Run Gradio with parser-composed launcher arguments."""
        run_gradio_app(
            session_manager=session_manager,
            args=gradio_args_from_namespace(parsed_args),
        )
        return 0


FRONTEND_PLUGIN = GradioFrontend()
