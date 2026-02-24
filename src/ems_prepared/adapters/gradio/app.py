"""Gradio UI for the Emergency Call Simulator.

This module contains the Gradio-specific UI components and event handlers.
For reusable logic, see the `core` module.
"""

import asyncio
import logging
import os
import time
from uuid import UUID

import gradio as gr
import requests
from gradio import ChatMessage
from pydantic_graph.graph import End, Graph

from ems_prepared.adapters.cli import args
from ems_prepared.adapters.gradio.core import (
    COMPLETION_MESSAGE,
    ChatMessageDict,
    cleanup_session,
    construct_scenario_desc,
    get_random_scenario,
    init_session,
    invoke_agent,
    invoke_graph,
    list_md_files,
)
from ems_prepared.adapters.gradio.survey import (
    DEFAULT_SURVEY,
    save_survey_responses,
)
from ems_prepared.policies.llm_only.agent import AgentPolicy
from ems_prepared.util.logger import flush_logger
from ems_prepared.util.settings import Settings

# TODO: add retry button, so that we can repeat a reuqest if all models throw an exception due to overload
# TODO: FIXME: currently, gr.error does not (always?) enable the reset button

logger = logging.getLogger(__name__)

# =============================================================================
# Streaming Helpers
# =============================================================================


async def stream_message_to_history(
    history: list[ChatMessageDict | ChatMessage],
    message: ChatMessage,
):
    """Stream a message character-by-character into the history.

    Adds a new message bubble and yields the history after each character.

    Args:
        history: Chat history
        message: ChatMessage to stream
    """
    content = message.content if isinstance(message.content, str) else ""
    role = message.role
    metadata = message.metadata

    # Append empty message bubble with metadata
    current_message = ChatMessage(role=role, content="", metadata=metadata)
    history.append(current_message)

    # Stream character by character
    for char in content:
        # Build up content string
        if isinstance(current_message.content, str):
            current_message.content += char
        yield history
        await asyncio.sleep(0.01)  # Small delay to make streaming visible


async def stream_policy_messages(
    history: list[ChatMessageDict | ChatMessage],
    policy: Graph | AgentPolicy,
    deps: Settings,
    user_msg: str | None = None,
):
    """Stream messages from the policy generator into history.

    Args:
        history: Current chat history
        policy: The policy to run
        deps: Settings/dependencies
        user_msg: User message to process (None for initial greeting)

    Yields:
        Updated history after each character is streamed
    """
    try:
        # Check if using agent policy (tuple) or policy policy (single object)
        if isinstance(policy, Graph):
            async_generator = invoke_graph(policy, deps, user_msg)
        else:
            async_generator = invoke_agent(policy, deps, user_msg)

        # Log session metadata at the start of the dialogue
        if not history:
            deps.state_logger.info(
                {
                    "session": {
                        "policy": deps.policy_name or args.policy,
                        "scenario": deps.scenario_name or "unspecified",
                    }
                },
                extra={"event": "metadata"},
            )
        if user_msg:
            deps.logger.info(f"User message: {user_msg}")

        streamed_any = False
        async for item in async_generator:
            streamed_any = True
            if isinstance(item, End):
                # Stream completion message
                async for updated_history in stream_message_to_history(
                    history, COMPLETION_MESSAGE
                ):
                    yield updated_history
                deps.messages_logger.info(
                    "",
                    extra={
                        "speaker": "operator",
                        "msg_text": COMPLETION_MESSAGE.content,
                    },
                )
                flush_logger(deps.state_logger)
                flush_logger(deps.messages_logger)
                break
            elif isinstance(item, str):
                deps.logger.debug(f"Streaming message: {item}")

                # Stream the message (either emitted message or question)
                async for updated_history in stream_message_to_history(
                    history, ChatMessage(role="assistant", content=item)
                ):
                    yield updated_history
        if not streamed_any:
            deps.logger.error(
                "Policy async_generator produced no items. "
                f"len(history)={len(history)}, policy_type={type(policy).__name__}"
            )
    except Exception as e:
        import traceback

        error_msg = str(e)
        deps.logger.error(
            f"Error streaming messages: {error_msg}\n{traceback.format_exc()}"
        )
        history.append(
            ChatMessage(
                role="assistant",
                content="❌ **Fatal Error**\n\nPlease click 'Reset session' to recover.",  # {error_msg}\n\n
            )
        )
        gr.Error(error_msg, duration=None)
        yield history


async def bot_respond(
    history: list[ChatMessageDict | ChatMessage],
    policy,
    deps: Settings,
    user_msg: str,
):
    """Stream responses from policy as messages are emitted.

    Args:
        history: Chat history
        policy: The policy to run
        deps: Settings/dependencies
        user_msg: The user's message to process
    """
    # Skip if message is empty (happens when message to user_submit was empty)
    if not user_msg or not user_msg.strip():
        yield history
        return

    # Stream messages using the helper
    async for updated_history in stream_policy_messages(
        history=history,
        policy=policy,
        deps=deps,
        user_msg=user_msg,
    ):
        yield updated_history


def set_input_interactive(enabled: bool):
    """Return updates for input_box and send button.

    Note: Send button is always disabled here since input is cleared.
    The input_box.input event handler enables send when user types.
    """
    return (
        gr.update(interactive=enabled, value="", autofocus=enabled),
        gr.update(interactive=False),  # Always disabled; enabled via input event
    )


def set_controls_interactive(enabled: bool):
    """Return updates for reset button and md_picker."""
    return (
        gr.update(interactive=enabled),  # reset
        gr.update(
            interactive=args.picker_interactive if enabled else False
        ),  # md_picker
    )


def format_session_info(
    user_id: str | None = None,
    session_id: str | None = None,
    scenario: str | None = None,
    experiment_name: str | None = None,
    policy: str | None = None,
) -> str:
    """Format session debug info string.

    Args:
        user_id: User ID to display, or None for placeholder.
        session_id: Session ID to display, or None for placeholder.
        scenario: Scenario name to display, or None for placeholder.
        experiment_name: Experiment name, defaults to args.experiment_name if None.
        policy: Policy name, defaults to args.policy if None.

    Returns:
        Formatted markdown string with session info.
    """
    # Use args defaults only when not explicitly provided
    experiment = (
        experiment_name if experiment_name is not None else args.experiment_name
    )
    policy_name = policy if policy is not None else args.policy

    user_id_str = f"`{user_id}`" if user_id else "_initializing..._"
    session_id_str = f"`{session_id}`" if session_id else "_pending session_"
    scenario_str = f"`{scenario or 'None'}`"

    return (
        f"**Experiment:** `{experiment or 'None'}`  \n"
        f"**User ID:** {user_id_str}  \n"
        f"**Session ID:** {session_id_str}  \n"
        f"**Policy:** `{policy_name}`  \n"
        f"**Scenario:** {scenario_str}"
    )


# =============================================================================
# UI with Session State
# =============================================================================

demo = gr.Blocks(
    title="Emergency Call Simulator",
    fill_height=True,
)  # demo is declared separately to work around bug with hot-reloading

with demo:
    gr.Markdown(
        """
        ## Emergency Call Simulator
        """
    )

    # Session state for policy and deps
    policy_state = gr.State(value=None)
    deps_state = gr.State(value=None)
    user_msg_state = gr.State(value="")
    # Persistent user ID stored in browser localStorage
    user_id_state = gr.BrowserState(default_value="", storage_key="ems_user_id")

    with gr.Row():
        with gr.Column(scale=2):
            with gr.Walkthrough(selected=1, elem_id="chat_walkthrough") as walkthrough:
                scenario_choices = list_md_files()
                initial_scenario = scenario_choices[0] if scenario_choices else None

                with gr.Step("Scenario Selection", id=1):
                    md_picker = gr.Dropdown(
                        choices=scenario_choices,
                        value=initial_scenario,
                        label="Scenario selection",
                        interactive=args.picker_interactive,
                        allow_custom_value=False,
                        filterable=False,
                    )
                    gr.Markdown(
                        """
                        ### ⚠️ Before You Begin

                        #### Please read the scenario description on the right carefully.

                        Once you understand the scenario, click the button below to start the emergency call simulation.
                        """
                    )
                    next_step_1 = gr.Button(
                        "Start Emergency Call",
                        variant="secondary",
                        size="lg",
                    )

                with gr.Step("Emergency Call", id=2):
                    chatbot = gr.Chatbot(
                        height="60vh",
                        label="112",
                        group_consecutive_messages=False,
                    )
                    with gr.Row():
                        input_box = gr.Textbox(
                            placeholder="Type your message and press Enter…",
                            show_label=False,
                            scale=4,
                            # container=False,
                            max_lines=5,
                            interactive=False,
                        )
                        send = gr.Button(
                            "Send", variant="primary", scale=1, interactive=False
                        )
                    reset = gr.Button("Reset session", variant="secondary")
                    next_step_2 = gr.Button(
                        "Continue to Survey", variant="primary", visible=False
                    )

                with gr.Step("Survey", id=3):
                    gr.Markdown(
                        """
                        ### Please rate your experience

                        Your feedback helps us improve the emergency call system.
                        All questions are required.
                        """
                    )
                    # Dynamically create Radio components from survey config
                    survey_radios: list[gr.Radio] = []
                    for q in DEFAULT_SURVEY.questions:
                        radio = gr.Radio(
                            choices=DEFAULT_SURVEY.get_choices(),
                            label=q.text,
                            type="value",
                            interactive=True,
                        )
                        survey_radios.append(radio)

                    survey_feedback = gr.Textbox(
                        label="Additional feedback (optional)",
                        placeholder="Share any additional comments about your experience...",
                        lines=3,
                        max_lines=6,
                        interactive=True,
                    )

                    survey_submit = gr.Button(
                        "Submit Survey", variant="primary", interactive=False
                    )
                    survey_thanks = gr.Markdown(
                        "### ✓ Thank you for your feedback!",
                        visible=False,
                    )
                    next_step_3 = gr.Button(
                        "Start New Session →", variant="primary", visible=False
                    )

        with gr.Column(scale=2):
            md_view = gr.Markdown(
                construct_scenario_desc(initial_scenario),
            )

    # Debug info visible across all steps (outside walkthrough)
    session_info_display = gr.Markdown(
        format_session_info(),
        visible=args.is_debug_enabled(),
        elem_id="session_info_display",
    )

    def initialize_user_id(stored_user_id: str, scenario_name: str | None):
        """Initialize user_id on page load, creating new one if needed."""
        from uuid import uuid4

        if stored_user_id:
            try:
                user_id = UUID(stored_user_id)
            except ValueError:
                logger.warning(
                    f"Invalid stored user_id: {stored_user_id}, generating new"
                )
                user_id = uuid4()
        else:
            user_id = uuid4()

        # Update debug display with user_id and current scenario
        debug_info = gr.update(
            value=format_session_info(
                user_id=str(user_id),
                scenario=scenario_name,
            ),
            visible=args.is_debug_enabled(),
        )
        return str(user_id), debug_info

    # Initialize user_id on page load
    demo.load(
        initialize_user_id,
        inputs=[user_id_state, md_picker],
        outputs=[user_id_state, session_info_display],
    )

    async def handle_confirm(
        scenario_name: str | None,
        stored_user_id: str,
        old_policy,
        old_deps: Settings | None,
    ):
        """Initialize session and prepare to switch to chat step."""
        # Clean up any existing session
        await cleanup_session(old_policy, old_deps)

        # Handle random scenario selection
        selected_scenario = scenario_name
        if args.random_scenario:
            selected_scenario = get_random_scenario()
            (old_deps.logger if old_deps else logger).info(
                f"Random scenario selection: chose '{selected_scenario}'"
            )

        # Parse stored user_id
        user_id: UUID | None = None
        if stored_user_id:
            try:
                user_id = UUID(stored_user_id)
            except ValueError:
                logger.warning(
                    f"Invalid stored user_id: {stored_user_id}, generating new one"
                )

        try:
            policy, deps = await init_session(
                scenario_name=selected_scenario,
                user_id=user_id,
            )

            session_info = gr.update(
                value=format_session_info(
                    user_id=str(deps.user_id),
                    session_id=str(deps.session_id),
                    scenario=selected_scenario,
                    experiment_name=deps.experiment_name,
                    policy=deps.policy_name,
                ),
                visible=args.is_debug_enabled(),
            )

            # Return state updates (walkthrough navigation is handled separately)
            return (
                policy,  # policy_state
                deps,  # deps_state
                str(deps.user_id),  # user_id_state
                [],  # chatbot (clear history)
                session_info,  # session_info_display
            )

        except Exception as e:
            import traceback

            error_msg = str(e)
            logger.error(
                f"Failed to initialize session: {error_msg}\n{traceback.format_exc()}"
            )
            gr.Error(
                f"❌ **Fatal Error**\n\n{error_msg}\n\nPlease refresh the page.",
                duration=None,
            )
            raise

    confirm_event = (
        next_step_1.click(
            handle_confirm,
            inputs=[md_picker, user_id_state, policy_state, deps_state],
            outputs=[
                policy_state,
                deps_state,
                user_id_state,
                chatbot,
                session_info_display,
            ],
        )
        .then(
            lambda: (*set_input_interactive(False), *set_controls_interactive(False)),
            outputs=[input_box, send, reset, md_picker],
        )
        .then(
            lambda: gr.Walkthrough(selected=2),
            outputs=walkthrough,
        )
    )

    async def trigger_greeting(history: list, policy, deps: Settings):
        """Stream initial greeting from policy."""
        async for updated in stream_policy_messages(
            history=history,
            policy=policy,
            deps=deps,
            user_msg=None,
        ):
            yield updated

    # Chain greeting after confirmation
    greeting_event = confirm_event.then(
        trigger_greeting,
        inputs=[chatbot, policy_state, deps_state],
        outputs=[chatbot],
    )

    greeting_event.success(
        lambda: (*set_input_interactive(True), *set_controls_interactive(True)),
        outputs=[input_box, send, reset, md_picker],
    )

    greeting_event.failure(
        lambda: (*set_input_interactive(False), *set_controls_interactive(True)),
        outputs=[input_box, send, reset, md_picker],
    )

    async def handle_reset(
        old_policy, old_deps: Settings | None, user_id: str, scenario_name: str | None
    ):
        """Reset session and prepare to return to confirmation step."""
        if old_deps is None and old_policy is None:
            # No active session at the time this event was triggered -> do nothing.
            # This prevents stale picker events from wiping a newly started session.
            return (
                gr.skip(),  # policy_state
                gr.skip(),  # deps_state
                gr.skip(),  # chatbot
                # ... gr.skip() for every survey/reset output you currently return ...
                gr.skip(),  # session_info_display (already updated elsewhere)
            )
        await cleanup_session(old_policy, old_deps)
        # Return state updates (walkthrough navigation is handled separately)
        # Returns: policy_state, deps_state, chatbot, then one update per survey radio,
        # then survey_feedback, survey_submit, survey_thanks, next_step_3, session_info_display
        num_radios = len(DEFAULT_SURVEY.questions)
        session_info = gr.update(
            value=format_session_info(
                user_id=user_id,
                scenario=scenario_name,
            ),
            visible=args.is_debug_enabled(),
        )
        return (
            None,  # policy_state
            None,  # deps_state
            [],  # chatbot (clear history)
            *[
                gr.update(value=None, interactive=True) for _ in range(num_radios)
            ],  # reset all radios
            gr.update(value="", interactive=True),  # survey_feedback
            gr.update(interactive=False),  # survey_submit
            gr.update(visible=False),  # survey_thanks
            gr.update(visible=False),  # next_step_3
            session_info,  # session_info_display
        )

    gr.on(
        triggers=[reset.click],  # , md_picker.input
        fn=handle_reset,
        inputs=[policy_state, deps_state, user_id_state, md_picker],
        outputs=[
            policy_state,
            deps_state,
            chatbot,
            *survey_radios,
            survey_feedback,
            survey_submit,
            survey_thanks,
            next_step_3,
            session_info_display,
        ],
        queue=False,
    ).then(
        lambda: (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.Walkthrough(selected=1),
        ),
        outputs=[reset, next_step_2, walkthrough],
    )

    # Update scenario description and debug info when picker changes
    def update_scenario_view(scenario_name: str | None, user_id: str):
        """Update scenario description and debug info with selected scenario."""
        scenario_desc = construct_scenario_desc(scenario_name)
        session_info = gr.update(
            value=format_session_info(
                user_id=user_id,
                scenario=scenario_name,
            ),
            visible=args.is_debug_enabled(),
        )
        return scenario_desc, session_info

    md_picker.input(
        update_scenario_view,
        inputs=[md_picker, user_id_state],
        outputs=[md_view, session_info_display],
    )

    # only re-enable send button if input is non-empty, wont trigger when dialogue is complete since input is disabled
    input_box.input(
        lambda text: gr.update(interactive=bool(text and text.strip())),
        inputs=[input_box],
        outputs=[send],
    )

    def user_submit(user_message: str, history: list):
        """Append user message and disable inputs."""
        if not user_message or not user_message.strip():
            # unchanged history and empty message for invalid input
            # should only trigger on inputbox_submit (Enter pressed), send button is disabled
            return None, history, gr.skip(), gr.skip(), gr.skip(), gr.skip()
        history.append(ChatMessage(role="user", content=user_message))
        return (
            user_message,
            history,
            *set_input_interactive(False),
            *set_controls_interactive(False),
        )

    submit_event = (
        gr.on(
            triggers=[input_box.submit, send.click],
            fn=lambda: None,  # this seems necessary for te send button to disable properly
        )
        .then(
            user_submit,
            inputs=[input_box, chatbot],
            outputs=[user_msg_state, chatbot, input_box, send, reset, md_picker],
        )
        .then(
            bot_respond,
            inputs=[chatbot, policy_state, deps_state, user_msg_state],
            outputs=[chatbot],
        )
    )

    def on_response_complete(history: list[ChatMessageDict]):
        """Enable inputs after bot responds, unless conversation completed."""
        is_complete = bool(
            history
            and history[-1]
            and history[-1].get("metadata")
            and (history[-1]["metadata"].get("id") == "completion_message")
        )
        return (
            *set_input_interactive(not is_complete),
            gr.update(
                visible=not is_complete, interactive=not is_complete
            ),  # Hide reset when complete
            gr.update(
                interactive=args.picker_interactive if not is_complete else False
            ),  # md_picker
            gr.update(visible=is_complete),  # Show next_step_2 when complete
        )

    submit_event.then(
        on_response_complete,
        inputs=[chatbot],
        outputs=[input_box, send, reset, md_picker, next_step_2],
    ).failure(
        lambda: (
            *set_input_interactive(False),
            *set_controls_interactive(True),
        ),
        outputs=[input_box, send, reset, md_picker],
    )

    # ==========================================================================
    # Survey Event Handlers
    # ==========================================================================

    def check_all_answered(*values):
        """Enable submit button only when all survey questions are answered."""
        all_answered = all(value is not None for value in values)
        return gr.update(interactive=all_answered)

    gr.on(
        triggers=[radio.change for radio in survey_radios],
        fn=check_all_answered,
        inputs=survey_radios,
        outputs=survey_submit,
        queue=False,  # dont compete with long chatbot events
        trigger_mode="always_last",  # last interaction wins
        concurrency_limit=1,  # avoid overlapping updates to same button
        concurrency_id="survey_validation",
    )

    def handle_survey_submit(deps: Settings | None, feedback: str, *responses):
        """Save survey responses and show thank-you message."""
        if deps is None:
            logger.warning("Survey submitted but no session active")
            return (
                *[gr.update() for _ in responses],
                gr.update(),  # survey_feedback
                gr.update(),  # survey_submit
                gr.update(
                    visible=True,
                    value="### ⚠️ Session expired. Please start a new session.",
                ),
                gr.update(visible=True),  # Show next_step_3 button
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

        # Save with metadata and optional feedback
        metadata: dict[str, str] = {
            "user_id": str(deps.user_id),
            "session_id": str(deps.session_id),
            "scenario": str(deps.scenario_name),
            "policy": str(deps.policy_name),
        }
        if feedback and feedback.strip():
            metadata["feedback"] = feedback.strip()

        save_survey_responses(deps.save_path, response, metadata)
        deps.logger.info(f"Survey saved for session {deps.session_id}")

        # Disable radios, feedback, and submit, show thanks and next step button
        return (
            *[gr.update(interactive=False) for _ in responses],
            gr.update(interactive=False),  # survey_feedback
            gr.update(interactive=False),  # survey_submit
            gr.update(visible=True),  # survey_thanks
            gr.update(visible=True),  # Show next_step_3 button
        )

    survey_submit.click(
        handle_survey_submit,
        inputs=[deps_state, survey_feedback, *survey_radios],
        outputs=[
            *survey_radios,
            survey_feedback,
            survey_submit,
            survey_thanks,
            next_step_3,
        ],
    )

    # Step 2: Next step button navigates to Step 3
    next_step_2.click(
        lambda: gr.Walkthrough(selected=3),
        outputs=walkthrough,
    )

    # Step 3: Next step button resets session and navigates to Step 1
    next_step_3.click(
        handle_reset,
        inputs=[policy_state, deps_state, user_id_state, md_picker],
        outputs=[
            policy_state,
            deps_state,
            chatbot,
            *survey_radios,
            survey_feedback,
            survey_submit,
            survey_thanks,
            next_step_3,
            session_info_display,
        ],
    ).then(
        lambda: (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(interactive=args.picker_interactive),  # Re-enable md_picker
            gr.Walkthrough(selected=1),
        ),
        outputs=[reset, next_step_2, md_picker, walkthrough],
    )


def notify_share_url(share_url: str) -> None:
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


if __name__ == "__main__":
    # TODO: use load / unload events to manage session lifecycle, can reset trigger the same logic? [maybe we can deal with the server not being reachable anymore? don't throw any errors in that case]

    # Use the queue for scalability
    app, local_url, share_url = demo.queue(default_concurrency_limit=16).launch(
        pwa=True,
        share=not args.debug,
        css="""
            .icon-button-wrapper.top-panel { display: none !important; }
            #chat_walkthrough > div:first-child { display: none !important; }

            /* Session info: move to bottom-right, transparent background,
               and allow clicks to pass through to underlying controls. */
            #session_info_display {
                position: fixed !important;
                bottom: 8px !important;
                right: 8px !important;
                left: auto !important;
                background: transparent !important;
                padding: 6px 10px !important;
                border: none !important;
                z-index: 1000 !important;
                pointer-events: none; /* allow clicks to go through */
                color: var(--text-color-primary) !important;
                opacity: 0.95 !important;
                text-align: right !important;
                max-width: 40vw;
            }

            /* If you want text selectable but still clicks to pass, allow
               pointer events only for text selection by enabling user-select. */
            #session_info_display * {
                user-select: text;
            }
        """,
        footer_links=["settings"],
        debug=args.debug,
        inbrowser=args.debug,
        show_error=True,
        prevent_thread_lock=True,
        # favicon_path:
    )

    logger.info("Gradio local URL: %s", local_url)
    logger.info("Gradio share URL: %s", share_url)
    if share_url:
        notify_share_url(share_url)
    else:
        logger.warning("No share URL returned from Gradio")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
