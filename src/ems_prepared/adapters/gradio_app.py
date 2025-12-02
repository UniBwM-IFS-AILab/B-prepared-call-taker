import argparse
import asyncio
import os
from pathlib import Path
from typing import Any, AsyncGenerator, Literal, TypedDict
from uuid import UUID

from pydantic_graph.graph import Graph

from ems_prepared.dialogue_state.meta_state import GraphState

try:
    from typing import NotRequired
except ImportError:
    from typing_extensions import NotRequired

import gradio as gr
from gradio import ChatMessage
from loguru import logger
from pydantic_graph.nodes import End

from ems_prepared.dialogue_state.emergency_call_state import EmergencyCall
from ems_prepared.policies.llm_only.agent import (
    AgentPolicy,
    build_emergency_agent,
    check_completion,
    process_state_update,
    save_results,
)
from ems_prepared.policies.pydantic_graph.custom_persistence.resumable_file_persistence import (
    clear_old_run,
)
from ems_prepared.policies.pydantic_graph.emergency_main_graph import (
    build_graph,
    run_graph,
)
from ems_prepared.util.logger import flush_logger
from ems_prepared.util.settings import InputMode, Settings


# Type definitions for ChatMessage
class ChatMessageMetadata(TypedDict, total=False):
    """Metadata for ChatMessage."""

    title: str
    id: str | int
    parent_id: str | int
    log: str
    duration: float
    status: Literal["pending", "done"]


class ChatMessageOption(TypedDict):
    """Options for a ChatMessage."""

    key: str
    value: str


class ChatMessageDict(TypedDict):
    """TypedDict representation of a ChatMessage."""

    role: Literal["user", "assistant", "system"]
    content: str
    metadata: NotRequired[ChatMessageMetadata]
    options: NotRequired[ChatMessageOption]


# Completion message shown when conversation ends
COMPLETION_MESSAGE = ChatMessage(
    # role="assistant",
    content="The emergency call has been processed. Thank you.",
    metadata={"id": "completion_message"},
)


# Unified argument parser - created once at module level
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument(
    "--scenario-dir",
    type=str,
    default=None,
    help="Path to the directory for the scenario descriptions.",
)
_parser.add_argument(
    "--user-id",
    type=int,
    default=0,
    help="Integer user ID (default: 0).",
)
_parser.add_argument(
    "--policy",
    type=str,
    default="graph",
    choices=["graph", "agent"],
    help="Policy to use: 'graph' for pydantic_graph (default) or 'agent' for LLM-only agent.",
)
_parser.add_argument(
    "--debug",
    action="store_true",
    help="Show additional session debug information in the UI.",
)
_args, _ = _parser.parse_known_args()


def debug_enabled() -> bool:
    """Determine whether debug visuals should be shown."""
    env_debug = os.getenv("GRADIO_DEBUG", "") == "1"
    return _args.debug or env_debug


def get_scenario_directory() -> Path:
    """Get the docs directory from environment variable or command line argument.

    Priority:
    1. Command line argument --scenario-dir
    2. Environment variable SCENARIO_DIR
    3. Default to ./docs relative to current working directory
    """
    if _args.scenario_dir:
        scenario_dir = Path(_args.scenario_dir).resolve()
    # Fall back to environment variable
    elif env_scenario_dir := os.getenv("SCENARIO_DIR"):
        scenario_dir = Path(env_scenario_dir).resolve()
    # Final fallback to default
    else:
        scenario_dir = Path.cwd() / "docs" / "scenarios"

    scenario_dir.mkdir(parents=True, exist_ok=True)
    return scenario_dir


async def init_session():
    """Initialize a new session with graph or agent based on policy.

    Args:
        policy: 'graph' or 'agent'. If None, reads from command line args.

    Returns:
        For graph policy: (graph, deps)
        For agent policy: (agents, deps)
    """
    policy_setting = _args.policy

    # Placeholder emit - will be replaced per conversation
    async def placeholder_emit(msg: str):
        pass

    deps = Settings(
        name=f"gradio_{policy_setting}",
        call_origin=InputMode.API,
        emit=placeholder_emit,
        user_id=UUID(int=_args.user_id),
    )

    await clear_old_run(deps.user_id)

    policy: AgentPolicy | Graph[GraphState, Settings, EmergencyCall]
    if policy_setting == "agent":
        # Agent policy: create agent and state, no initial run yet
        agent = build_emergency_agent()
        state = EmergencyCall()
        policy = AgentPolicy(agent=agent, state=state, history=[])
    else:
        # Graph policy: build graph
        policy = await build_graph()

    gr.Info(
        f"Session initialized: {str(deps.session_id)}",
        duration=3,
    )
    return policy, deps


def list_md_files() -> list[str]:
    """Return available Markdown filenames (base names)."""
    scenario_dir = get_scenario_directory()
    return sorted(
        p.name for p in scenario_dir.glob("*.md") if not p.name.startswith("_")
    )


def read_md(filename: str | None) -> str:
    if not filename:
        return "### No file selected"
    scenario_dir = get_scenario_directory()
    path = scenario_dir / filename
    if not path.exists():
        return f"### File not found: `{filename}`"
    return path.read_text(encoding="utf-8")


def construct_scenario_desc(filename: str | None) -> str:
    """Merge the shared instructions with the selected scenario content."""
    scenario_dir = get_scenario_directory()
    instructions_path = scenario_dir / "_Instructions.md"
    instructions = read_md("_Instructions.md") if instructions_path.exists() else ""

    scenario_content = read_md(filename)
    return f"{instructions}\n\n{scenario_content}" if instructions else scenario_content


async def invoke_graph(graph, deps, msg: str | None):
    """Generator that yields messages from the graph.

    Repeatedly calls run_graph, yielding any messages returned,
    until a question or End is encountered.
    """
    if graph is None:
        raise ValueError("Graph is not initialized.")

    # Keep running the graph until we get a question or End
    while result := await run_graph(graph, deps, msg):
        # After first call, subsequent calls don't need the msg
        msg = None

        # Check what we got back
        if isinstance(result, End):
            # Graph finished
            # yield COMPLETION_MESSAGE  # This makes detection of completion harder (string matching)
            yield result
            break
        elif hasattr(result, "question") and isinstance(result.question, str):
            # Got a question - yield it and stop
            yield result.question
            break
        elif hasattr(result, "messages") and isinstance(result.messages, dict):
            # Got a node with messages - yield the message and continue
            message = result.messages.get(deps.locale, "No message for this locale.")
            yield message
            # Continue the loop to run graph again
        else:
            # Unexpected result type - log and break
            logger.warning(
                f"Unexpected result type from run_graph: {type(result).__name__}"
            )
            break


async def invoke_agent(
    policy: AgentPolicy,
    deps: Settings,
    msg: str | None,
) -> AsyncGenerator[End[EmergencyCall] | str]:
    """Generator that yields messages from the agent loop.

    Processes user messages and yields operator questions,
    until a final outcome is reached.

    Args:
        policy: AgentPolicy with agent, state, and history (modified in place)
        deps: Settings/dependencies
        msg: User message to process (None for initial greeting)

    Yields:
        str: Messages to display to user (questions from operator)
        End: When conversation is complete

    Note:
        The state and history are modified in place within the policy object.
    """
    agent, state, agent_history = policy.agent, policy.state, policy.history
    if agent is None:
        raise ValueError("Agent is not initialized.")

    # Run agent with or without user message
    if msg is None:
        result = await agent.run(message_history=agent_history)
    else:
        deps.messages_logger.info("", extra={"speaker": "caller", "msg_text": msg})
        result = await agent.run(user_prompt=msg, message_history=agent_history)

    # Update agent_history in place by extending with new messages
    agent_history.extend(result.new_messages())

    # Process any remaining state update (when both state and question are present)
    if result.output.state is not None:
        process_state_update(state, result.output.state, deps)

        # Check for completion after processing final state
        if check_completion(state):
            logger.info("Outcome reached")
            yield End(data=state)
            return

    # At this point, result.output.next_question should have the operator's question
    if result.output.next_question is not None:
        deps.messages_logger.info(
            "", extra={"speaker": "operator", "msg_text": result.output.next_question}
        )
        logger.debug(result.output.next_question)
        yield result.output.next_question


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
    policy,
    deps,
    scenario_name: str | None = None,
    user_msg: str | None = None,
):
    """Stream messages from the policy generator into history.

    Args:
        history: Current chat history
        policy: The policy to run
        deps: Settings/dependencies
        scenario_name: Selected scenario filename
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

        async for item in async_generator:
            # Check if this is the final result
            if isinstance(item, End):
                deps.state_logger.info(
                    {
                        "session": {
                            "policy": _args.policy,
                            "scenario": scenario_name or "unspecified",
                        }
                    },
                    extra={"event": "metadata"},
                )
                flush_logger(deps.state_logger)

                # Stream completion message
                async for updated_history in stream_message_to_history(
                    history, COMPLETION_MESSAGE
                ):
                    yield updated_history
                break
            elif isinstance(item, str):
                # Stream the message (either emitted message or question)
                async for updated_history in stream_message_to_history(
                    history, ChatMessage(role="assistant", content=item)
                ):
                    yield updated_history
    except Exception as e:
        import traceback

        error_msg = str(e)
        logger.error(f"Error streaming messages: {error_msg}\n{traceback.format_exc()}")
        history.append(
            ChatMessage(
                role="assistant",
                content=f"❌ **Fatal Error**\n\n{error_msg}\n\nPlease click 'Reset session' to recover.",
            )
        )
        yield history
        raise gr.Error(error_msg, duration=None)


async def bot_respond(
    history: list[ChatMessageDict | ChatMessage],
    policy,
    deps,
    user_msg: str,
    scenario_name: str | None = None,
):
    """Stream responses from policy as messages are emitted.

    Args:
        history: Chat history
        policy: The policy to run
        deps: Settings/dependencies
        user_msg: The user's message to process
        scenario_name: Selected scenario filename
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
        scenario_name=scenario_name,
        user_msg=user_msg,
    ):
        yield updated_history


def user_submit(
    user_message: str, history: list[ChatMessageDict | ChatMessage] = []
) -> tuple[Any, Any, list[ChatMessageDict | ChatMessage], str]:
    """Append the user message and disable inputs while waiting for response.

    If message is empty or only whitespace, returns current state unchanged.
    """
    # don't append empty messages to history
    if not user_message or not user_message.strip():
        return gr.update(), gr.update(), history, ""

    history.append(ChatMessage(role="user", content=user_message))
    # Returns: input_box, send, chatbot, user_message
    input_box, send = disable_input()
    return (input_box, send, history, user_message)


def enable_input():
    """Enable textbox and send button."""
    return (
        gr.update(interactive=True, autofocus=True),  # textbox enabled
        gr.update(interactive=True),  # send button enabled
    )


def disable_input():
    """Disable textbox and send button."""
    return (
        gr.update(interactive=False, value=""),  # textbox disabled and cleared
        gr.update(interactive=False),  # send button disabled
    )


def handle_conversation_end(history: list[ChatMessageDict]):
    """Check if conversation ended and return appropriate input states.

    Returns disable_input() for both ended and continuing cases.
    The difference is handled by reset button state in the event chain.
    """
    if (
        history
        and history[-1]["metadata"]
        and history[-1]["metadata"].get("id") == "completion_message"
    ):
        return disable_input()
    else:
        return enable_input()


async def init_or_reset_session(old_deps=None, old_policy=None, scenario_name=None):
    """Initialize or reset the session (unified handler for load and reset).

    Args:
        old_deps: Current dependencies (None on initial load, existing deps on reset)
        old_policy: Current policy (None on initial load, tuple/graph on reset)
        scenario_name: Selected scenario filename from the dropdown

    Returns session state, markdown files, and empty chatbot history.
    Greeting messages will be streamed separately via .then() chaining.
    """
    # Save and clean up old session if resetting
    try:
        if old_deps and isinstance(old_policy, AgentPolicy):
            save_results(old_policy.state, old_policy.history, old_deps)
            flush_logger(old_deps.messages_logger)
            flush_logger(old_deps.state_logger)
    except Exception as e:
        logger.warning(f"Could not save/flush old session: {e}")
        gr.Warning(f"Session cleanup issue: {str(e)}", duration=3)

    try:
        policy, deps = await init_session()

        # Format session info (visible only in debug mode)
        session_info = gr.update(
            value=(
                f"**User ID:** `{deps.user_id}`  \n"
                f"**Session ID:** `{deps.session_id}`  \n"
                f"**Policy:** `{_args.policy}`  \n"
                f"**Scenario:** `{scenario_name or 'None selected'}`"
            ),
            visible=debug_enabled(),
        )

        # Return empty history - messages will be streamed separately
        return (
            [],  # Empty chatbot history
            policy,
            deps,
            session_info,
            *disable_input(),  # Disable during initial streaming
        )

    except Exception as e:
        import traceback

        error_msg = str(e)
        full_error = (
            f"Failed to initialize session: {error_msg}\n{traceback.format_exc()}"
        )
        logger.error(full_error)

        # Raise error - this will show the modal to the user
        raise gr.Error(
            f"❌ **Fatal Error**\n\n{error_msg}\n\nPlease try clicking 'Reset session' or refresh the page.",
            duration=None,
        )


def trigger_greeting(
    trigger_event,
    chatbot,
    policy_state,
    deps_state,
    input_box,
    send,
    reset,
    scenario_component,
):
    """Setup the chain of events for initialization (load or reset).

    Args:
        trigger_event: The event that triggers initialization (demo.load or reset.click)
        chatbot, policy_state, deps_state: Components whose values are used for streaming
        input_box, send: Input components to enable/disable
        reset: Reset button to enable on success/failure

    Returns:
        The final event in the chain (for potential further chaining)
    """
    # Stream initial greeting messages - explicitly specify inputs to use updated state values
    greeting_event = trigger_event.then(
        lambda: gr.update(interactive=False),
        inputs=None,
        outputs=[reset],
        queue=False,
    ).then(
        stream_policy_messages,
        inputs=[chatbot, policy_state, deps_state, scenario_component],
        outputs=[chatbot],
    )
    # Enable inputs after streaming completes
    greeting_event.success(
        lambda: (*enable_input(), gr.update(interactive=True)),
        inputs=None,
        outputs=[input_box, send, reset],
        queue=False,
    )
    # On failure, disable input but enable reset for recovery
    greeting_event.failure(
        lambda: (*disable_input(), gr.update(interactive=True)),
        inputs=None,
        outputs=[input_box, send, reset],
        queue=False,
    )
    return greeting_event


# ------------------- UI with Session State -------------------
demo = gr.Blocks(
    title="Chatbot + Markdown Side Panel",
    fill_height=True,
    css=(
        ".icon-button-wrapper.top-panel { display: none !important; } "
        "#scenario-view { font-size: 1.9rem !important; }"
    ),  # hides the clear (trashbin) button in the chatwindow and bumps scenario text size
    theme=gr.themes.Ocean(),
)  # this is separate from with statement to work around bug with `gradio dev` (hot-reloading)
with demo:
    gr.Markdown(
        """
        ## Emergency Call Simulator
        Chat interface on the left, scenario selection on the right.
        """
    )

    # Session state for policy and deps
    policy_state = gr.State(value=None)
    deps_state = gr.State(value=None)
    user_msg_state = gr.State(value="")

    with gr.Row():
        # --- Left column: Chat ---
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                type="messages",
                height="65vh",
                label="112",
                group_consecutive_messages=False,
            )
            with gr.Row():
                input_box = gr.Textbox(
                    placeholder="Type your message and press Enter…",
                    scale=4,
                    container=False,
                    # submit_btn="Send",
                    max_lines=1,
                )
                send = gr.Button("Send", variant="primary", scale=1)
            reset = gr.Button("Reset session", variant="secondary")
            session_info_display = gr.Markdown(
                "**User ID:** _loading..._  \n**Session ID:** _loading..._  \n**Policy:** _loading..._  \n**Scenario:** _loading..._",
                elem_id="session-info",
                visible=debug_enabled(),
            )

        # --- Right column: Markdown reference ---
        with gr.Column(scale=2):
            # gr.Markdown("### Reference")
            initial_choices = list_md_files()
            initial_value = initial_choices[0] if initial_choices else None
            md_picker = gr.Dropdown(
                choices=initial_choices,
                value=initial_value,
                label="Scenario selection",
                interactive=True,
                allow_custom_value=False,
                filterable=False,
            )
            initial_content = construct_scenario_desc(initial_value)
            md_view = gr.Markdown(initial_content, elem_id="scenario-view")

    # Initialize session and UI on load or reset - unified event chain
    init_event = gr.on(
        triggers=[demo.load, reset.click, md_picker.change],
        fn=init_or_reset_session,
        inputs=[deps_state, policy_state, md_picker],
        outputs=[
            chatbot,
            policy_state,
            deps_state,
            session_info_display,
            input_box,
            send,
        ],
        queue=False,
    )
    trigger_greeting(
        init_event,
        chatbot,
        policy_state,
        deps_state,
        input_box,
        send,
        reset,
        md_picker,
    )

    # Load Markdown content when scenario selection changes
    md_picker.change(construct_scenario_desc, inputs=md_picker, outputs=md_view)

    submit_event = gr.on(
        triggers=[input_box.submit, send.click],
        fn=lambda input_box, chatbot: (
            *user_submit(input_box, chatbot),
            gr.update(interactive=False),
        ),
        inputs=[input_box, chatbot],
        outputs=[input_box, send, chatbot, user_msg_state, reset],
        queue=False,
    )

    bot_event = submit_event.then(
        bot_respond,
        inputs=[chatbot, policy_state, deps_state, user_msg_state, md_picker],
        outputs=[chatbot],
    )
    bot_event.success(
        lambda chatbot: (
            *handle_conversation_end(chatbot),
            gr.update(interactive=True),
        ),
        inputs=[chatbot],
        outputs=[input_box, send, reset],
        queue=False,
    )
    # On failure, disable input but enable reset for recovery
    bot_event.failure(
        lambda: (*disable_input(), gr.update(interactive=True)),
        inputs=None,
        outputs=[input_box, send, reset],
        queue=False,
    )

# Use the queue for scalability
demo.queue(default_concurrency_limit=16).launch(pwa=True)
