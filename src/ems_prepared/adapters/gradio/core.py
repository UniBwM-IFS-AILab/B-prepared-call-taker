"""Core logic for the Gradio emergency call simulator.

This module contains reusable components that can be imported independently:
- Type definitions for chat messages
- Session management (init, cleanup)
- Scenario utilities (reading markdown files)
- Policy runners (invoke_graph, invoke_agent)
"""

import os
import random
from pathlib import Path
from typing import AsyncGenerator, Literal, TypedDict
from uuid import UUID, uuid4

import gradio as gr
from gradio import ChatMessage
from pydantic_graph.graph import Graph
from pydantic_graph.nodes import End

from ems_prepared.adapters.cli import args
from ems_prepared.dialogue_state.emergency_call_state import EmergencyCall
from ems_prepared.dialogue_state.meta_state import GraphState
from ems_prepared.policies.llm_only.agent import (
    AgentPolicy,
    build_emergency_agent,
    extract_last_model_response,
    run_agent_with_capture,
    save_agent_run_results,
    update_history_and_merge_state,
)
from ems_prepared.policies.pydantic_graph.custom_persistence.resumable_file_persistence import (
    clear_old_run as clear_debug_user_run,
)
from ems_prepared.policies.pydantic_graph.emergency_main_graph import (
    build_graph,
    run_graph,
)
from ems_prepared.policies.pydantic_graph.nodes import MessageNode, QuestionNode
from ems_prepared.util.logger import flush_logger
from ems_prepared.util.settings import InputMode, Settings

try:
    from typing import NotRequired
except ImportError:
    from typing_extensions import NotRequired


# =============================================================================
# Type Definitions
# =============================================================================


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


# =============================================================================
# Constants
# =============================================================================

# Completion message shown when conversation ends
COMPLETION_MESSAGE = ChatMessage(
    content="The emergency call has been processed. Thank you.",
    metadata={"id": "completion_message"},
)


# =============================================================================
# Session Management
# =============================================================================


async def cleanup_session(policy, deps: Settings | None) -> None:
    """Save and flush session data before reset.

    Args:
        policy: Current policy (AgentPolicy or Graph).
        deps: Current settings/dependencies.
    """
    if not deps:
        return

    try:
        flush_logger(deps.messages_logger)
        flush_logger(deps.state_logger)
    except Exception as e:
        deps.logger.warning(f"Could not save/flush session: {e}")
        gr.Warning(f"Session cleanup issue: {str(e)}", duration=3)


async def init_session(
    scenario_name: str | None = None,
    user_id: UUID | None = None,
) -> tuple[AgentPolicy | Graph[GraphState, Settings, EmergencyCall], Settings]:
    """Initialize a new session with graph or agent based on policy.

    Args:
        scenario_name: Name of the selected scenario file for logging context.
        user_id: User ID from BrowserState (persisted across page reloads).
            If None, generates a new random UUID.

    Returns:
        Tuple of (policy, deps) where policy is AgentPolicy or Graph.
    """
    policy_setting = args.resolve_policy()
    # Priority: CLI override > BrowserState > new random UUID
    if args.user_id:
        user_id = UUID(int=args.user_id)
    elif user_id is None:
        user_id = uuid4()

    experiment_name = args.experiment_name or ""
    deps = Settings(
        name=f"gradio_{policy_setting}",
        call_origin=InputMode.API,
        user_id=user_id,
        scenario_name=scenario_name,
        policy_name=policy_setting,
        experiment_name=experiment_name,
    )

    # Clear debug user's old run data (only affects user_id=0)
    await clear_debug_user_run(deps.user_id, experiment_name=deps.experiment_name)

    policy: AgentPolicy | Graph[GraphState, Settings, EmergencyCall]
    if policy_setting == "agent":
        agent = build_emergency_agent(deps)
        state = EmergencyCall()
        policy = AgentPolicy(agent=agent, state=state, history=[], deps=deps)
    else:
        policy = await build_graph()

    gr.Info(f"Session initialized: {str(deps.session_id)}", duration=3)
    deps.logger.warning(
        f"Session initialized: {str(deps.session_id)}, {str(deps.user_id)}, {policy_setting}"
    )
    return policy, deps


# =============================================================================
# Scenario Utilities
# =============================================================================


def get_scenario_directory() -> Path:
    """Get the docs directory from environment variable or command line argument.

    Priority:
    1. Command line argument --scenario-dir
    2. Environment variable SCENARIO_DIR
    3. Default to ./docs relative to current working directory
    """
    if args.scenario_dir:
        scenario_dir = Path(args.scenario_dir).resolve()
    # Fall back to environment variable
    elif env_scenario_dir := os.getenv("SCENARIO_DIR"):
        scenario_dir = Path(env_scenario_dir).resolve()
    # Final fallback to default
    else:
        scenario_dir = Path.cwd() / "docs" / "scenarios"

    scenario_dir.mkdir(parents=True, exist_ok=True)
    return scenario_dir


def list_md_files() -> list[str]:
    """Return available Markdown filenames (base names)."""
    scenario_dir = get_scenario_directory()
    return sorted(
        p.name for p in scenario_dir.glob("*.md") if not p.name.startswith("_")
    )


def get_random_scenario() -> str | None:
    """Get a random scenario filename from available scenarios."""
    scenarios = list_md_files()
    return random.choice(scenarios) if scenarios else None


def read_md(filename: str | None) -> str:
    """Read a markdown file from the scenario directory.

    Args:
        filename: Name of the file to read (just the filename, not full path).

    Returns:
        Content of the file, or an error message if not found.
    """
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


# =============================================================================
# Policy Runners
# =============================================================================


async def invoke_graph(graph, deps: Settings, msg: str | None):
    """Generator that yields messages from the graph.

    Repeatedly calls run_graph, yielding any messages returned,
    until a question or End is encountered.

    Args:
        graph: The pydantic graph to run.
        deps: Settings/dependencies.
        msg: User message to process (None for initial greeting).

    Yields:
        str: Messages to display to user.
        End: When conversation is complete.
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
            yield result
            break
        elif isinstance(result, QuestionNode):
            # Got a question node - yield its question and stop
            yield result.question
            break
        elif isinstance(result, MessageNode):
            # Got a node with messages - yield the message and continue
            message = result.messages.get(deps.locale, "No message for this locale.")
            yield message
            # Continue the loop to run graph again
        else:
            # Unexpected result type - log and break
            deps.logger.warning(
                f"Unexpected result type from run_graph: {type(result).__name__}"
            )
            break


async def invoke_agent(
    policy: AgentPolicy,
    deps: Settings,
    msg: str | None,
) -> AsyncGenerator[End[EmergencyCall] | str, None]:
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

    # Run the agent and capture any provider messages
    result, captured_messages = await run_agent_with_capture(
        agent,
        agent_history,
        msg,
    )
    deps.logger.info(f"agent_history_len={len(agent_history)} usage={result.usage()}")

    # Log provider/model information when available
    response = extract_last_model_response(result, captured_messages)
    if response:
        deps.logger.info(
            f"provider={response.provider_name} model={response.model_name}"
        )
    else:
        deps.logger.warning("No ModelResponse found for this run.")

    # Update history, merge state, and get next action
    is_complete, next_question = update_history_and_merge_state(
        agent_history, result, state, deps, caller_msg=msg
    )

    if is_complete:
        deps.logger.info("Outcome reached")
        try:
            save_agent_run_results(state, agent_history, deps)
            flush_logger(deps.messages_logger)
            flush_logger(deps.state_logger)
        except Exception as e:
            deps.logger.warning(f"Could not save/flush session on completion: {e}")
        yield End(data=state)
        return

    # Yield the operator's question (or an error if missing)
    if not next_question or not str(next_question).strip():
        deps.logger.error(
            "Agent returned neither completion nor next_question. "
            f"state={result.output.state!r}, next_question={next_question!r}, "
            f"len(agent_history)={len(agent_history)}"
        )
        deps.logger.error("Full agent result: %r", result)
        yield "❌ Internal error: agent returned no next question. Please reset the session."
        return

    deps.logger.debug(next_question)
    yield next_question
