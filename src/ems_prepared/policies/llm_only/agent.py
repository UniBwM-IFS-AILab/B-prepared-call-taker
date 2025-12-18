"""Agent representing an LLM-Driven loop for variable extraction and outcome determination."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from pydantic_ai._agent_graph import capture_run_messages
from pydantic_ai.agent import Agent, AgentRunResult
from pydantic_ai.messages import ModelMessage, ModelResponse
from rich import print

from ems_prepared.agents.history_processors import remove_before_extracion_processor
from ems_prepared.agents.reusable_prompts import (
    BASE_SYSTEM_PROMPT,
    extend_system_prompt,
)
from ems_prepared.agents.system_prompt import system_prompt
from ems_prepared.dialogue_state.emergency_call_state import EmergencyCall
from ems_prepared.dialogue_state.structured_output import DialogueOutput
from ems_prepared.policies.pydantic_graph.utils import save_state_json
from ems_prepared.util.custom_deepmerge import ignore_empty_merger
from ems_prepared.util.logger import flush_logger
from ems_prepared.util.models import (
    build_fallback_agent,
)

# removed unused imports
from ems_prepared.util.save_utils import save_message_history_json
from ems_prepared.util.settings import Settings
from ems_prepared.util.user_interaction import prompt_user

logger = logging.getLogger(__name__)


@dataclass
class AgentPolicy:
    """Container for agent policy state."""

    agent: Agent[Settings, DialogueOutput]
    state: EmergencyCall
    history: list[ModelMessage]
    deps: Settings


def build_emergency_agent(deps: Settings) -> Agent[Settings, DialogueOutput]:
    """Build and return the emergency call agent with configured system prompt.

    Returns:
        Configured PydanticAI agent for emergency call handling
    """

    prompt: system_prompt = extend_system_prompt(
        BASE_SYSTEM_PROMPT,
        task=(
            "You receive a phone call from a caller who wants to report an emergency. \n"
            "First, present an appropriate greeting. \n"
            "Then, begin determining the basic information about the emergency. \n"
            "The outcome and dispatch decisions are determined automatically by other systems. \n"
            "This is time-critical, so keep the conversation efficient and to the point."
        ),
        rules=(
            "Only ask questions that help you fill personalia or RD1-related variables; do not ask about anything outside of the schema.\n"
            "You may not ask any questions that target RD2 symptoms directly.\n"
            "When speaking to the caller, never mention the words 'RD1', 'RD2', 'schema', 'JSON', or 'State'; these are internal concepts."
            "only when at least one outcome is true, decide if you have enough information or need to ask further quesitons."
        ),
        decisions=(
            "Use the provided schema to understand which fields belong to personalia and which correspond to RD1 and RD2.\n"
            "Continue asking targeted questions only as needed to determine and fill RD1-related variables.\n"
            "Only AFTER RD1 is true are you allowed to ask for more details in a single generic and open question.\n"
            "Do not mention RD2 or any RD2 symptom names directly in this question.\n"
            "Do not ask any additional new questions specifically targeting RD2 symptoms after this generic question.\n"
        ),
    )

    agent = build_fallback_agent(
        output_type=DialogueOutput,
        system_prompt=prompt.full_prompt,
        # history_processors=[remove_before_extracion_processor],
        instructions=f"Only use the following language: {deps.locale.value}.\n",
    )
    return agent  # type: ignore


async def run_agent_with_capture(
    agent: Agent[Settings, DialogueOutput],
    agent_history: list[ModelMessage],
    user_prompt: str | None,
) -> tuple[AgentRunResult[DialogueOutput], list]:
    """Run the agent inside a capture context and return result + captured messages.

    Args:
        agent: The Agent instance to run.
        agent_history: Mutable message history to pass to the agent.
        user_prompt: User-provided prompt (None for initial greeting).

    Returns:
        Tuple of (AgentRunResult, captured_messages_list)
    """

    logger.debug(f"{len(agent_history)} messages in history")
    # from devtools import debug

    # debug(agent_history[-4:])
    with capture_run_messages() as captured_messages:
        if user_prompt is None:
            result = await agent.run(
                user_prompt="",
                message_history=agent_history,
            )
        else:
            result = await agent.run(
                user_prompt=user_prompt,
                message_history=agent_history,
            )

    return result, list(captured_messages)


def extract_last_model_response(
    result: AgentRunResult[DialogueOutput], captured_messages: list | None = None
) -> ModelResponse | None:
    """Return the last ModelResponse from result.new_messages(), falling back to captured_messages."""
    responses = [m for m in result.new_messages() if isinstance(m, ModelResponse)]
    if responses:
        return responses[-1]
    if captured_messages:
        responses = [m for m in captured_messages if isinstance(m, ModelResponse)]
        if responses:
            return responses[-1]
    return None


def update_history_and_merge_state(
    agent_history: list[ModelMessage],
    result: AgentRunResult[DialogueOutput],
    current_state: EmergencyCall,
    deps: Settings,
    caller_msg: str | None = None,
) -> tuple[bool, str | None]:
    """Extend agent_history, merge result state into current state, log operator/question,
    and return (is_complete, next_question).
    """

    # Extend history with new messages
    agent_history.extend(result.new_messages())

    # Merge any returned state
    new_state = result.output.state
    if new_state is not None:
        merge_state(current_state, new_state, deps)

    # Check completion
    is_complete = (
        check_completion(current_state, deps=deps)
        or result.output.enough_information_gathered
    )
    deps.logger.debug(
        f"Completion check: {is_complete}, enough_info: {result.output.enough_information_gathered}, no outcomes: {current_state.no_outcomes}, rd2: {current_state.rd2}"
    )
    if is_complete:
        deps.state_logger.info(
            {"state": current_state.model_dump(exclude_none=True)},
            extra={"event": "complete"},
        )
        return True, None

    # Log operator question (may be None)
    next_question = result.output.next_question
    deps.messages_logger.info(
        "",
        extra={"speaker": "operator", "msg_text": next_question},
    )

    # Log extraction event if caller response present
    if caller_msg is not None and new_state is not None:
        deps.state_logger.info(
            {
                "result": new_state.model_dump(exclude_none=True),
                "question": next_question,
                "response": caller_msg,
            },
            extra={"event": "extraction"},
        )

    return False, next_question


def merge_state(
    current_state: EmergencyCall,
    new_state: EmergencyCall,
    deps: Settings,
) -> None:
    """Process a state update by merging and logging.

    Args:
        state: Current state to merge into (modified in place)
        new_state: New state to merge from
        deps: Settings with loggers
    """

    deps.logger.debug(f"Old State:\t{current_state.model_dump(exclude_none=True)}")
    deps.logger.debug(f"New State:\t{new_state.model_dump(exclude_none=True)}")

    # Merge new state into existing state
    _ = ignore_empty_merger.merge(current_state.__dict__, new_state.__dict__)

    # Log the merged state as JSON (same format as graph's MergeState node)
    state_data = current_state.model_dump(exclude_none=True)
    deps.state_logger.info(
        {"state": state_data},
        extra={"event": "state_merged"},
    )
    deps.logger.debug(f"Merged State:\t{state_data}")


def check_completion(state: EmergencyCall, deps: Settings | None = None) -> bool:
    """Check if the emergency call has reached a completion state.

    Args:
        state: Current emergency call state

    Returns:
        True if any completion condition is met (rd2, cpr_needed, time_critical)
    """

    if state.no_outcomes:
        return False

    # if state.enough_information_gathered:
    #     return True

    if state.rd1:
        (deps.logger if deps else logger).info("reached RD1")

    return any([state.rd2, state.cpr_needed])  # , state.time_critical


def save_agent_run_results(
    state: EmergencyCall,
    message_history: list[ModelMessage],
    deps: Settings,
) -> None:
    """Save the final state, schema, and message history to JSON files.

    Args:
        state: Final emergency call state
        message_history: Complete message history
        deps: Settings with save_path
    """
    save_message_history_json(message_history, deps.save_path)

    save_state_json(state, deps.save_path)


# FIXME: Why does this get agent and result as input? Improve API?
async def talk_to_user(
    state: EmergencyCall,
    result: AgentRunResult[DialogueOutput],
    agent: Agent[Settings, DialogueOutput],
    deps: Settings,
) -> AgentRunResult[DialogueOutput]:
    """Talk to the user based on the agent's output type. Intended to be used without graphs with agents directly."""
    question: str | None = None
    response: str | None = None

    # If the agent produced a question, present it to the operator and capture their response
    if isinstance(result.output.next_question, str):
        question = result.output.next_question
        deps.messages_logger.info(
            "", extra={"speaker": "operator", "msg_text": question}
        )

        user_response: str | None = await prompt_user(question, deps=deps)
        response = user_response

        if user_response:
            deps.messages_logger.info(
                "", extra={"speaker": "caller", "msg_text": user_response}
            )

    # Build the follow-up prompt for the agent incorporating current state and caller response
    followup_prompt = f"current state: {state} \nuser response: {response}"

    # Run the agent with capture using the accumulated messages from the previous run
    prev_messages = result.all_messages()
    new_result, captured_messages = await run_agent_with_capture(
        agent,
        prev_messages,
        followup_prompt,
    )
    from devtools import debug

    debug(new_result)

    # Log provider/model info when available
    response_obj = extract_last_model_response(new_result, captured_messages)
    if response_obj:
        deps.logger.info(
            f"provider={response_obj.provider_name} model={response_obj.model_name}"
        )

    # Update history and merge state; pass caller response for extraction logging
    update_history_and_merge_state(
        prev_messages, new_result, state, deps, caller_msg=response
    )

    return new_result


async def main():
    """Run the emergency call agent in CLI mode."""
    state = EmergencyCall()

    deps = Settings(name="llm_loop")
    agent = build_emergency_agent(deps)

    result: AgentRunResult[DialogueOutput] = await agent.run()
    while result := await talk_to_user(state, result, agent, deps=deps):
        if check_completion(state, deps=deps):
            print("Outcome reached")
            break

    # Flush loggers to ensure all data is written
    flush_logger(deps.messages_logger)
    flush_logger(deps.state_logger)

    # Save results to files
    save_agent_run_results(state, result.all_messages(), deps)


if __name__ == "__main__":
    asyncio.run(main())
