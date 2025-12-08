"""Agent representing an LLM-Driven loop for variable extraction and outcome determination."""

# pyright: strict
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from pydantic.main import BaseModel
from pydantic_ai.agent import Agent, AgentRunResult
from pydantic_ai.models.fallback import FallbackModel
from pydantic_core import to_jsonable_python
from rich import print

from ems_prepared.agents.system_prompt import system_prompt
from ems_prepared.dialogue_state.emergency_call_state import EmergencyCall
from ems_prepared.policies.pydantic_graph.utils import save_state_json
from ems_prepared.util.custom_deepmerge import ignore_empty_merger
from ems_prepared.util.logger import flush_logger
from ems_prepared.util.models import build_models
from ems_prepared.util.settings import Settings
from ems_prepared.util.user_interaction import prompt_user


class DialogueOutput(BaseModel):
    state: EmergencyCall | None
    next_question: str | None


@dataclass
class AgentPolicy:
    """Container for agent policy state."""

    agent: Agent[None, DialogueOutput]
    state: EmergencyCall
    history: list


def build_emergency_agent() -> Agent[None, DialogueOutput]:
    """Build and return the emergency call agent with configured system prompt.

    Returns:
        Configured PydanticAI agent for emergency call handling
    """

    prompt = system_prompt(
        role=(
            "You are a professional call taker in a 112 emergency call center (Public Safety Answering Point). "
            "Your job is to calmly question callers and capture structured information about the emergency in the State object. "
            "You never give medical or practical instructions; you only gather information."
        ),
        task=(
            "You receive a phone call from a caller who wants to report an emergency. "
            "Begin determining the basic information about the emergency. "
            "Ask clear, focused questions to extract only the information needed to fill the variables defined in the State schema. "
            "The State schema describes all fields you are allowed to care about (personalia and RD1/RD2-related variables). "
            "For each caller answer, extract values that fit the State and update fields only when you can infer them from what was actually said. "
            "If the latest caller message does not contain any new relevant information, do not change the State. "
            "The emergency outcome and dispatch decisions will be determined automatically by other systems. "
            "This is a time-critical situation, so keep the conversation efficient and to the point. "
            "Your goal is to figure out what the emergency is and capture it in the State, not to give advice or instructions on how to deal with it."
        ),
        rules=(
            "Whenever possible, update the State representing the emergency call using only information explicitly provided by the caller. "
            "Never invent or guess values for any field in the State. "
            "Only ask questions that help you fill personalia or RD1-related variables in the State; do not ask about anything outside of the schema. "
            "Before RD1 is true, you may not ask any questions that target RD2 symptoms or RD2-level detail. "
            "Never mention the words 'RD1', 'RD2', 'schema', 'JSON', or 'State' to the caller; these are internal concepts. "
            "Ask only one question at a time. "
            "The caller might not know medical terms, so when you ask about RD1 symptoms use simple, layperson language and short explanations. "
            "If an answer is ambiguous for a specific State field, ask a short, direct follow-up question to clarify that field only. "
            "Do not repeat questions whose answers are already clearly stored in the State, unless you need to verify or resolve a contradiction. "
            "Maintain a single language throughout the call; use the language of the caller’s first message and do not switch languages mid-conversation. "
            # "Do not add another question once the call has been fully processed (personalia and RD1 variables are reasonably complete and the generic post-RD1 question has been asked and answered). "
            "Confirmations from the caller (e.g. 'okay', 'yes', 'ready') count only as acknowledgements and should not change clinical or factual fields unless they clearly answer a question."
        ),
        decisions=(
            "Use the provided State schema to understand which fields belong to personalia and which correspond to RD1 and RD2. "
            # "At the start of the call, focus on collecting personalia and any RD1-related information required by the State: "
            # "for example, identity and contact details of the caller/patient, location of the emergency, and the basic nature of what has happened. "
            "Continue asking targeted questions only as needed to determine and fill RD1-related variables. "
            # "As soon as the RD1 field in State is True"
            "Only AFTER RD1 is True are you allowed to ask for more details in a single generic and open question"
            # "'Please briefly describe exactly what is happening there right now.' or "
            # "'In a few words, can you tell me what is going on?'. "
            "Do not mention RD2 or any RD2 symptom names directly in this question. "
            # "Use ONLY the caller’s answer to this one generic question, together with information already collected, to decide whether RD2 should be set to True or False in the State. "
            "Do not ask any additional new questions specifically targeting RD2 symptoms after this generic question. "
            "If you are unsure whether a specific State variable should be set, ask the caller to clarify using simple, concrete language. "
            "If the caller’s reply still does not clearly support setting a field, leave that field unset or unknown instead of guessing. "
            "Once personalia and RD1 or RD2 have been filled as far as reasonably possible, stop asking new questions and allow the conversation to end naturally."
        ),
    )

    agent = Agent(
        model=FallbackModel(
            *build_models(
                "github:gpt-4.1-mini",
                "google-gla:gemini-2.5-flash",
                "google-gla:gemini-2.5-pro",
            )
        ),
        output_type=DialogueOutput,
        system_prompt=prompt.full_prompt,
    )

    return agent  # type: ignore


def process_state_update(
    state: EmergencyCall,
    new_state: EmergencyCall,
    deps: Settings,
) -> None:
    """Process a state update by merging and logging.

    Args:
        state: Current state to merge into (modified in place)
        new_state: New state to merge from
        deps: Settings with loggers
    """
    # Merge new state into existing state
    _ = ignore_empty_merger.merge(state.__dict__, new_state.__dict__)

    # Log state change
    state_data = state.model_dump(exclude_none=True)
    deps.state_logger.info(
        {"state": state_data},
        extra={"event": "extraction"},
    )
    logger.debug(state_data)


def check_completion(state: EmergencyCall) -> bool:
    """Check if the emergency call has reached a completion state.

    Args:
        state: Current emergency call state

    Returns:
        True if any completion condition is met (rd2, cpr_needed, time_critical)
    """
    if state.rd1:
        logger.info("reached RD1")

    return any([state.rd2, state.cpr_needed, state.time_critical])


def save_agent_run_results(
    state: EmergencyCall,
    message_history: list,
    deps: Settings,
) -> None:
    """Save the final state, schema, and message history to JSON files.

    Args:
        state: Final emergency call state
        message_history: Complete message history
        deps: Settings with save_path
    """
    prompts_file_path = Path(deps.save_path / "prompts.json")
    prompts_json: dict[str, str] = to_jsonable_python(message_history)
    _ = prompts_file_path.write_text(json.dumps(prompts_json), encoding="utf-8")

    save_state_json(state, deps.save_path)


async def talk_to_user(
    state: EmergencyCall,
    result: AgentRunResult[DialogueOutput],
    agent: Agent[None, DialogueOutput],
    deps: Settings,
) -> AgentRunResult[DialogueOutput]:
    """Talk to the user based on the agent's output type. Intended to be used without graphs with agents directly."""
    if isinstance(result.output.next_question, str):
        # Log operator question
        deps.messages_logger.info(
            "", extra={"speaker": "operator", "msg_text": result.output}
        )

        user_response: str | None = await prompt_user(
            result.output.next_question, deps=deps
        )

        # Log caller response
        if user_response:
            deps.messages_logger.info(
                "", extra={"speaker": "caller", "msg_text": user_response}
            )

        result: AgentRunResult[DialogueOutput] = await agent.run(
            user_prompt=user_response,
            message_history=result.all_messages(),
        )
    if isinstance(result.output.state, EmergencyCall):
        _ = ignore_empty_merger.merge(state, result.output.state)

        result = await agent.run(
            user_prompt=(
                f"current state: {state}\nAsk the Caller for the next logical symptom."
            ),
            message_history=result.all_messages(),
        )
    return result


async def main():
    """Run the emergency call agent in CLI mode."""
    state = EmergencyCall()
    agent = build_emergency_agent()

    deps = Settings(name="llm_loop")

    result: AgentRunResult[DialogueOutput] = await agent.run()
    while result := await talk_to_user(state, result, agent, deps=deps):
        if isinstance(result.output.state, EmergencyCall):
            logger.debug(
                f"AgentRunResult: {result.output.state.model_dump(exclude_none=True)}"
            )
            new_state = result.output.state
            process_state_update(state, new_state, deps)

            if check_completion(state):
                print("Outcome reached")
                break

    # Flush loggers to ensure all data is written
    flush_logger(deps.messages_logger)
    flush_logger(deps.state_logger)

    # Save results to files
    save_agent_run_results(state, result.all_messages(), deps)


if __name__ == "__main__":
    asyncio.run(main())
