from functools import lru_cache
from typing import TypedDict

from pydantic_ai.agent import Agent, AgentRunResult
from pydantic_ai.messages import ModelMessage
from pydantic_graph import GraphRunContext

from ems_prepared.agents.history_processors import remove_before_extracion_processor
from ems_prepared.agents.reusable_prompts import (
    BASE_SYSTEM_PROMPT,
    extend_system_prompt,
)
from ems_prepared.dialogue_state.emergency_call_state import EmergencyCall
from ems_prepared.dialogue_state.meta_state import GraphState
from ems_prepared.dialogue_state.structured_output import (
    NonEmptyEmergencyCall,
    NonEmptyStr,
)
from ems_prepared.model.context import Settings
from ems_prepared.util.models import build_fallback_agent

type StateFillOutput = NonEmptyEmergencyCall | NonEmptyStr
type StateFillRunResult = AgentRunResult[StateFillOutput] | AgentRunResult[NonEmptyStr]


class _StateFillRunArgs(TypedDict):
    user_prompt: str
    instructions: str


def build_state_fill_run_args(
    *,
    question: str,
    user_response: str,
    locale: str,
) -> _StateFillRunArgs:
    return {
        "user_prompt": (
            f"Last question from you: {question}.\n"
            f"Answer from caller: {user_response}.\n"
        ),
        "instructions": (
            "Try to extract structured information from the user's answer.\n"
            "Decide if you need any more questions you need to ask.\n"
            f"Only use the following language: {locale}.\n"
        ),
    }


def _record_state_fill_result(
    ctx: GraphRunContext[GraphState, Settings],
    result: StateFillRunResult,
) -> StateFillOutput:
    result_data = (
        result.output.model_dump(exclude_none=True)
        if isinstance(result.output, EmergencyCall)
        else result.output
    )
    ctx.deps.telemetry.logger.debug("state_fill_task result: %s", result_data)
    ctx.state.message_history.extend(result.new_messages())
    return result.output


state_fill_prompt = extend_system_prompt(
    BASE_SYSTEM_PROMPT,
    task=(
        "You receive a user-provided answer to a question about the situation. "
        "Extract values from the answer that fit the variables defined in the model."
    ),
    rules=(
        "Never invent or guess values. Only set fields when the user explicitly provided the information or it follows unambiguously.\n"
        "If you cannot extract any new values, ask a single open follow-up question ONLY if it is likely to enable extracting new state.\n"
        "When you cannot extract new data, you are not allowed to ask for specific fields directly; ask a single open follow-up question.\n"
        "Use null/None for unknown values.\n"
        "Never ask the exact same question twice in a row."
    ),
    decisions=(
        "Verify whether the caller answered the last question; if not, you may re-ask it once. "
    ),
)


@lru_cache(maxsize=1)
def get_state_fill_agent() -> Agent[Settings, NonEmptyEmergencyCall | NonEmptyStr]:
    return build_fallback_agent(
        output_type=[NonEmptyEmergencyCall, NonEmptyStr],  # DialogueOutput
        instructions=state_fill_prompt.full_prompt,
        deps_type=Settings,
    )


@lru_cache(maxsize=1)
def get_contextual_question_agent() -> Agent[Settings, NonEmptyStr]:
    return build_fallback_agent(
        output_type=NonEmptyStr,  # DialogueOutput
        instructions=BASE_SYSTEM_PROMPT.full_prompt,
        history_processors=[remove_before_extracion_processor],
        deps_type=Settings,
    )


async def run_state_fill(
    *,
    question: str,
    user_response: str,
    deps: Settings,
    message_history: list[ModelMessage] | None = None,
) -> StateFillRunResult:
    run_args = build_state_fill_run_args(
        question=question,
        user_response=user_response,
        locale=deps.locale.value,
    )
    initial_result: AgentRunResult[StateFillOutput] = await get_state_fill_agent().run(
        **run_args,
        deps=deps,
    )
    if isinstance(initial_result.output, str) and message_history is not None:
        deps.telemetry.logger.info(
            "Retrying with full message history, current result %s",
            initial_result.output,
        )
        return await get_contextual_question_agent().run(
            **run_args,
            deps=deps,
            message_history=message_history,
        )
    return initial_result


async def state_fill_task(
    question: str,
    user_response: str,
    ctx: GraphRunContext[GraphState, Settings],
) -> StateFillOutput:
    """Extract structured information from a user's response using AI."""
    result = await run_state_fill(
        question=question,
        user_response=user_response,
        deps=ctx.deps,
        message_history=ctx.state.message_history,
    )
    return _record_state_fill_result(ctx, result)
