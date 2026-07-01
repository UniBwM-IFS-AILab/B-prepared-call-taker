from functools import lru_cache

from deepdiff.diff import DeepDiff
from pydantic_ai.agent import AgentRunResult
from pydantic_graph import GraphRunContext
from rich import print

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
from ems_prepared.util.models import build_fallback_agent
from ems_prepared.util.settings import Settings

state_fill_prompt = extend_system_prompt(
    BASE_SYSTEM_PROMPT,
    task=(
        "You receive a user-provided answer to a question about the situation. "
        "Extract values from the answer that fit the variables defined in the model."
    ),
    rules=(
        "Never invent or guess values. Only set fields when the user explicitly provided the information or it follows unambiguously.\n"
        # "When you can extract new information, return ONLY a valid JSON object matching the output schema (no extra text, no additional keys).\n"
        "If you cannot extract any new values, ask a single open follow-up question ONLY if it is likely to enable extracting new state.\n"
        # "If you cannot extract any new values compared to the current state, ask a single open follow-up question ONLY if it is likely to enable extracting new state.\n"
        "When you cannot extract new data, you are not allowed to ask for specific fields directly; ask a single open follow-up question.\n"
        "Use null/None for unknown values.\n"
        # "If message history is provided, do not repeat any follow-up question already present there (even if paraphrased); if you cannot think of a meaningfully new follow-up question, return empty JSON {}.\n"
        "Never ask the exact same question twice in a row."
    ),
    decisions=(
        "Verify whether the caller answered the last question; if not, you may re-ask it once. "
        "If message history is provided and you already asked that question (or a close paraphrase), "
        # "do not repeat it and instead ask a meaningfully different open question or extract state if possible."
    ),
)


@lru_cache(maxsize=1)
def get_state_fill_agent():
    return build_fallback_agent(
        output_type=[NonEmptyEmergencyCall, NonEmptyStr],  # DialogueOutput
        instructions=state_fill_prompt.full_prompt,
    )


@lru_cache(maxsize=1)
def get_contextual_question_agent():
    return build_fallback_agent(
        output_type=NonEmptyStr,  # DialogueOutput
        instructions=BASE_SYSTEM_PROMPT.full_prompt,
        history_processors=[remove_before_extracion_processor],
    )


async def state_fill_task(
    question: str,
    user_response: str,
    ctx: GraphRunContext[GraphState, Settings],
) -> NonEmptyEmergencyCall | NonEmptyStr:
    """Extract structured information from a user's response using AI."""

    # agent_task: str = (
    #     "Try to extract structured information from the user's answer.\n"
    #     "Decide if you need any more questions you need to ask.\n"
    #     f"Only use the following language: {ctx.deps.locale.value}.\n"
    #     f"Last question from you: {question}.\n"
    #     f"Answer from caller: {user_response}.\n"
    # )
    agent_task: dict[str, str] = {
        "user_prompt": (
            f"Last question from you: {question}.\n"
            f"Answer from caller: {user_response}.\n"
        ),
        "instructions": (
            "Try to extract structured information from the user's answer.\n"
            "Decide if you need any more questions you need to ask.\n"
            f"Only use the following language: {ctx.deps.locale.value}.\n"
        ),
    }

    # message_history: Sequence[ModelMessage] | None = ctx.state.message_history
    result: AgentRunResult[
        NonEmptyEmergencyCall | NonEmptyStr
    ] = await get_state_fill_agent().run(
        **agent_task,
        deps=ctx.deps,
    )
    if isinstance(result.output, str):  # message_history is None and
        ctx.deps.logger.info(
            f"Retrying with full message history, current result {result.output}"
        )
        result: AgentRunResult[NonEmptyStr] = await get_contextual_question_agent().run(  # type: ignore
            **agent_task,
            deps=ctx.deps,  # type: ignore
            message_history=ctx.state.message_history,
        )

    result_data = (
        result.output.model_dump(exclude_none=True)
        if isinstance(result.output, EmergencyCall)
        else result.output
    )
    ctx.deps.logger.debug("state_fill_task result: %s", result_data)
    ctx.state.message_history.extend(result.new_messages())
    return result.output


if __name__ == "__main__":
    state = EmergencyCall()
    prompt: str = (
        "Extract Values from the Statement to fit the variables defined int the State."
        "Ask the user for more information if you can't extract new values from the Statement compared to the current state."
        "Statement: here is Carl, there is a man that fell off his bike. He is bleeding and holding his knee"
        "State: {state}"
    )
    deps = Settings(name="state_fill_agent_main", emit=print)
    state_fill_agent = get_state_fill_agent()

    result = state_fill_agent.run_sync(  # type: ignore
        user_prompt=prompt.format(state=state),
        deps=deps,  # type: ignore
    )
    print(result)
    state2 = result.output
    result2 = state_fill_agent.run_sync(  # type: ignore        user_prompt=prompt.format(state=state2),
        deps=deps,  # type: ignore
    )
    print(result2)

    if isinstance(result2.output, EmergencyCall):
        print("2nd call to agent found more info when it shouldn't")
        print(f"Diff: {DeepDiff(result2.output, result.output)}")
