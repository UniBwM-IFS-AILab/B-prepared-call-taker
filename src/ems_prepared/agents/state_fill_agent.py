# from google.genai.types import HarmBlockThreshold, HarmCategory


from typing import Literal

from deepdiff import DeepDiff
from pydantic_ai.agent import Agent, AgentRunResult
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.output import NativeOutput, PromptedOutput
from pydantic_graph import GraphRunContext
from rich import print

from ems_prepared.agents.reusable_prompts import (
    BASE_SYSTEM_PROMPT,
    extend_system_prompt,
)
from ems_prepared.dialogue_state.emergency_call_state import EmergencyCall
from ems_prepared.dialogue_state.meta_state import GraphState
from ems_prepared.dialogue_state.type_defs import EmergencyType
from ems_prepared.util.models import build_fallback_agent
from ems_prepared.util.settings import Settings

state_fill_prompt = extend_system_prompt(
    BASE_SYSTEM_PROMPT,
    task=(
        "You receive a user-provided answer to a question about the situation. "
        "Extract values from the answer that fit the variables defined in the model."
    ),
    rules=(
        "When you can extract new information, return ONLY a valid JSON object matching the output schema (no extra text, no additional keys).\n"
        "If you cannot extract any new values compared to the current state, then ask the user for more information instead of returning JSON.\n"
        "When you cannot extract new data, you are not allowed to ask for specific fields directly; ask a single open follow-up question.\n"
        "Use null/None represents unknown values; if the user does not know the anser, return empy JSON."
        "Never ask the exact same question twice in a row"
    ),
    decisions=(
        "Verify whether the user answered the question you asked; if not, ask the same question again."
    ),
)


async def state_fill_task(
    # agent: Agent[str, EmergencyCall],
    prompt: str,
    user_response: str,
    ctx: GraphRunContext[GraphState, Settings],
) -> EmergencyCall | str:
    """Extract structured information from a user's response using AI.

    Parameters
    ----------
    prompt : str
        The question that was asked to the user.
    response : str
        The user's response to the question.
    state : EmergencyCall
        The current state of the emergency call.

    Returns
    -------
    EmergencyCall
        The merged state with extracted information.

    """
    agent_task: str = (
        f"Try to extract structured information from the user's answer."
        "\n"
        f"Decide if you need no more questions you need to ask."
        "\n"
        f"Only use the following language: {ctx.deps.locale.value}"
        "\n"
        f"Last question from agent: {prompt} "
        "\n"  #
        f"Answer from caller: {user_response} "
        # "\n"
        f"Current State: {ctx.state.call_state.model_dump(exclude_none=True)}"
        # f"State: {state.model_dump_json(indent=2)}"
        # f"Schema: {state.model_json_schema(mode='serialization')}"
    )

    try:
        result: AgentRunResult[EmergencyCall | str] = await state_fill_agent.run(  # type: ignore
            agent_task,
            deps=ctx.deps,  # type: ignore
            # message_history=ctx.state.message_history,
        )

        # collect history but don't use it here
        ctx.state.message_history.extend(
            result.new_messages()
        )  #  ctx.state.message_history.extend(result.new_messages())

        from devtools import debug

        debug(ctx.state.message_history[-1])
    except Exception as e:
        print("Error during state_fill_agent.run:")
        print(e)
        raise e

    cleaned = response_cleanup(result.output)

    return cleaned


def response_cleanup(input: EmergencyCall | str) -> EmergencyCall | str:
    """Apply various fixes to strings returned by LLMs."""
    if isinstance(input, str):
        # Case: LLM returns markdown codeblock instead of strucured data / code
        if input.startswith("```") and input.endswith("```"):
            print("deteced Markdown codeblock in Agent response")

            input = input.removeprefix("```")
            input = input.removesuffix("```")

            if input.startswith("json"):
                input = input.removeprefix("json")

        # input = input[input.find("\n") + 1 :  input.rfind("\n")]

        # try to produce String at the end of methods
        try:
            return EmergencyCall.model_validate_json(input)
        except Exception as _:
            return input

    return input


state_fill_agent = build_fallback_agent(
    output_type=([EmergencyCall, str]),
    system_prompt=(state_fill_prompt.full_prompt),
)

emergency_type_agent = build_fallback_agent(
    output_type=[Literal[EmergencyType.MEDICAL, EmergencyType.FIRE]],
    system_prompt=(state_fill_prompt.full_prompt),
)

enough_info_prompt = extend_system_prompt(
    BASE_SYSTEM_PROMPT,
    task=(
        "Decide if enough information has been gathered based on the current state to make a final decision. "
        "Return ONLY a JSON boolean: true if enough info is gathered, false otherwise."
    ),
    rules=(
        "You must return a bare JSON boolean (true/false), no extra keys, no text. "
        "Evaluate the completeness of key fields like patient_symptoms, location, and outcomes."
    ),
    decisions=(
        "Return true if the state is sufficient for disposition without more questions; otherwise false."
    ),
)

enough_info_agent = build_fallback_agent(
    output_type=bool,
    system_prompt=(enough_info_prompt.full_prompt),
)

if __name__ == "__main__":
    state = EmergencyCall()
    prompt: str = (
        "Extract Values from the Statement to fit the variables defined int the State."
        "Ask the user for more information if you can't extract new values from the Statement compared to the current state."
        "Statement: here is Carl, there is a man that fell off his bike. He is bleeding and holding his knee"
        "State: {state}"
    )
    deps = Settings(name="state_fill_agent_main", emit=print)

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
