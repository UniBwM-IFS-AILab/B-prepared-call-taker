from enum import StrEnum
from functools import lru_cache

from pydantic_ai.agent import Agent
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.output import PromptedOutput
from rich import print

from ems_prepared.agents.system_prompt import system_prompt
from ems_prepared.dialogue_state.emergency_call_state import EmergencyCall
from ems_prepared.model.context import Settings
from ems_prepared.util.models import build_models

state_fill_prompt: system_prompt = system_prompt(
    task=(
        "You receive a user provided Answer to a question about the situation"
        "Determine if the caller confirms the current instruction, reports EMS arrival, or needs further assistance."
    ),
    rules=(
        "If the Caller needs help assist him with completing the current task."
        "If the Caller confirms the current instruction, return confirmed."
        "If the Caller says the ambulance, EMS, or paramedics have arrived, return ems_arrived."
        "Return a string when the caller needs more guidance or you need to ask a clarifying follow-up."
    ),
    decisions=(
        "Also ask the caller to specify when you are unsure if a variable should be set or not."
        "Also ask further if the answer does not provide enough information to fill the variable fully."
        "Confirmations can be any affirmation including okay and ready"
    ),
)


class CallerFeedback(StrEnum):
    CONFIRMED = "confirmed"
    EMS_ARRIVED = "ems_arrived"


async def var_fill_task(
    prompt: str,
    user_response: str,
    state: EmergencyCall,
    deps: Settings,
) -> CallerFeedback | str:
    """Extract structured information from one caller response."""
    agent_task: str = (
        f"Instruction: {prompt}"
        #
        f"Response: {user_response}"
        #
        f"State: {state}"
        # f"State: {state.model_dump_json(indent=2)}"
        # f"Schema: {state.model_json_schema(mode='serialization')}"
    )

    result = await get_var_fill_agent().run(user_prompt=agent_task, deps=deps)

    print(result.usage())

    output = result.output
    if isinstance(output, (CallerFeedback, str)):
        return output
    raise TypeError(f"Unsupported var_fill_task output type: {type(output)!r}")


@lru_cache(maxsize=1)
def get_var_fill_agent():
    return Agent(
        FallbackModel(
            *build_models(
                "github:gpt-4.1-mini",
            )
        ),
        output_type=PromptedOutput(
            outputs=[CallerFeedback, str],
            name="Structured Output if possible",
            description="A caller feedback signal or a string with further instructions.",
            template="Return valid JSON that matches this schema: {schema}",
        ),
        deps_type=Settings,
        instructions=(state_fill_prompt.full_prompt),
    )
