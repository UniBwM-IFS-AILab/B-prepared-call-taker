# from google.genai.types import HarmBlockThreshold, HarmCategory


from typing import Literal

from pydantic_ai.agent import Agent, AgentRunResult
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.output import PromptedOutput
from pydantic_graph import GraphRunContext
from rich import print

from ems_prepared.agents.system_prompt import system_prompt
from ems_prepared.dialogue_state.emergency_call_state import EmergencyCall
from ems_prepared.util.models import build_models
from ems_prepared.util.settings import Settings

state_fill_prompt: system_prompt = system_prompt(
    task=(
        "You receive a user provided Answer to a question about the situation"
        "Determine if the caller confirms their current instruction or needs further assistance."
    ),
    rules=(
        "If the Caller needs help assist him with completing the current task."
        "If the Caller utters an confirmation, return True."
        "If the Caller provides information that might map to a variable in the State, return the new state as json."
    ),
    decisions=(
        "Also ask the caller to specify when you are unsure if a variable should be set or not."
        "Also ask further if the answer does not provide enough information to fill the variable fully."
        "Confirmations can be any affirmation including okay and ready"
    ),
)


async def var_fill_task(
    prompt: str,
    user_response: str,
    ctx: GraphRunContext[EmergencyCall, Settings],
) -> EmergencyCall | bool | str:
    """Extract structured information from a user's response using AI.

    Parameters
    ----------
    question : str
        The question that was asked to the user.
    response : str
        The user's response to the question.
    state : BaseModel
        The current state of the emergency call.

    Returns
    -------
    BaseModel
        The merged state with extracted information.

    """

    agent_task: str = (
        f"Instruction: {prompt}"
        #
        f"Response: {user_response}"
        #
        f"State: {ctx.state}"
        # f"State: {state.model_dump_json(indent=2)}"
        # f"Schema: {state.model_json_schema(mode='serialization')}"
    )

    result: AgentRunResult[
        EmergencyCall | Literal[True] | str
    ] = await var_fill_agent.run(user_prompt=agent_task)

    print(result.usage())

    return result.output


var_fill_agent = Agent(
    FallbackModel(
        *build_models(
            "github:gpt-4.1-mini",
        )
    ),
    output_type=PromptedOutput(
        outputs=[EmergencyCall, Literal[True], str],
        name="Structured Output if possible",
        description="The extracted state or confirmation as boolean. If extraction is not possible, return a string with further instructions.",
        template="Follow the schema: {schema}",
    ),
    deps_type=Settings,
    instructions=(state_fill_prompt.full_prompt),
)

if __name__ == "__main__":
    state = EmergencyCall()
