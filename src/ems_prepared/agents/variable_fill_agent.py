# from google.genai.types import HarmBlockThreshold, HarmCategory

from deepdiff import DeepDiff
from pydantic import BaseModel
from pydantic.types import T
from pydantic_ai.agent import Agent, AgentRunResult
from pydantic_ai.models.google import GoogleModelSettings
from rich import print

from ems_prepared.agents.models import gemini_flash_model, gpt4o_model, llama3_model
from ems_prepared.state_model.emergency_call_state import EmergencyCall

ROLE: str = "You are a Call taker in a call center for Emergencies who speaks english and german."

TASK: str = (
    "You receive a user provided Answer to a question about the situation"
    "Determine if tye caller confirms their current instruction or needs further assistance."
)

RULES: str = (
    "If the Caller needs help assist him with completing the current task."
    "If the Caller utters an confirmation, return True."
    "If the Caller provides information that might map to a variable in the State, return the new state as json."
)

DECISIONS: str = (
    "Also ask for specification if you are unsure if a variable should be set or not."
    "Also ask further if the answer does not provide enough information to fill the variable fully."
    "Confirmations can be any affirmation including okay and ready"
)


async def var_fill_task(
    prompt: str,
    user_response: str,
    current_state: BaseModel,
) -> BaseModel | bool | str:
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
        f"State: {current_state}"
        # f"State: {state.model_dump_json(indent=2)}"
        # f"Schema: {state.model_json_schema(mode='serialization')}"
    )

    result: AgentRunResult[BaseModel | bool | str] = await var_fill_agent.run(
        user_prompt=agent_task,  # deps=state
    )

    print(result.usage())

    return result.output


var_fill_agent = Agent[BaseModel, BaseModel | str | bool](
    gpt4o_model,
    output_type=[EmergencyCall, str, bool],
    system_prompt=(ROLE, TASK, DECISIONS, RULES),  # noqa: E501
)


if __name__ == "__main__":
    state = EmergencyCall()
