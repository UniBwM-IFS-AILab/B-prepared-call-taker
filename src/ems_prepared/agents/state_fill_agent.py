# from google.genai.types import HarmBlockThreshold, HarmCategory


from ems_prepared.models.google_models import build_gemini_flash_model
from deepdiff import DeepDiff
from pydantic_ai.agent import Agent, AgentRunResult
from pydantic_ai.models.google import GoogleModelSettings
from pydantic_graph import GraphRunContext
from rich import print

from ems_prepared.models.openai_models import build_gpt4o_model
from ems_prepared.models.system_prompt import system_prompt
from ems_prepared.agents.reusable_prompts import calltaker_role
from ems_prepared.settings import Settings
from ems_prepared.state_model.emergency_call_state import EmergencyCall

state_fill_prompt = system_prompt(
    role=calltaker_role,
    task=(
        "You receive a user provided Answer to a question about the situation"
        "Extract Values from the Answer to fit the variables defined in the State."
    ),
    rules=(
        "Only return Json as a string according to the schema, unless you can't extract new values from the Answer compared to the current state. Only then, ask the user for more information"
        "When you cannot extract new data, you are not allowed to ask for specific fields directly."
        "None signifies unknown values"
        "Return only a valid JSON object that satisfies the schema above. Do not include any additional keys or explanatory text."
        "If the user's message contains a value for any field, copy that value into the JSON. Leave a field null only when the user truly did not supply it."
    ),
    decisions=(
        "Ask the user for more information if you can't extract new values from the Answer compared to the current state."
        "Also ask for specification if you are unsure if a variable should be set or not"
        "Also ask further if the answer does not provide enough information to fill the variable fully."
    ),
)


settings = GoogleModelSettings(
    temperature=1.0,
    extra_body={
        "response_mime_type": "application/json",
        "response_schema": EmergencyCall,
    },
)


async def state_fill_task(
    # agent: Agent[str, EmergencyCall],
    prompt: str,
    user_response: str,
    ctx: GraphRunContext[EmergencyCall, Settings],
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
        f"Question: {prompt}"
        #
        f"Answer: {user_response}"
        #
        f"State: {ctx.state}"
        # f"State: {state.model_dump_json(indent=2)}"
        # f"Schema: {state.model_json_schema(mode='serialization')}"
    )

    result: AgentRunResult[EmergencyCall | str] = await state_fill_agent.run(
        user_prompt=agent_task, deps=ctx.deps
    )

    print(result.usage())

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

            # TODO: remove everything until first \n char instead
            if input.startswith("json"):
                input = input.removeprefix("json")

        # input = input[input.find("\n") + 1 :  input.rfind("\n")]

        # try to produce String at the end of methods
        try:
            return EmergencyCall.model_validate_json(input)
        except Exception as _:
            return input

    return input


state_fill_agent = Agent(
    build_gemini_flash_model(),
    output_type=[EmergencyCall, str],
    deps_type=Settings,
    system_prompt=(state_fill_prompt.full_prompt),
)

if __name__ == "__main__":
    state = EmergencyCall()
    prompt: str = (
        "Extract Values from the Statement to fit the variables defined int the State."
        "Ask the user for more information if you can't extract new values from the Statement compared to the current state."
        "Statement: here is Carl, there is a man that fell off his bike. He is bleeding and holding his knee"
        "State: {state}"
    )
    deps = Settings(name="state_fill_agent_main")

    result = state_fill_agent.run_sync(
        user_prompt=prompt.format(state=state), deps=deps
    )
    print(result)
    state2 = result.output
    result2 = state_fill_agent.run_sync(
        user_prompt=prompt.format(state=state2), deps=deps
    )
    print(result2)

    if isinstance(result2.output, EmergencyCall):
        print("2nd call to agent found more info when it shouldnt")
        print(f"Diff: {DeepDiff(result2.output, result.output)}")
