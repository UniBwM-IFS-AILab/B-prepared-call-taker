"""Agent representing an LLM-Driven loop for variable extraction and outcome determination."""

# pyright: strict
import asyncio
import json
from pathlib import Path

from pydantic_ai.agent import Agent, AgentRunResult
from pydantic_core import to_jsonable_python
from rich import print

from ems_prepared.models.openai_models import build_gpt4o_model
from ems_prepared.models.system_prompt import system_prompt
from ems_prepared.agents.reusable_prompts import calltaker_role
from ems_prepared.util.user_interaction import prompt_user
from ems_prepared.util.settings import Settings
from ems_prepared.dialogue_state.emergency_call_state import EmergencyCall
from ems_prepared.util.custom_deepmerge import ignore_empty_merger


async def talk_to_user(
    state: EmergencyCall,
    result: AgentRunResult[EmergencyCall | str],
    agent: Agent[None, EmergencyCall | str],
) -> AgentRunResult[EmergencyCall | str]:
    if isinstance(result.output, str):
        user_response: str | None = await prompt_user(result.output)
        result = await agent.run(
            user_prompt=user_response,
            message_history=result.all_messages(),
        )
    else:  # elif isinstance(result.output, EmergencyCall):
        _ = ignore_empty_merger.merge(state, result.output)

        result = await agent.run(
            user_prompt=(
                f"current state: {state}\nAsk the Caller for the next logical symptom."
            ),
            message_history=result.all_messages(),
        )
    return result


async def main():
    state_fill_prompt = system_prompt(
        task=(
            "Extract Values from the Answer to fit the variables defined in the State."
            "You receive a user provided Answer to a question about the situation"
        )
    )
    conv_prompt = ...

    prompt = system_prompt(
        role="You are a Call taker in a call center for Emergencies",  # calltaker_role,
        task=(
            "You receive a phone call from a caller who wants to report an emergency."
            "Start with asking who is calling."
            "Next, ask where the emergency is supposed to be."
            "Then ask what happened to determine the type of emergency."
            "In case of an medical emergency. You have to talk to the caller to find out what symptoms the patient suffers from."
            "Extract Values from the caller's answers to fit the variables defined in the State."
            "The emergency outcome will be determined automatically."
        ),
        rules=(
            "Whenever possible, update the state variables."
            "Only ask one question at a time."
            "Do not ask for any RD2 Boolean directly"
            "The Caller might not know the medical terms for these symptoms, you will have to explain them in laymen terms."
            "Do not switch language within one conversation."
        ),
        decisions=(
            "When the RD1 state is True, ask for more details to determine if RD2 is true as well"
            "Also ask the caller to specify if you are unsure if a variable should be set or not."
            "Also ask further if the answer does not provide enough information to fill the variable fully."
            "Confirmations can be any affirmation including okay and ready"
        ),
    )

    state = EmergencyCall()
    state_fill_agent_new = Agent(
    model=build_gpt4o_model(),
        output_type=[EmergencyCall, str],
        system_prompt=prompt.full_prompt,
    )
    conv_agent = Agent(
    model=build_gpt4o_model(),
        output_type=[EmergencyCall, str],
        system_prompt=prompt.full_prompt,
    )

    print(f"Full system prompt:\n {prompt.full_prompt}")

    result: AgentRunResult[EmergencyCall | str] = await state_fill_agent_new.run(
        # user_prompt=prompt.full_prompt,
    )

    while True:
        result = await talk_to_user(state, result, state_fill_agent_new)

        if isinstance(result.output, EmergencyCall):
            state = result.output
            print(f"new current state: {state.model_dump(exclude_none=True)}")
            # print(f"result.output: {state.model_dump(exclude_none=True)}")

            if state.rd1:
                print("reached RD1")

            if any([state.rd2, state.cpr_needed, state.urgency_needed]):
                print("Outcome reached")
                break

    deps = Settings(name="naive_loop")
    save_path = deps.log_dir / deps.file_name
    messages_file_path = Path(f"{save_path}_messages.json")
    messages_json: dict[str, str] = to_jsonable_python(result.all_messages()) # pyright: ignore[reportAny]
    _ = messages_file_path.write_text(json.dumps(messages_json), encoding="utf-8")

    state_file_path = Path(f"{save_path}_final_state.json")
    _ = state_file_path.write_text(
        state.model_dump_json(
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    asyncio.run(main())
