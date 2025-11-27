from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any, override

from pydantic.dataclasses import dataclass
from pydantic.main import BaseModel
from pydantic_graph.graph import GraphRunResult
from pydantic_graph.nodes import BaseNode, End, GraphRunContext
from rich import print

from ems_prepared.agents.variable_fill_agent import var_fill_task
from ems_prepared.dialogue_state.emergency_call_state import EmergencyCall
from ems_prepared.policies.pydantic_graph.type_defs import EmergencyNode
from ems_prepared.policies.pydantic_graph.utils import (
    init_graph,
    save_mermaid_graph,
    save_state_json,
)
from ems_prepared.util.custom_deepmerge import ignore_empty_merger
from ems_prepared.util.settings import LOCALE, Settings
from ems_prepared.util.user_interaction import (
    converse_with_user,
    tell_user,
)

instructions = {
    "en": [
        "Please let me know once the Ambulance is here. \n"
        "We will now start with CPR. Please follow my instructions carefully."
        # "We have to perform CPR now. \n"
        "Please follow the instructions, confirm once you finished. \n"
        # "If possible, place the patient on the floor so that he or she is lying on their back. Is there enough space there?"
        "Kneel beside the patient's chest so that your knees are next to each other at chest level."
        # "Expose the patient's upper body.",
        # "Place the heel of one hand on the middle of the patient's bony chest, i.e., on the lower half of the sternum—clearly above the pit of the stomach."
        # "Place the heel of your second hand on the back of your first hand.",
        "Lean over the patient so that you can push straight down with your arms extended.",
        "Perform 30 chest compressions. Alternate between deep compressions, at least 5 cm deep, and complete release without losing contact with the chest.",
    ],
    "de": [
        "Ein Krankenwagen ist auf dem Weg zu Ihnen, bitte geben Sie mir Bescheid sobald er eintrifft."
        "Bitte folgen Sie den Anweisungen, bestätigen Sie sobald fertig sind."
        "Legen Sie den Patienten nach Möglichkeit auf den Fußboden, so dass er/sie auf dem Rücken liegt.",  #  Ist dort genug Platz?
        "Knien Sie sich seitlich neben den Brustkorb des Patienten, so dass Ihre Knie nebeneinander in Höhe der Brust sind."
        "Machen Sie den Oberkörper des Patienten frei.",
        "Legen Sie einen Handballen Ihrer Hand auf die Mitte des knöchernen Brustkorbs vom Patienten, also auf die untere Hälfte des Brustbeins - das ist deutlich oberhalb der Magengrube."
        "Legen Sie den Handballen Ihrer zweiten Hand auf den Handrücken Ihrer ersten Hand.",
        "Beugen Sie sich so über den Patienten, dass Sie mit gestreckten Armen senkrecht drücken können.",
        "Drücken Sie immer wieder kräftig, mindestens 5 cm tief, auf den Brustkorb. Sie sollten im Wechsel tief drücken und dann komplett entlasten - ganz runter und ganz hoch, ohne den Kontakt zum Brustkorb zu verlieren.",
    ],
}
questions = {
    "en": [
        "Will the emergency services have unobstructed access when they arrive?",
        "Can you send someone to the street to signal them?",
    ],
    "de": [
        "Hat der Rettungsdienst ungehindert Zutritt, wenn er gleich bei Ihnen eintrifft?",
        "Können Sie jemanden auf die Straße schicken, der sich bemerkbar macht?",
    ],
}

# TODO: move to state
instruct_iterator: Iterator[str] = iter(instructions[LOCALE])
ask_iterator: Iterator[str] = iter(questions[LOCALE])


### PREPARATION
@dataclass
class Instruct(BaseNode[EmergencyCall, Settings]):
    instruction: str

    @override
    async def run(
        self,
        ctx: GraphRunContext[EmergencyCall, Settings],
    ) -> Instruct | ArtificialVentilation | EMSArrived:
        parse_response = await converse_with_user(self.instruction, ctx, var_fill_task)

        try:
            if parse_response is True:
                return Instruct(next(instruct_iterator))

            elif parse_response is False:
                raise TypeError(
                    f"Error with type returned by var_fill_agent, value is {parse_response}"
                )
            elif type(parse_response) is str:
                pass
            elif type(parse_response) is EmergencyCall:
                _ = ignore_empty_merger.merge(
                    ctx.state.__dict__, parse_response.__dict__
                )
                if ctx.state.ems_arrived:
                    return EMSArrived()
            return Instruct(next(instruct_iterator))
        except StopIteration:
            return ArtificialVentilation()


###


### CPR LOOP
# @dataclass
# class GenericInstruct(BaseNode[StateT, DepsT]):
#     """A base node for a step in the CPR cycle (30 compressions, 2 breaths)."""

#     instruction: str
#     next_node_type: type[BaseNode]

#     @override
#     async def run(
#         self,
#         ctx: GraphRunContext[StateT, DepsT],
#     ) -> None:
#         pass


async def instruct_user(
    ctx: GraphRunContext[EmergencyCall, Settings],
    instruction: str,
    next_node: EmergencyNode,
) -> EMSArrived | EmergencyNode | None:
    parse_result = await converse_with_user(instruction, ctx, var_fill_task)

    if type(parse_result) is EmergencyCall:
        _ = ignore_empty_merger.merge(ctx.state.__dict__, parse_result.__dict__)
        if ctx.state.ems_arrived:
            return EMSArrived()
    elif parse_result is True:
        return next_node
    elif parse_result is False:
        raise TypeError(
            f"Error with type returned by var_fill_agent, value is {parse_result}"
        )
    elif type(parse_result) is str:
        # user_response: str | None = await prompt_user(parse_result)
        # return await instruct_user(
        #     ctx, instruction=user_response, next_node_type=next_node_type
        # )
        # result = await converse_with_user(
        #     prompt=parse_result, state=ctx.state, task_function=var_fill_task
        # )
        return await instruct_user(ctx, parse_result, next_node)
    else:
        raise TypeError(
            f"Unexpected type returned by var_fill_agent, value is {parse_result}, type is {type(parse_result)}"
        )
    return next_node


@dataclass
class ArtificialVentilation(BaseNode[EmergencyCall, Settings]):
    """Represents ventilation using mouth-to-mouth, mouth-to-nose or with an external device."""

    instruction: str = {
        "en": "Please perform two rescue breaths and confirm when done.",
        "de": "Bitte beatmen Sie den Patienten zwei mal und bestätigen Sie danach.",
    }[LOCALE]

    @override
    async def run(
        self,
        ctx: GraphRunContext[EmergencyCall, Settings],
    ) -> EmergencyNode | EMSArrived:
        result = await instruct_user(ctx, self.instruction, ChestCompression())
        if result is None:
            raise TypeError("Return is none, while it should not be")

        return result


@dataclass
class ChestCompression(BaseNode[EmergencyCall, Settings]):
    instruction: str = {
        "en": "Please perform 30 chest compressions and confirm when done.",
        "de": "Bitte führen Sie 30 Herzdruckmassagen durch und bestätigen Sie danach.",
    }[LOCALE]

    @override
    async def run(
        self,
        ctx: GraphRunContext[EmergencyCall, Settings],
    ) -> EmergencyNode | EMSArrived:
        result = await instruct_user(ctx, self.instruction, ArtificialVentilation())
        if result is None:
            raise TypeError("Return is none, while it should not be")
        return result


###


@dataclass
class AEDArrived(EmergencyNode):
    @override
    async def run(
        self,
        ctx: GraphRunContext[EmergencyCall, Settings],
    ):
        pass


@dataclass
class EMSArrived(EmergencyNode):
    @override
    async def run(
        self,
        ctx: GraphRunContext[EmergencyCall, Settings],
    ) -> End[EmergencyCall]:
        await tell_user(
            {
                "en": "Please hand over to the paramedics",
                "de": "Bitte übergeben Sie an die Einsatzkräfte",
            }[LOCALE],
            ctx.deps,
        )

        return End(ctx.state)

    # @dataclass
    # class Placeholder(BaseNode[EmergencyCall, Settings]):
    #     @override
    #     async def run(
    #         self,
    #         ctx: GraphRunContext[EmergencyCall, Settings],
    #     ):
    #         pass


async def run_graph(
    init_state: EmergencyCall, deps: Settings = Settings(name="tcpr")
) -> GraphRunResult[EmergencyCall, EmergencyCall] | None:
    graph, persistence = await init_graph(
        node_list=[Instruct, ArtificialVentilation, ChestCompression, EMSArrived],
        # Instruct(), persistence=persistence, state=EmergencyCall()
        init_node=Instruct(next(instruct_iterator)),
        deps=deps,
        init_state=init_state,
        prefix="tcpr_",
    )

    async with graph.iter_from_persistence(persistence=persistence, deps=deps) as run:
        print(run.state)

        async for node in run:
            if isinstance(node, BaseNode):
                print(f"[Node] {node.get_node_id()}")  # type: ignore
            else:
                print(node)
    if run.result is not None:
        _ = asyncio.create_task(save_state_json(run.result, deps.save_path))
        _ = asyncio.create_task(
            save_mermaid_graph(
                graph,
                deps.save_path,
            )
        )
        _ = (deps.save_path / "deps.json").write_text(
            data=deps.model_dump_json(indent=2), encoding="utf-8"
        )

    return run.result


async def main(deps=None, state=None):
    """Function to test the graph in isolation."""

    deps: Settings = deps or Settings(name="tcp_subgraph_only")
    state: EmergencyCall = state or EmergencyCall(cardiac_arrest=True)

    result = await run_graph(init_state=state, deps=deps)
    if result is not None:
        print("Graph finished with state:", result.state)


if __name__ == "__main__":
    asyncio.run(main())
