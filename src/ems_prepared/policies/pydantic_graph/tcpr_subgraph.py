from __future__ import annotations

import asyncio
from typing import ClassVar, Literal, overload, override

from pydantic.dataclasses import dataclass
from pydantic_graph import End, GraphRunContext
from pydantic_graph.graph import Graph

from ems_prepared.agents.variable_fill_agent import CallerFeedback, var_fill_task
from ems_prepared.dialogue_state.emergency_call_state import EmergencyCall
from ems_prepared.dialogue_state.meta_state import GraphState
from ems_prepared.model.context import Locale, Settings
from ems_prepared.policies.pydantic_graph.custom_persistence.resumable_file_persistence import (
    setup_resumable_file_persistence,
)
from ems_prepared.policies.pydantic_graph.graph_helpers import (
    restore_graph,
    save_mermaid_graph,
)
from ems_prepared.policies.pydantic_graph.nodes import MessageNode, QuestionNode
from ems_prepared.policies.pydantic_graph.type_defs import EmergencyNode
from ems_prepared.policies.shared import record_completion_artifacts
from ems_prepared.util.logger import flush_logger
from ems_prepared.util.user_interaction import prompt_user

instructions = {
    Locale.EN: [
        "Please let me know once the Ambulance is here. \n"
        "We will now start with CPR. Please follow my instructions carefully."
        "Please follow the instructions, confirm once you finished. \n"
        "Kneel beside the patient's chest so that your knees are next to each other at chest level."
        "Lean over the patient so that you can push straight down with your arms extended.",
        "Perform 30 chest compressions. Alternate between deep compressions, at least 5 cm deep, and complete release without losing contact with the chest.",
    ],
    Locale.DE: [
        "Ein Krankenwagen ist auf dem Weg zu Ihnen, bitte geben Sie mir Bescheid sobald er eintrifft."
        "Bitte folgen Sie den Anweisungen, bestätigen Sie sobald fertig sind."
        "Legen Sie den Patienten nach Möglichkeit auf den Fußboden, so dass er/sie auf dem Rücken liegt.",
        "Knien Sie sich seitlich neben den Brustkorb des Patienten, so dass Ihre Knie nebeneinander in Höhe der Brust sind."
        "Machen Sie den Oberkörper des Patienten frei.",
        "Legen Sie einen Handballen Ihrer Hand auf die Mitte des knöchernen Brustkorbs vom Patienten, also auf die untere Hälfte des Brustbeins - das ist deutlich oberhalb der Magengrube."
        "Legen Sie den Handballen Ihrer zweiten Hand auf den Handrücken Ihrer ersten Hand.",
        "Beugen Sie sich so über den Patienten, dass Sie mit gestreckten Armen senkrecht drücken können.",
        "Drücken Sie immer wieder kräftig, mindestens 5 cm tief, auf den Brustkorb. Sie sollten im Wechsel tief drücken und dann komplett entlasten - ganz runter und ganz hoch, ohne den Kontakt zum Brustkorb zu verlieren.",
    ],
}

confirmation_questions = {
    "instruct": {
        Locale.EN: "Please confirm once you finished, ask if you need help, or tell me if EMS has arrived.",
        Locale.DE: "Bitte bestaetigen Sie, sobald Sie fertig sind, fragen Sie nach Hilfe oder sagen Sie mir, falls der Rettungsdienst eingetroffen ist.",
    },
    "ventilation": {
        Locale.EN: "Please confirm when done, ask if you need help, or tell me if EMS has arrived.",
        Locale.DE: "Bitte bestaetigen Sie, sobald Sie fertig sind, fragen Sie nach Hilfe oder sagen Sie mir, falls der Rettungsdienst eingetroffen ist.",
    },
    "compression": {
        Locale.EN: "Please confirm when done, ask if you need help, or tell me if EMS has arrived.",
        Locale.DE: "Bitte bestaetigen Sie, sobald Sie fertig sind, fragen Sie nach Hilfe oder sagen Sie mir, falls der Rettungsdienst eingetroffen ist.",
    },
}

phase_messages = {
    "ventilation": {
        Locale.EN: "Please perform two rescue breaths.",
        Locale.DE: "Bitte beatmen Sie den Patienten zwei mal.",
    },
    "compression": {
        Locale.EN: "Please perform 30 chest compressions.",
        Locale.DE: "Bitte fuehren Sie 30 Herzdruckmassagen durch.",
    },
}

InstructionPhase = Literal["instruct", "ventilation", "compression"]
RunTCPRNode = QuestionNode | MessageNode | End[EmergencyCall]


@dataclass
class Instruct(EmergencyNode):
    step_index: int = 0

    @override
    async def run(
        self,
        ctx: GraphRunContext[GraphState, Settings],
    ) -> TCPRMessage | ArtificialVentilation:
        instruction_steps = instructions[ctx.deps.locale]
        if self.step_index >= len(instruction_steps):
            return ArtificialVentilation()
        return TCPRMessage(
            messages={
                locale: messages[self.step_index]
                for locale, messages in instructions.items()
            },
            phase="instruct",
            step_index=self.step_index,
        )


@dataclass
class ArtificialVentilation(EmergencyNode):
    @override
    async def run(
        self,
        ctx: GraphRunContext[GraphState, Settings],
    ) -> TCPRMessage:
        _ = ctx
        return TCPRMessage(
            messages=phase_messages["ventilation"],
            phase="ventilation",
        )


@dataclass
class ChestCompression(EmergencyNode):
    @override
    async def run(
        self,
        ctx: GraphRunContext[GraphState, Settings],
    ) -> TCPRMessage:
        _ = ctx
        return TCPRMessage(
            messages=phase_messages["compression"],
            phase="compression",
        )


@dataclass
class TCPRMessage(MessageNode):
    messages: dict[Locale, str]
    phase: InstructionPhase
    step_index: int = 0

    @override
    async def run(
        self,
        ctx: GraphRunContext[GraphState, Settings],
    ) -> Instruct | TCPRQuestion:
        message = self.messages[ctx.deps.locale]
        ctx.deps.telemetry.messages_logger.info(
            "", extra={"speaker": "operator", "msg_text": message}
        )

        if self.phase == "instruct":
            instruction_steps = instructions[ctx.deps.locale]
            if self.step_index + 1 < len(instruction_steps):
                return Instruct(step_index=self.step_index + 1)

        return TCPRQuestion(
            question=confirmation_questions[self.phase][ctx.deps.locale],
            phase=self.phase,
            step_index=self.step_index,
        )


@dataclass
class TCPRQuestion(QuestionNode):
    question: str
    phase: InstructionPhase
    step_index: int = 0

    @override
    async def run(
        self,
        ctx: GraphRunContext[GraphState, Settings],
    ) -> ProcessCallerResponse:
        raise RuntimeError(
            "TCPRQuestion node should not be executed directly when using persistence."
        )


@dataclass
class ProcessCallerResponse(EmergencyNode):
    question: str
    response: str
    phase: InstructionPhase
    step_index: int = 0

    @override
    async def run(
        self,
        ctx: GraphRunContext[GraphState, Settings],
    ) -> (
        TCPRQuestion | Instruct | ArtificialVentilation | ChestCompression | EMSArrived
    ):
        ctx.deps.telemetry.messages_logger.info(
            "", extra={"speaker": "operator", "msg_text": self.question}
        )
        ctx.deps.telemetry.messages_logger.info(
            "", extra={"speaker": "caller", "msg_text": self.response}
        )

        parse_result = await var_fill_task(
            self.question,
            self.response,
            ctx.state.call_state,
            ctx.deps,
        )

        if parse_result is CallerFeedback.EMS_ARRIVED:
            ctx.state.call_state.ems_arrived = True
            return EMSArrived()
        if parse_result is CallerFeedback.CONFIRMED:
            if self.phase == "instruct":
                return Instruct(step_index=self.step_index + 1)
            if self.phase == "ventilation":
                return ChestCompression()
            return ArtificialVentilation()

        if isinstance(parse_result, str):
            return TCPRQuestion(
                question=parse_result,
                phase=self.phase,
                step_index=self.step_index,
            )

        raise TypeError(
            f"Unsupported TCPR parse result type: {type(parse_result)!r}"
        )


@dataclass
class EMSArrived(MessageNode):
    messages: ClassVar[dict[Locale, str]] = {
        Locale.EN: "Please hand over to the paramedics",
        Locale.DE: "Bitte uebergeben Sie an die Einsatzkraefte",
    }

    @override
    async def run(
        self,
        ctx: GraphRunContext[GraphState, Settings],
    ) -> End[EmergencyCall]:
        message = self.messages[ctx.deps.locale]
        ctx.deps.telemetry.messages_logger.info(
            "", extra={"speaker": "operator", "msg_text": message}
        )
        return End(ctx.state.call_state)


@overload
async def run_graph(
    *,
    deps: Settings = Settings(name="tcpr"),
    answer: str | None = None,
    init_state: GraphState | EmergencyCall,
) -> RunTCPRNode: ...


@overload
async def run_graph(
    *,
    deps: Settings = Settings(name="tcpr"),
    answer: str | None = None,
    init_state: None = None,
) -> RunTCPRNode | None: ...


async def run_graph(
    *,
    deps: Settings = Settings(name="tcpr"),
    answer: str | None = None,
    init_state: GraphState | EmergencyCall | None = None,
) -> RunTCPRNode | None:
    graph = Graph[GraphState, Settings, EmergencyCall](
        nodes=[
            Instruct,
            ArtificialVentilation,
            ChestCompression,
            TCPRMessage,
            TCPRQuestion,
            ProcessCallerResponse,
            EMSArrived,
        ],
    )
    persistence = await setup_resumable_file_persistence(
        graph, deps.storage.save_path, prefix="tcpr_"
    )

    if isinstance(init_state, GraphState):
        graph_state = init_state
    elif init_state is None:
        graph_state = None
    else:
        graph_state = GraphState(call_state=init_state)
    restored = await restore_graph(
        graph,
        persistence,
        init_node=Instruct(step_index=0) if graph_state is not None else None,
        init_state=graph_state,
    )
    if restored is None:
        return None

    node, _ = restored
    if answer is None and isinstance(node, TCPRQuestion):
        deps.telemetry.logger.debug(str(node))
        return node
    if answer is not None and isinstance(node, TCPRQuestion):
        node = ProcessCallerResponse(
            question=node.question,
            response=answer,
            phase=node.phase,
            step_index=node.step_index,
        )

    async with graph.iter_from_persistence(persistence=persistence, deps=deps) as run:
        while not isinstance(node := await run.next(node), End):
            deps.telemetry.logger.debug(f"[TCPR Node] {node.get_node_id()}")
            if isinstance(node, QuestionNode):
                deps.telemetry.logger.debug(str(node))
                return node
            if isinstance(node, MessageNode):
                return node

    flush_logger(deps.telemetry.messages_logger)
    flush_logger(deps.telemetry.state_logger)

    result = run.result
    assert result is not None
    save_mermaid_graph(graph, deps.storage.save_path, stem="tcpr_graph")
    record_completion_artifacts(
        deps,
        result.state.call_state,
        list(result.state.message_history),
    )

    return node


async def main(
    deps: Settings | None = None,
    state: GraphState | EmergencyCall | None = None,
):
    """Run the TCPR graph in isolation."""
    deps = deps or Settings(name="tcp_subgraph_only")
    result = await run_graph(
        deps=deps,
        init_state=state or EmergencyCall(cardiac_arrest=True),
    )

    while not isinstance(result, End):
        if isinstance(result, MessageNode):
            print(result.messages[deps.locale])
            result = await run_graph(deps=deps)
            continue
        if isinstance(result, QuestionNode):
            answer = await prompt_user(result.question, deps=deps)
            result = await run_graph(deps=deps, answer=answer)
            continue
        raise TypeError(f"Unsupported TCPR result type: {type(result)!r}")

    if isinstance(result, End):
        print("Graph finished with state:", result.data)


if __name__ == "__main__":
    asyncio.run(main())
