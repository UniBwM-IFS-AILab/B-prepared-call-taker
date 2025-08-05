"""Defines the graph for the emergency call workflow."""

from typing import Annotated, override

from pydantic import BaseModel
from pydantic.dataclasses import dataclass
from pydantic_graph.graph import GraphRunResult
from pydantic_graph.nodes import Edge, End, GraphRunContext
from rich import print

from ems_prepared.agents.state_fill_agent import (
    response_cleanup,
    state_fill_agent,
    state_fill_task,
)
from ems_prepared.agents.user_interaction import (
    converse_with_user,
    tell_user,
)
from ems_prepared.graphs import tcpr_subgraph
from ems_prepared.graphs.type_defs import EmergencyNode
from ems_prepared.settings import LOCALE, Settings
from ems_prepared.state_model.custom_deepmerge import ignore_empty_merger
from ems_prepared.state_model.emergency_call_state import EmergencyCall
from ems_prepared.state_model.type_defs import DispoType, EmergencyType, Unknown
from ems_prepared.steps_data.iterator import QuestionIterator

# logger.add(
#     sys.stdout, colorize=True, format="<green>{time}</green> <level>{message}</level>"
# )

# TODO: replace me wtih proper localisation


emergency_questions = QuestionIterator(LOCALE, "Intro")
state_print_filter = {"patient_symptoms"}


@dataclass
class Greeting(EmergencyNode):
    """Node representing the greeting step in the emergency call workflow.

    Initiates the conversation and greets the caller.
    """

    greeting: str = {
        "en": "You're speaking with the emergency call services.",
        "de": "Hier ist der Notruf für Feuerwehr und Rettungsdienst.",
    }[LOCALE]

    @override
    async def run(
        self,
        ctx: GraphRunContext[EmergencyCall, Settings],
    ) -> "AskCaller":
        """Greet the user."""
        _ = tell_user(self.greeting)
        return AskCaller(question=next(emergency_questions))


@dataclass
class AskCaller(EmergencyNode):
    """Generic node for asking questions in the emergency call workflow."""

    question: str

    @override
    async def run(
        self,
        ctx: GraphRunContext[EmergencyCall, Settings],
    ) -> "EvaluateAgentOutput":
        """Ask the user."""

        parse_result = await converse_with_user(self.question, ctx, state_fill_task)

        return EvaluateAgentOutput(run_result=parse_result)


@dataclass
class EvaluateAgentOutput(EmergencyNode):
    run_result: BaseModel | str

    @override
    async def run(
        self, ctx: GraphRunContext[EmergencyCall, Settings]
    ) -> "Annotated[AskCaller, Edge(label='Agent asks clarifying question')] | Annotated[EvaluateState, Edge(label='New State detected')] | Annotated[Disposition, Edge(label='Trigger w/ RD1, no RD2 symptoms detected')]":
        if isinstance(self.run_result, BaseModel):
            print("Model returned new data")
            _ = ignore_empty_merger.merge(ctx.state.__dict__, self.run_result.__dict__)
            return EvaluateState()

        # elif isinstance(self.run_result, str):
        else:
            if ctx.state.rd1:
                # NOTE: we could not extract new state (RD2) and already have an outcome
                return Disposition(DispoType.RD1)

            print("Model could not extract new data, likely asking for more info")
            return AskCaller(question=self.run_result)


@dataclass
class EvaluateState(EmergencyNode):
    @override
    async def run(
        self, ctx: GraphRunContext[EmergencyCall, Settings]
    ) -> "Annotated[ChooseQuestion, Edge(label='No Outcome yet')] | RD2 | RD1 | TCPR | Annotated[HighUrgency, Edge(label='Immediate Disposition')]":
        # -> "Annotated[ChooseSubGraph, Edge(label='No more Questions')] | Annotated[AskCaller, Edge(label='Next Question')] | RD2 | RD1 | TCPR | Annotated[HighUrgency, Edge(label='Immediate Disposition')]":
        print(f"Current State: {ctx.state.model_dump(exclude_none=True)}")

        if ctx.state.cpr_needed:
            # needs to take precedence over RD2 because cpr_needed implies RD2
            return TCPR()
        elif ctx.state.urgency_needed:
            return HighUrgency()
        elif ctx.state.rd2:
            return RD2()
        elif ctx.state.rd1:
            return RD1()
        else:
            return ChooseQuestion()


@dataclass
class ChooseQuestion(EmergencyNode):
    @override
    async def run(
        self, ctx: GraphRunContext[EmergencyCall, Settings]
    ) -> "Annotated[ChooseSubGraph, Edge(label='No more Questions')] | Annotated[AskCaller, Edge(label='Next Question')]":
        #
        try:
            next_question: str = next(emergency_questions)
            return AskCaller(question=next_question)
        except StopIteration:
            print("choosing new subgraph")
            return ChooseSubGraph()


@dataclass
class ChooseSubGraph(EmergencyNode):
    @override
    async def run(
        self,
        ctx: GraphRunContext[EmergencyCall, Settings],
    ) -> "Annotated[AskCaller, Edge(label='New set of questions')] | Annotated[EvaluateAgentOutput, Edge(label='Force-update emergency type in state')] ":
        # NOTE: We do not really need a subgraph (for now),
        # we only need to switch the set of questions to go through
        match ctx.state.emergency_type:
            case EmergencyType.MEDICAL:
                # sub_graph_result: GraphRunResult[EmergencyCall] = await run_graph(
                #     graph_name="medical_sub_graph"
                # )
                #  medical_subgraph_finished: bool = true
                # _ = ignore_empty_merger.merge(
                #     ctx.state.__dict__, sub_graph_result.__dict__
                # )

                global emergency_questions
                emergency_questions = QuestionIterator(LOCALE, "Key Questions")

                global state_print_filter
                state_print_filter = {
                    "caller_name",
                    "caller_phone",
                    "location",
                    # "emergency_type",
                    # "situation_description",
                }

                return AskCaller(next(emergency_questions))
            case EmergencyType.FIRE:
                pass
            case EmergencyType.FIRE_MEDICAL:
                pass
            case EmergencyType.NON_EMERGENCY:
                pass
            case _:
                # TODO: deuplicate this code, use one agents for stae fill and one for conversation to make this easier
                print("Forcing Agent to decide the Emergency type...")
                result = await state_fill_agent.run(
                    user_prompt=(
                        "Fill out the Emergency Type based on the current state"
                        f"State: {ctx.state}"
                    ),
                    deps=ctx.deps,
                )
                print(result.output)
                print(type(result.output))
                cleaned = response_cleanup(result.output)

                if isinstance(cleaned, EmergencyCall):
                    if cleaned.emergency_type is None:
                        raise
                    print(
                        f"Detected Type: {cleaned.model_dump(include={'emergency_type'})}"
                    )
                    # _ = ignore_empty_merger.merge(ctx.state.__dict__, cleaned.__dict__) # seems duplicated / unnecessary

                return EvaluateAgentOutput(cleaned)

        raise


@dataclass
class RD1(EmergencyNode):
    """ """

    question: str = (
        "Können Sie die Symptome genauer beschreiben?"
        if LOCALE == "de"
        else "Can you describe the symptoms with more detail?"
    )
    # visited: bool = False

    @override
    async def run(
        self,
        ctx: GraphRunContext[EmergencyCall, Settings],
    ) -> "RD2 | Annotated[Disposition, Edge(label='Use RD1')]|AskCaller":
        if ctx.state.rd1 is not True:
            raise
        else:
            print("Reached RD1")
            # print(f"already visited: {self.visited}")

        if ctx.state.rd2 is Unknown:  # and not self.visited:
            # NOTE: this is a hack!
            # since computed fields shouldn't be overwritten normally, we introduce this proxy
            # ctx.state.hidden_rd2 = False

            print("Asking for RD2")
            # self.visited = True
            return AskCaller(self.question)
        # self.visited = True
        return RD2() if ctx.state.rd2 else Disposition(DispoType.RD1)


@dataclass
class RD2(EmergencyNode):
    """ """

    @override
    async def run(
        self,
        ctx: GraphRunContext[EmergencyCall, Settings],
    ) -> "Annotated[Disposition, Edge(label='Use RD2')]":
        if not ctx.state.rd2:
            raise
        print("Reached RD2")

        return Disposition(DispoType.RD2)


@dataclass
class TCPR(EmergencyNode):
    """ """

    @override
    async def run(
        self,
        ctx: GraphRunContext[EmergencyCall, Settings],
    ) -> "Disposition | Annotated[End[EmergencyCall], Edge(label='EMS arrived')]":
        if not any([ctx.state.agonal_breathing, ctx.state.cardiac_arrest]):
            raise
        print("Patient needs T-CPR")

        result: (
            GraphRunResult[EmergencyCall, EmergencyCall] | None
        ) = await tcpr_subgraph.run_graph(ctx.state, ctx.deps)
        if result is not None:
            _ = ignore_empty_merger.merge(ctx.state.__dict__, result.state.__dict__)
        else:
            raise TypeError

        if ctx.state.ems_arrived:
            return End(ctx.state)
        return Disposition(DispoType.RD2)


@dataclass
class HighUrgency(EmergencyNode):
    @override
    async def run(
        self,
        ctx: GraphRunContext[EmergencyCall, Settings],
    ) -> "Disposition":
        # TODO: raise if expected state is not true
        return Disposition()


@dataclass
class Disposition(EmergencyNode):
    """_summary_

    Args:
        BaseNode (_type_): _description_
    """

    disposition_kind: DispoType | None = None

    @override
    async def run(
        self,
        ctx: GraphRunContext[EmergencyCall, Settings],
    ) -> "End[EmergencyCall]":  # AskCaller |
        # TODO: fix this, how to run this only once
        # if ctx.state.emergency_type is EmergencyType.MEDICAL:
        #     return AskCaller(f"Wie viele Menschen sind an dem {'???'} beteiligt?")
        # else:
        print("Starting Disposition")
        print(f"final state: {ctx.state.model_dump(exclude_none=True)}")

        return End[EmergencyCall](ctx.state)


## TODO Implement me
@dataclass
class RunSubGraph(EmergencyNode):
    pass


# @dataclass  # TODO: finish
# class EvaluateMedicalAnswer(EvaluateState):
#     prev_question: str

#     @override
#     async def run(self, ctx: GraphRunContext[EmergencyCall, Settings]) -> "AskCaller | RD2 | RD1":
#         if ctx.state.rd2:
#             return RD2()
#         elif ctx.state.rd1:
#             return RD1()
#         next_question: str = super(EvaluateMedicalAnswer, self).run(ctx)
#         return AskCaller(question=next_question)
