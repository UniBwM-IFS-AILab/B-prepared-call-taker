"""Defines the graph for the emergency call workflow."""

from __future__ import annotations

import inspect
import logging
from typing import Annotated, Any, override

# logger = logging.getLogger(__name__)
from loguru import logger
from parso.tree import BaseNode
from pydantic import BaseModel
from pydantic.dataclasses import dataclass
from pydantic_ai._run_context import AgentDepsT
from pydantic_graph.nodes import Edge, End, GraphRunContext

from ems_prepared.agents.state_fill_agent import (
    emergency_type_agent,
    enough_info_agent,
    response_cleanup,
    state_fill_agent,
    state_fill_task,
)
from ems_prepared.dialogue_state.emergency_call_state import EmergencyCall
from ems_prepared.dialogue_state.meta_state import GraphState
from ems_prepared.dialogue_state.type_defs import DispoType, EmergencyType, Unknown
from ems_prepared.policies.pydantic_graph.type_defs import EmergencyNode
from ems_prepared.util.custom_deepmerge import ignore_empty_merger
from ems_prepared.util.helpers import async_wrapper
from ems_prepared.util.settings import Locale, Settings

state_print_filter = {"patient_symptoms"}


class MessageNode(EmergencyNode):
    """Base class for nodes that emit informative messages to the caller.

    Subclasses MUST provide a `messages` mapping keyed by `Locale` (type: dict[Locale, str]).
    Declaring the attribute here improves static typing and makes intent explicit.
    """

    messages: dict[Locale, str]


class QuestionNode(EmergencyNode):
    """Base class for nodes that ask a question and expect a response.

    Subclasses MUST provide a `question` attribute (type: str).
    """

    question: str


@dataclass
class Start(EmergencyNode):
    """Node representing the start of the Graph."""

    @override
    async def run(
        self,
        ctx: GraphRunContext[GraphState, Settings],
    ) -> Greeting:
        return Greeting()


@dataclass
class Greeting(MessageNode):
    """Node representing the greeting step in the emergency call workflow.

    Initiates the conversation and greets the caller.
    """

    messages = {
        Locale.EN: "Hello! You're speaking with the emergency call services.",
        Locale.DE: "Hallo! Hier ist der Notruf für Feuerwehr und Rettungsdienst.",
    }

    @override
    async def run(
        self,
        ctx: GraphRunContext[GraphState, Settings],
    ) -> ChooseQuestion:
        """Greet the user."""

        message = self.messages[ctx.deps.locale]
        # asyncio.create_task(tell_user(self.messages[ctx.deps.locale], ctx.deps))
        await async_wrapper(ctx.deps.emit(message))
        ctx.deps.messages_logger.info(
            "", extra={"speaker": "operator", "msg_text": message}
        )
        ctx.state.call_state.emergency_type = EmergencyType.INTRO

        return ChooseQuestion()


## TODO: Doesn't work due to return of self.next_node, this could instead work with V2 Graph API
# @dataclass
# class TellCaller(EmergencyNode):
#     message: str
#     next_node: BaseNode

#     @override
#     async def run(
#         self,
#         ctx: GraphRunContext[GraphState, Settings],
#     ) -> BaseNode:
#         return self.next_node


@dataclass
class ChooseQuestion(EmergencyNode):
    @override
    async def run(
        self, ctx: GraphRunContext[GraphState, Settings]
    ) -> (
        Annotated[ChooseSubGraph, Edge(label="No more Questions")]
        | Annotated[AskCaller, Edge(label="Next Question")]
    ):
        try:
            assert ctx.state.call_state.emergency_type is not None
            current_question_set: list[str] = ctx.state.questions.questions[
                ctx.deps.locale
            ][ctx.state.call_state.emergency_type]

            # next_question: str = list(
            #     reversed(ctx.state.questions[ctx.deps.locale][ctx.state.phase])
            # ).pop()

            next_question, remaining_questions = (
                current_question_set[0],
                current_question_set[1:],
            )
            ctx.state.questions.questions[ctx.deps.locale][
                ctx.state.call_state.emergency_type
            ] = remaining_questions

            return AskCaller(question=next_question)
        except IndexError:
            logger.debug("No more questions in current set")
            return ChooseSubGraph()


@dataclass
@dataclass
class AskCaller(QuestionNode):
    """Generic node for asking questions in the emergency call workflow."""

    question: str

    @override
    async def run(
        self,
        ctx: GraphRunContext[GraphState, Settings],
    ) -> Annotated[ExtractState, Edge(label="Try to extract state")]:
        """Ask the user. THIS NODE WILL NEVER BE EXECTUED WHEN USING PERSISTENCE."""

        # response: str = await prompt_user(self.question, deps=ctx.deps)
        ctx.deps.logger.debug("ran node AskCaller")
        raise Exception(
            "AskCaller node should not be executed directly when using persistence."
        )

        return ExtractState(question=self.question, response="")
        # return ExtractState(question=self.question, response=response)


@dataclass
class ExtractState(EmergencyNode):
    """Node for extracting state from user responses."""

    question: str
    response: str

    @override
    async def run(
        self, ctx: GraphRunContext[GraphState, Settings]
    ) -> Annotated[EvaluateAgentOutput, Edge(label="Evaluate extraction result")]:
        """Extract state from the current context."""

        # Log operator question and caller response
        ctx.deps.messages_logger.info(
            "", extra={"speaker": "operator", "msg_text": self.question}
        )
        ctx.deps.messages_logger.info(
            "", extra={"speaker": "caller", "msg_text": self.response}
        )

        parse_result: EmergencyCall | str = await state_fill_task(
            self.question, self.response, ctx
        )

        # Log the extraction result as JSON
        result_data = (
            str(parse_result)
            if not isinstance(parse_result, BaseModel)
            else parse_result.model_dump(
                exclude_none=True, exclude={"questions", "current_position"}
            )
        )
        ctx.deps.state_logger.info(
            {
                "question": self.question,
                "response": self.response,
                "result": result_data,
            },
            extra={"event": "extraction"},
        )

        return EvaluateAgentOutput(run_result=parse_result)


@dataclass
class EvaluateAgentOutput(EmergencyNode):
    run_result: EmergencyCall | str

    @override
    async def run(
        self, ctx: GraphRunContext[GraphState, Settings]
    ) -> (
        Annotated[AskCaller, Edge(label="Agent asks clarifying question")]
        | Annotated[MergeState, Edge(label="New State detected")]
        # | Annotated[Disposition, Edge(label="Trigger w/ RD1, no RD2 symptoms detected")]
    ):
        if isinstance(self.run_result, EmergencyCall):
            logger.info("Model returned new data")

            return MergeState(self.run_result)

        # elif ctx.state.call_state.rd1:
        # elif ctx.state.call_state.no_more_questions_needed:
        #     # FIXME: this prevents multi-turn clarifying questions
        #     logger.info("Rd1 is already true and we could not extract extra info")
        #     return Disposition(DispoType.RD1)

        # elif isinstance(self.run_result, str):
        else:
            ctx.deps.logger.info(
                "No new Data extracted, asking clarifying question (probably)"
            )
            return AskCaller(question=self.run_result)


@dataclass
class MergeState(EmergencyNode):
    new_state: EmergencyCall

    @override
    async def run(
        self, ctx: GraphRunContext[GraphState, Settings]
    ) -> (
        Annotated[EvaluateState, Edge(label="New State merged")]
        | Annotated[Disposition, Edge(label="Accept RD1 as final")]
    ):
        ctx.deps.logger.debug(
            f"Old State:\t{ctx.state.call_state.model_dump(exclude_none=True)}"
        )
        ctx.deps.logger.debug(
            f"New State:\t{self.new_state.model_dump(exclude_none=True)}"
        )

        ignore_empty_merger.merge(
            ctx.state.call_state.__dict__, self.new_state.__dict__
        )
        ctx.deps.logger.debug(
            f"Merged State:\t{ctx.state.call_state.model_dump(exclude_none=True)}"
        )

        # Log the merged state as JSON
        state_data = ctx.state.model_dump(exclude_none=True, exclude={"questions"})
        ctx.deps.state_logger.info(
            {"state": state_data},
            extra={"event": "state_merged"},
        )

        # if (
        #     rd1_already_done
        # ):  # NOTE: we could not extract new state (RD2) and already have an outcome
        #     logger.info("RD1 accepted as final state.")
        #     return Disposition(DispoType.RD1)

        return EvaluateState()


@dataclass
class EvaluateState(EmergencyNode):
    @override
    async def run(
        self, ctx: GraphRunContext[GraphState, Settings]
    ) -> (
        Annotated[ChooseQuestion, Edge(label="No Outcome yet")] | RD2 | RD1 | TCPR
        # | Disposition
        # | Annotated[HighUrgency, Edge(label="Immediate Disposition")]
    ):
        ctx.deps.logger.debug(
            f"Evaluate State:\t{ctx.state.call_state.model_dump(exclude_none=True)}"
        )

        if ctx.state.call_state.cpr_needed:
            # needs to take precedence over RD2 because cpr_needed implies RD2
            return TCPR()
        # elif ctx.state.call_state.time_critical:
        #     return HighUrgency()
        elif ctx.state.call_state.rd2:
            return RD2()
        elif ctx.state.call_state.rd1:
            return RD1()

        else:
            return ChooseQuestion()


@dataclass
class ChooseSubGraph(EmergencyNode):
    @override
    async def run(
        self,
        ctx: GraphRunContext[GraphState, Settings],
    ) -> (
        Annotated[ChooseQuestion, Edge(label="New set of questions")]
        | Annotated[
            EvaluateAgentOutput, Edge(label="Force-update emergency type in state")
        ]
    ):
        logger.info("Forcing Agent to decide the Emergency type...")

        result = await emergency_type_agent.run(  # type: ignore
            user_prompt=(
                "Fill out the Emergency Type based on the current state"
                f"State: {ctx.state}"
            ),
            deps=ctx.deps,  # type: ignore
            message_history=ctx.state.message_history,
        )
        ctx.deps.logger.debug(f"Result type: {type(result.output)}")
        ctx.deps.logger.debug(f"Emergency type: {result.output}")

        return EvaluateAgentOutput(EmergencyCall(emergency_type=result.output))


@dataclass
class RD1(EmergencyNode):
    """ """

    questions = {
        Locale.DE: "Können Sie die Symptome genauer beschreiben?",
        Locale.EN: "Can you describe the symptoms with more detail?",
    }

    @override
    async def run(
        self,
        ctx: GraphRunContext[GraphState, Settings],
    ) -> (
        Annotated[RD2, Edge(label="Increase to RD2")]
        | Annotated[Disposition, Edge(label="Use RD1")]
        | Annotated[AskCaller, Edge(label="Check for RD2")]
        | Annotated[EvaluateAgentOutput, Edge(label="Evaluate agent output")]
    ):
        if ctx.state.call_state.rd1 is not True:
            raise
        else:
            ctx.deps.logger.info("Reached RD1")

        # Ask at least one time for RD2
        if ctx.state.call_state.enough_information_gathered is None:
            ctx.state.call_state.enough_information_gathered = False
            logger.info("Asking for RD2")
            return AskCaller(self.questions[ctx.deps.locale])

        # if RD2 is unkown, decide if more question or go to RD1 if already done
        if ctx.state.call_state.rd2 is None:
            if ctx.state.call_state.enough_information_gathered:
                logger.info("No more questions needed, accepting RD1")
                return Disposition(DispoType.RD1)

            elif not ctx.state.call_state.enough_information_gathered:
                logger.info("Forcing Agent to decide if enough information gathered...")

                result = await enough_info_agent.run(  # type: ignore
                    user_prompt=(
                        "Decide if enough information has been gathered based on the current state"
                        f"State: {ctx.state}"
                    ),
                    deps=ctx.deps,  # type: ignore
                    message_history=ctx.state.message_history,
                )
                logger.debug(f"Enough info result: {result.output}")

                return EvaluateAgentOutput(
                    EmergencyCall(enough_information_gathered=result.output)
                )

        # self.visited = True
        return RD2() if ctx.state.call_state.rd2 else Disposition(DispoType.RD1)


@dataclass
class RD2(EmergencyNode):
    """ """

    @override
    async def run(
        self,
        ctx: GraphRunContext[GraphState, Settings],
    ) -> Annotated[Disposition, Edge(label="Use RD2")]:
        if not ctx.state.call_state.rd2:
            raise

        return Disposition(DispoType.RD2)


@dataclass
class TCPR(EmergencyNode):
    """ """

    @override
    async def run(
        self,
        ctx: GraphRunContext[GraphState, Settings],
    ) -> Disposition | Annotated[End[EmergencyCall], Edge(label="EMS arrived")]:
        if not ctx.state.call_state.cpr_needed:
            raise Exception("TCPR node reached without cpr_needed being true")
        ctx.state.call_state.emergency_type = EmergencyType.MEDICAL

        # Log the state update after changing emergency_type
        state_data = ctx.state.model_dump(exclude_none=True, exclude={"questions"})
        ctx.deps.state_logger.info(
            {"state": state_data},
            extra={"event": "emergency_type_set"},
        )

        if ctx.state.call_state.ems_arrived:
            return End(ctx.state.call_state)
        return Disposition(DispoType.RD2)


@dataclass
class HighUrgency(EmergencyNode):
    @override
    async def run(
        self,
        ctx: GraphRunContext[GraphState, Settings],
    ) -> Disposition:
        # TODO: raise if expected state is not true
        return Disposition()


@dataclass
class Disposition(MessageNode):
    """_summary_

    Args:
        BaseNode (_type_): _description_
    """

    disposition_kind: DispoType | None = None
    messages = {
        Locale.EN: "A Vehicle is on it's way to you. Please stand by.",
        Locale.DE: "TODO: Dispo message",
    }

    @override
    async def run(
        self,
        ctx: GraphRunContext[GraphState, Settings],
    ) -> End[EmergencyCall]:
        # TODO: ask for number of persons
        ctx.deps.logger.info(
            f"final state: {ctx.state.call_state.model_dump(exclude_none=True, exclude={'questions'})}"
        )

        message = self.messages[ctx.deps.locale]
        await async_wrapper(ctx.deps.emit(message))
        ctx.deps.messages_logger.info(
            "", extra={"speaker": "operator", "msg_text": message}
        )

        return End[EmergencyCall](ctx.state.call_state)
