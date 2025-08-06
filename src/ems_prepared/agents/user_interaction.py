"""Input mode management for emergency call workflows.

This module provides:
- InputMode: enumeration defining CLI and REQUEST input modes
- ask_user: utility function for prompting users based on input mode
"""

from collections.abc import Awaitable, Callable
from enum import Enum, auto

from pydantic.types import T
from pydantic_graph import GraphRunContext
from rich.pretty import pprint
from rich.prompt import Prompt

from ems_prepared.settings import Settings
from ems_prepared.state_model.emergency_call_state import EmergencyCall


class InputMode(Enum):
    """Enumeration for input modes in the emergency call workflow.

    Defines whether input is collected via CLI or as a request.
    """

    CLI = auto()
    REQUEST = auto()
    TEST = auto()


async def prompt_user(question: str, mode: InputMode = InputMode.CLI) -> str | None:
    """Prompt the user for input in the given mode."""
    match mode:
        case InputMode.CLI:
            # return input(question)
            return Prompt.ask(question)
        case InputMode.REQUEST:
            pass  # TODO write fastapi server and send request from here to the client (ask prakash about bi-directional requests)
        case InputMode.TEST:
            pass  # handle differently if it makes sense
    return None


def tell_user(question: str, mode: InputMode = InputMode.CLI) -> None:
    """Tell the user in the given mode."""
    match mode:
        case InputMode.CLI:
            # return input(question)
            pprint(question)
        case InputMode.REQUEST:
            pass  # TODO write fastapi server and send request from here to the client (ask prakash about bi-directional requests)
        case InputMode.TEST:
            pass  # handle differently if it makes sense


async def converse_with_user(
    prompt: str,
    ctx: GraphRunContext[EmergencyCall, Settings],
    task_function: Callable[
        [str, str, GraphRunContext[EmergencyCall, Settings]], Awaitable[T]
    ],
) -> T:
    user_response: str | None = await prompt_user(prompt)
    if user_response is None:
        raise
    parse_result = await task_function(
        prompt,
        user_response,
        ctx,
    )
    return parse_result
