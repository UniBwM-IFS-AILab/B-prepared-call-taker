"""Input mode management for emergency call workflows.

This module provides:
- InputMode: enumeration defining CLI and REQUEST input modes
- ask_user: utility function for prompting users based on input mode
"""

from collections.abc import Awaitable, Callable
from typing import TypeVar

from pydantic.types import T
from pydantic_graph import GraphRunContext
from rich.pretty import pprint
from rich.prompt import Prompt

from ems_prepared.dialogue_state.emergency_call_state import EmergencyCall
from ems_prepared.util.settings import InputMode, Settings

T = TypeVar("T")


async def prompt_user(question: str, deps: Settings) -> str | None:
    """Prompt the user for input in the given mode."""
    match deps.call_origin:
        case InputMode.CLI:
            # return input(question)
            return Prompt.ask(question)  # blocking read
        case InputMode.API:
            pprint(question)
            return Prompt.ask(question)
        case InputMode.REQUEST:
            pass  # TODO write fastapi server and send request from here to the client (ask prakash about bi-directional requests)

        case InputMode.TEST:
            pass
    return None


async def tell_user(message: str, deps: Settings) -> None:
    """Tell the user in the given mode."""
    match deps.call_origin:
        case InputMode.CLI:
            pprint(message)
        case InputMode.API:
            pprint(message)
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
    user_response: str | None = await prompt_user(prompt, deps=ctx.deps)
    if user_response is None:
        raise
    parse_result = await task_function(
        prompt,
        user_response,
        ctx,
    )
    return parse_result
