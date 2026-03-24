"""Input mode management for emergency call workflows.

This module provides:
- InputMode: enumeration defining CLI and REQUEST input modes
- ask_user: utility function for prompting users based on input mode
"""

import warnings
from collections.abc import Awaitable, Callable
from typing import TypeVar

from pydantic_graph import GraphRunContext
from rich import print
from rich.prompt import Prompt

from ems_prepared.dialogue_state.emergency_call_state import EmergencyCall
from ems_prepared.model.context import InputMode, Settings
from ems_prepared.util.helpers import async_wrapper

T = TypeVar("T")


async def prompt_user(question: str, deps: Settings) -> str:
    """Prompt the user for input in the given mode."""
    match deps.call_origin:
        case InputMode.CLI:
            # return input(question)
            return Prompt.ask(question)  # blocking read
        case InputMode.API:
            print(question)
            request_input = deps.transport.request_input
            if request_input is None:
                raise RuntimeError("No API request_input transport configured.")
            return await request_input(deps, question)
        case InputMode.TEST:
            raise NotImplementedError("Test input mode not implemented yet.")


async def tell_user(message: str, deps: Settings) -> None:
    """Tell the user in the given mode."""
    warnings.warn(
        "tell_user() is deprecated; prefer deps.emit(...) for outbound messages.",
        DeprecationWarning,
        stacklevel=2,
    )
    match deps.call_origin:
        case InputMode.CLI:
            print(message)
        case InputMode.API:
            print(message)
            await async_wrapper(deps.transport.emit(deps, message))
        case InputMode.TEST:
            pass  # handle differently if it makes sense


async def converse_with_user(
    prompt: str,
    ctx: GraphRunContext[EmergencyCall, Settings],
    task_function: Callable[
        [str, str, GraphRunContext[EmergencyCall, Settings]], Awaitable[T]
    ],
) -> T:
    """Prompt for user input and pass response through the provided task function."""
    user_response: str | None = await prompt_user(prompt, deps=ctx.deps)
    if user_response is None:
        raise RuntimeError("prompt_user returned None; expected a string response.")
    parse_result = await task_function(
        prompt,
        user_response,
        ctx,
    )
    return parse_result
