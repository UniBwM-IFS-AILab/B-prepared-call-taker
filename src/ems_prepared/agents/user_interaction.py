"""Input mode management for emergency call workflows.

This module provides:
- InputMode: enumeration defining CLI and REQUEST input modes
- ask_user: utility function for prompting users based on input mode
"""

from enum import Enum, auto

from rich.pretty import pprint
from rich.prompt import Prompt


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


def tell_user(question: str, mode: InputMode = InputMode.CLI) -> str | None:
    """Tell the user in the given mode."""
    match mode:
        case InputMode.CLI:
            # return input(question)
            return pprint(question)
        case InputMode.REQUEST:
            pass  # TODO write fastapi server and send request from here to the client (ask prakash about bi-directional requests)
        case InputMode.TEST:
            pass  # handle differently if it makes sense
