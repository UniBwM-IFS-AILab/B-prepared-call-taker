"""Shared chat streaming helpers for the Gradio adapter."""

from __future__ import annotations

import asyncio
import logging

from gradio import ChatMessage

from ems_prepared.adapters.gradio.scenarios import COMPLETION_MESSAGE
from ems_prepared.model.contracts import BackendEvent, BackendEventKind

logger = logging.getLogger(__name__)


async def stream_message_to_history(history: list[ChatMessage], message: ChatMessage):
    """Stream one assistant message character-by-character into history."""
    content = message.content if isinstance(message.content, str) else ""
    current_message = ChatMessage(
        role=message.role,
        content="",
        metadata=message.metadata,
    )
    history.append(current_message)

    for char in content:
        if isinstance(current_message.content, str):
            current_message.content += char
        yield history
        await asyncio.sleep(0.01)


async def stream_backend_events(history: list[ChatMessage], events: list[BackendEvent]):
    """Stream backend events into chat history."""
    if not events:
        yield history
        return

    for event in events:
        if event.kind is BackendEventKind.COMPLETED:
            async for updated_history in stream_message_to_history(
                history,
                COMPLETION_MESSAGE,
            ):
                yield updated_history
            continue

        if event.kind in {BackendEventKind.MESSAGE, BackendEventKind.QUESTION}:
            text = event.text or ""
        elif event.kind is BackendEventKind.ERROR:
            text = event.text or "❌ Internal error. Please reset the session."
        else:
            text = event.text or ""

        if not text.strip():
            continue
        async for updated_history in stream_message_to_history(
            history,
            ChatMessage(role="assistant", content=text),
        ):
            yield updated_history


def events_include_completion(events: list[BackendEvent]) -> bool:
    """Return whether the backend emitted a completion event."""
    return any(event.kind is BackendEventKind.COMPLETED for event in events)
