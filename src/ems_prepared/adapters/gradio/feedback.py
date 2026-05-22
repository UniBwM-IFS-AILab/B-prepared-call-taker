"""Chat message feedback helpers for Gradio like/dislike events."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import gradio as gr

from ems_prepared.adapters.gradio.state import SessionRef
from ems_prepared.model.contracts import MessageFeedback, MessageRole, SessionManager

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeedbackSelection:
    """Resolved chat message target for feedback submission."""

    role: MessageRole
    content: str
    index: int | None


def _extract_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_fragments: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                text_fragments.append(item["text"])
            elif isinstance(item, str):
                text_fragments.append(item)
        return "\n".join(text_fragments)
    return str(content)


def _extract_from_history(
    *,
    history: list[Any],
    index: int | tuple[int, int] | None,
    fallback_value: Any,
) -> FeedbackSelection:
    if isinstance(index, tuple):
        target_index = int(index[0])
    elif isinstance(index, int):
        target_index = index
    else:
        target_index = None

    if target_index is not None and 0 <= target_index < len(history):
        message = history[target_index]
        if isinstance(message, gr.ChatMessage):
            return FeedbackSelection(
                role=MessageRole(message.role),
                content=_extract_text_from_content(message.content),
                index=target_index,
            )
        if isinstance(message, dict):
            raw_role = str(message.get("role", MessageRole.ASSISTANT))
            try:
                resolved_role = MessageRole(raw_role)
            except ValueError:
                resolved_role = MessageRole.ASSISTANT
            return FeedbackSelection(
                role=resolved_role,
                content=_extract_text_from_content(message.get("content")),
                index=target_index,
            )

    return FeedbackSelection(
        role=MessageRole.ASSISTANT,
        content=_extract_text_from_content(fallback_value),
        index=target_index,
    )


def _normalize_feedback_value(liked: bool | str) -> str:
    if liked is True:
        return "like"
    if liked is False:
        return "dislike"
    return str(liked)


async def persist_chat_feedback(
    history: list[Any],
    session: SessionRef | None,
    like_data: gr.LikeData,
    *,
    session_manager: SessionManager,
    skip_policy_calls: bool,
) -> None:
    """Persist one like/dislike event through the shared session manager."""
    if skip_policy_calls or session is None:
        return

    selection = _extract_from_history(
        history=history,
        index=like_data.index,
        fallback_value=like_data.value,
    )
    feedback = MessageFeedback(
        role=selection.role,
        content=selection.content,
        value=_normalize_feedback_value(like_data.liked),
        message_index=selection.index,
        metadata={
            "ui": "gradio",
            "source": "chatbot.like",
        },
    )
    try:
        await session_manager.submit_feedback(UUID(session.session_id), feedback)
    except LookupError:
        logger.warning(
            "Feedback dropped for expired session %s",
            session.session_id,
        )
