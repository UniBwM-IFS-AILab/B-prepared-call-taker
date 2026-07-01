from dataclasses import replace

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from ems_prepared.dialogue_state.structured_output import DialogueOutput


def remove_before_extracion_processor(
    messages: list[ModelMessage],
) -> list[ModelMessage]:
    fake_history: list[ModelMessage] = []
    system_prompt: SystemPromptPart | None = None

    for message in messages:
        clean_request: list[TextPart | SystemPromptPart | UserPromptPart] = []
        if isinstance(message, ModelRequest):
            for part in message.parts:
                match part:
                    case SystemPromptPart():  # need to retain system prompt
                        # clean_request.append(part)
                        system_prompt = part

                    case UserPromptPart():
                        clean_request.append(part)
                        pass
                    case RetryPromptPart():
                        pass
        if isinstance(message, ModelResponse):
            for part in message.parts:
                match part:
                    case ToolCallPart():  # state extraction return
                        pass
                    case TextPart():
                        clean_request.append(part)
        if clean_request:  # exclude empty messages
            fake_history.append(replace(message, parts=clean_request))

    start: ModelMessage = fake_history[0]
    assert isinstance(start, ModelRequest)
    if system_prompt is not None:
        # Ensure start.parts is a list before concatenation
        start.parts = [system_prompt] + list(start.parts)

    return fake_history


def keep_user_and_text_only(messages: list[ModelMessage]) -> list[ModelMessage]:
    """
    History processor that:
    - keeps only UserPromptPart in ModelRequest.parts
    - keeps only TextPart in ModelResponse.parts
    - discards all other messages and parts
    """
    filtered: list[ModelMessage] = []

    for msg in messages:
        if isinstance(msg, ModelRequest):
            # Keep only user prompts in the request
            new_parts = [p for p in msg.parts if isinstance(p, UserPromptPart)]
            if not new_parts:
                # If nothing left, drop this request entirely
                continue

            # dataclasses.replace preserves other fields (instructions, run_id, metadata, …)
            filtered.append(replace(msg, parts=new_parts))

        elif isinstance(msg, ModelResponse):
            # Keep only plain text parts in the response
            new_parts = [p for p in msg.parts if isinstance(p, TextPart)]
            if not new_parts:
                # If nothing left, drop this response entirely
                continue

            filtered.append(replace(msg, parts=new_parts))

        # Any other message type (if it ever appears) is discarded

    return filtered


def state_fill_history_processor(
    messages: list[ModelMessage],
) -> list[ModelMessage]:
    """
    Decide whether to include history, trim it accordingly, and ensure a system prompt.

    Pydantic AI calls this with:
      - the messages from `message_history` (if any),
      - plus the new ModelRequest for this run as the LAST element.
    """

    if not messages:
        return messages

    # The last message is the current request for this step.
    history_without_current = messages[:-1]

    # Find the most recent successful extraction (DialogueOutput.state is not None).
    last_success_idx: int | None = None
    for i in range(len(history_without_current) - 1, -1, -1):
        msg = history_without_current[i]
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if (
                    isinstance(part, ToolReturnPart)
                    and isinstance(part.content, DialogueOutput)
                    and getattr(part.content, "state", None) is not None
                ):
                    last_success_idx = i
                    break
        if last_success_idx is not None:
            break

    # Decide which slice of messages the model should see:
    #
    # - No successful extraction yet -> fresh run: only the current request.
    # - Last successful extraction is immediately before current request
    #   (i.e. the previous run just extracted state) -> also treat as fresh.
    # - Otherwise, keep only conversation AFTER the last extraction
    #   (follow-up questions / answers) plus the current request.
    if last_success_idx is None or last_success_idx == len(messages) - 2:
        processed: list[ModelMessage] = [messages[-1]]
    else:
        processed = messages[last_success_idx + 1 :]

    # Ensure at least one SystemPromptPart is present.
    # Pydantic ONLY auto-adds a system prompt if message_history is
    # empty or None; when we pass non-empty history, we must ensure
    # the system instructions are in the messages ourselves.
    has_system_prompt = any(
        isinstance(part, SystemPromptPart)
        for msg in processed
        if isinstance(msg, ModelRequest)
        for part in msg.parts
    )

    if not has_system_prompt:
        processed.insert(
            0,
            ModelRequest(
                parts=[SystemPromptPart(content=state_fill_prompt.full_prompt)]
            ),
        )

    return processed
