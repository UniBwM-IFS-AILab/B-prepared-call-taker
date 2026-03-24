"""Tests for transport-neutral user interaction helpers."""

from __future__ import annotations

import pytest

from ems_prepared.model.context import InputMode, Settings
from ems_prepared.util.user_interaction import prompt_user


@pytest.mark.asyncio
async def test_prompt_user_api_requires_request_input(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """API mode should fail when no input transport callback is configured."""
    monkeypatch.chdir(tmp_path)
    deps = Settings(name="test_api_mode", call_origin=InputMode.API)

    with pytest.raises(RuntimeError, match="request_input"):
        _ = await prompt_user("Where are you?", deps)


@pytest.mark.asyncio
async def test_prompt_user_api_uses_request_input_callback(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """API mode should delegate inbound text to configured request_input callback."""
    monkeypatch.chdir(tmp_path)
    prompts: list[str] = []

    async def fake_request_input(_deps: Settings, prompt: str) -> str:
        prompts.append(prompt)
        return "At Main Street"

    deps = Settings(
        name="test_api_mode_callback",
        call_origin=InputMode.API,
        request_input=fake_request_input,
    )

    answer = await prompt_user("Where are you?", deps)

    assert answer == "At Main Street"
    assert prompts == ["Where are you?"]
