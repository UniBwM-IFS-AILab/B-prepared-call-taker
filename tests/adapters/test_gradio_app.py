"""Smoke tests for Gradio adapter wiring."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

import gradio as gr
import pytest

from ems_prepared.adapters.gradio.app import (
    build_demo,
    history_is_complete,
    stream_backend_events,
)
from ems_prepared.adapters.gradio.args import GradioAppArgs
from ems_prepared.adapters.gradio.core import COMPLETION_MESSAGE
from ems_prepared.adapters.gradio.guided import (
    build_initial_guided_state,
    current_context_markdown,
)
from ems_prepared.model.context import Locale
from ems_prepared.model.contracts import (
    BackendEvent,
    SessionHandle,
    SessionParameters,
    SessionState,
)


class _FakeSessionManager:
    """Minimal async session manager stub for UI build-time wiring tests."""

    async def start_session(
        self,
        request: SessionParameters,
    ) -> tuple[SessionHandle, list[BackendEvent]]:
        return (
            SessionHandle(
                user_id=request.user_id or uuid4(),
                session_id=request.session_id or uuid4(),
                locale=request.locale,
                frontend_name=request.frontend_name,
                policy_name=request.policy_name,
                scenario_name=request.scenario_name,
            ),
            [],
        )

    async def handle_input(self, session_id: UUID, text: str) -> list[BackendEvent]:
        _ = (session_id, text)
        return []

    async def resume_session(self, session_id: UUID) -> SessionState | None:
        _ = session_id
        return None

    async def end_session(self, session_id: UUID) -> bool:
        _ = session_id
        return True

    async def submit_survey(
        self,
        session_id: UUID,
        responses: list[dict[str, object]],
        feedback: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> Path:
        _ = (session_id, responses, feedback, metadata)
        return Path("survey.json")

    def get_view_state(self, session_id: UUID) -> SessionState | None:
        _ = session_id
        return None



def test_build_standard_demo_wires_events_inside_blocks_context(tmp_path: Path) -> None:
    """Standard `build_demo` should not raise context errors when wiring events."""
    (tmp_path / "Scenario_01.md").write_text("# Scenario 01\n", encoding="utf-8")

    args = GradioAppArgs(
        scenario_dir=str(tmp_path),
        user_id=0,
        policy="graph",
        locale=Locale.EN.value,
        debug=False,
        random_scenario=False,
    )
    demo = build_demo(session_manager=_FakeSessionManager(), args=args)

    assert isinstance(demo, gr.Blocks)
    assert demo.mode == "blocks"
    assert isinstance(args.resolve_policy(), str)
    assert args.is_debug_enabled() is False
    assert args.picker_interactive is True
    assert Locale.EN.value == "english"



def test_build_guided_demo_renders_one_outer_walkthrough_and_shared_context(tmp_path: Path) -> None:
    """Guided mode should render one outer walkthrough and one shared sidebar context."""
    scenario_names = ("Scenario_01.md", "Scenario_02.md", "Scenario_03.md")
    for scenario_name in scenario_names:
        (tmp_path / scenario_name).write_text(f"# {scenario_name}\n", encoding="utf-8")

    args = GradioAppArgs(
        scenario_dir=str(tmp_path),
        user_id=0,
        policy="graph",
        locale=Locale.EN.value,
        debug=False,
        random_scenario=False,
        ui="guided",
    )
    demo = build_demo(session_manager=_FakeSessionManager(), args=args)

    walkthrough_blocks = [
        block for block in demo.blocks.values() if isinstance(block, gr.Walkthrough)
    ]
    step_blocks = [block for block in demo.blocks.values() if isinstance(block, gr.Step)]
    scenario_context = current_context_markdown(
        build_initial_guided_state(scenario_names),
        args=args,
    )
    scenario_briefs = [
        block
        for block in demo.blocks.values()
        if isinstance(block, gr.Markdown)
        and scenario_context.strip() in str(getattr(block, "value", ""))
    ]
    guided_chatbots = [
        block
        for block in demo.blocks.values()
        if isinstance(block, gr.Chatbot) and block.show_label is False
    ]
    html_blocks = [block for block in demo.blocks.values() if isinstance(block, gr.HTML)]

    assert len(walkthrough_blocks) == 1
    assert len(step_blocks) == len(scenario_names) + 1
    assert step_blocks[0].interactive is True
    assert all(step.interactive is False for step in step_blocks[1:])
    assert len(scenario_briefs) == 1
    assert len(guided_chatbots) == len(scenario_names)
    assert all(chatbot.container is False for chatbot in guided_chatbots)
    assert all(chatbot.buttons == [] for chatbot in guided_chatbots)
    assert html_blocks == []



def test_build_guided_demo_wires_events_with_dynamic_scenario_count(tmp_path: Path) -> None:
    """Guided `build_demo` should support scenario counts loaded from the directory."""
    scenario_names = (
        "Scenario_01.md",
        "Scenario_02.md",
        "Scenario_03.md",
        "Scenario_04.md",
    )
    for scenario_name in scenario_names:
        (tmp_path / scenario_name).write_text(f"# {scenario_name}\n", encoding="utf-8")

    args = GradioAppArgs(
        scenario_dir=str(tmp_path),
        user_id=0,
        policy="graph",
        locale=Locale.EN.value,
        debug=False,
        random_scenario=False,
        ui="guided",
    )
    demo = build_demo(session_manager=_FakeSessionManager(), args=args)
    walkthrough_blocks = [
        block for block in demo.blocks.values() if isinstance(block, gr.Walkthrough)
    ]
    html_blocks = [block for block in demo.blocks.values() if isinstance(block, gr.HTML)]

    assert isinstance(demo, gr.Blocks)
    assert demo.mode == "blocks"
    assert len(walkthrough_blocks) == 1
    assert html_blocks == []
    assert args.picker_interactive is False



def test_history_is_complete_accepts_plain_completion_text() -> None:
    """Completion detection should not depend solely on metadata surviving in history."""
    assert history_is_complete([
        {
            "role": "assistant",
            "content": COMPLETION_MESSAGE.content,
        }
    ]) is True



def test_history_is_complete_accepts_gradio_processed_messages() -> None:
    """Completion detection should work with Gradio Chatbot message models."""
    chatbot = gr.Chatbot()
    processed = chatbot.postprocess([COMPLETION_MESSAGE])

    assert history_is_complete(processed) is True


@pytest.mark.asyncio
async def test_stream_backend_events_appends_and_streams_assistant_message() -> None:
    """Streaming should follow the official append-and-mutate assistant pattern."""
    history = [gr.ChatMessage(role="user", content="Hello")]
    events = [BackendEvent(kind="message", text="Hello there")]

    snapshots = []
    async for updated_history in stream_backend_events(history=history, events=events):
        snapshots.append(
            [
                item.content if isinstance(item, gr.ChatMessage) else item.get("content")
                for item in updated_history
            ]
        )

    assert snapshots
    assert snapshots[0] == ["Hello", "H"]
    assert snapshots[-1] == ["Hello", "Hello there"]
