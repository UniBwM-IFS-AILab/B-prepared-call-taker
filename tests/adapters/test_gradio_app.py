"""Smoke tests for Gradio adapter wiring."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

import gradio as gr
import pytest

import ems_prepared.adapters.gradio.app as gradio_app
from ems_prepared.adapters.gradio.app import (
    GRADIO_CSS,
    _launch_gradio_app,
    build_demo,
    run_gradio_app,
)
from ems_prepared.adapters.gradio.args import GradioAppArgs
from ems_prepared.adapters.gradio.chat_stream import stream_backend_events
from ems_prepared.adapters.gradio.flow import (
    build_initial_guided_state,
    current_context_markdown,
    bot_respond,
    reset_standard_session,
    restore_standard_controls_after_response,
    save_standard_survey,
)
from ems_prepared.adapters.gradio.scenarios import COMPLETION_MESSAGE
from ems_prepared.adapters.gradio.session_runtime import DEBUG_STATIC_REPLY
from ems_prepared.adapters.gradio.state import SessionRef
from ems_prepared.adapters.gradio.survey import DEFAULT_SURVEY
from ems_prepared.model.context import Locale
from ems_prepared.model.contracts import (
    BackendEvent,
    BackendEventKind,
    MessageFeedback,
    SessionHistory,
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

    async def submit_survey(
        self,
        session_id: UUID,
        responses: list[dict[str, object]],
        feedback: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> Path:
        _ = (session_id, responses, feedback, metadata)
        return Path("survey.json")

    async def submit_feedback(self, session_id: UUID, feedback: MessageFeedback) -> None:
        _ = (session_id, feedback)

    def get_view_state(self, session_id: UUID) -> SessionState | None:
        _ = session_id
        return None

    def get_history(self, session_id: UUID) -> SessionHistory | None:
        _ = session_id
        return None


class _LaunchRecorderDemo:
    """Tiny launch stub that records queue/launch parameters."""

    def __init__(self) -> None:
        self.queue_limit: int | None = None
        self.launch_kwargs: dict[str, object] | None = None

    def queue(self, *, default_concurrency_limit: int):
        self.queue_limit = default_concurrency_limit
        return self

    def launch(self, **kwargs):
        self.launch_kwargs = kwargs
        return object(), "http://127.0.0.1:7860", "https://example.gradio.live"

    def close(self) -> None:
        return


class _CloseRecorderDemo:
    """Small demo stub that records whether close() was called."""

    def __init__(self) -> None:
        self.closed = False
        self.is_running = True

    def close(self) -> None:
        self.closed = True
        self.is_running = False


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
    step_blocks = [block for block in demo.blocks.values() if isinstance(block, gr.Step)]

    assert isinstance(demo, gr.Blocks)
    assert demo.mode == "blocks"
    assert len(step_blocks) == 3
    assert "#session_walkthrough [role=\"tab\"]" in GRADIO_CSS
    assert isinstance(args.resolve_policy(), str)
    assert args.is_debug_enabled() is False
    assert args.picker_interactive is True
    assert Locale.EN.value == "english"


def test_build_standard_demo_with_consent_adds_preliminary_step(
    tmp_path: Path,
) -> None:
    """Consent-enabled standard mode should insert a consent step before selection."""
    (tmp_path / "Scenario_01.md").write_text("# Scenario 01\n", encoding="utf-8")
    args = GradioAppArgs(
        scenario_dir=str(tmp_path),
        user_id=0,
        policy="graph",
        locale=Locale.EN.value,
        debug=False,
        random_scenario=False,
        require_consent=True,
    )

    demo = build_demo(session_manager=_FakeSessionManager(), args=args)
    step_blocks = [block for block in demo.blocks.values() if isinstance(block, gr.Step)]
    checkboxes = [block for block in demo.blocks.values() if isinstance(block, gr.Checkbox)]
    continue_buttons = [
        block
        for block in demo.blocks.values()
        if isinstance(block, gr.Button)
        and getattr(block, "value", None) == "Review Scenario"
    ]

    assert len(step_blocks) == 4
    assert len(checkboxes) == 1
    assert continue_buttons
    assert any(button.interactive is False for button in continue_buttons)


def test_standard_navigation_callbacks_are_non_queued_on_first_load(
    tmp_path: Path,
) -> None:
    """UI-only standard walkthrough transitions should bypass the Gradio queue."""
    (tmp_path / "Scenario_01.md").write_text("# Scenario 01\n", encoding="utf-8")
    args = GradioAppArgs(
        scenario_dir=str(tmp_path),
        user_id=0,
        policy="graph",
        locale=Locale.EN.value,
        debug=True,
        random_scenario=False,
        require_consent=True,
    )

    demo = build_demo(session_manager=_FakeSessionManager(), args=args)
    steps_by_label = {
        block.label: block
        for block in demo.blocks.values()
        if isinstance(block, gr.Step)
        and getattr(block, "label", None)
        in {"Consent", "Scenario Selection", "Scenario 1", "Survey"}
    }
    walkthrough_blocks = [
        block for block in demo.blocks.values() if isinstance(block, gr.Walkthrough)
    ]
    assert len(walkthrough_blocks) == 1
    step_navigation_dependencies = [
        dep
        for dep in demo.config["dependencies"]
        if dep.get("outputs")
        and walkthrough_blocks[0]._id in dep["outputs"]
    ]

    assert step_navigation_dependencies
    assert all(dep["queue"] is False for dep in step_navigation_dependencies)


def test_launch_gradio_app_uses_verbose_flag_without_debug_ui_toggle() -> None:
    """Verbose mode should enable Gradio launch debug logs without debug UI mode."""
    args = GradioAppArgs(
        scenario_dir=None,
        user_id=0,
        policy="graph",
        locale=Locale.EN.value,
        debug=False,
        verbose=True,
        random_scenario=False,
    )
    demo = _LaunchRecorderDemo()

    _ = _launch_gradio_app(demo=demo, args=args)

    assert demo.queue_limit == 16
    assert demo.launch_kwargs is not None
    assert demo.launch_kwargs["debug"] is True
    assert demo.launch_kwargs["share"] is True
    assert demo.launch_kwargs["inbrowser"] is False
    assert demo.launch_kwargs["prevent_thread_lock"] is True


def test_launch_gradio_app_keeps_debug_launch_behavior() -> None:
    """Debug mode should still launch in-browser while keeping share links enabled."""
    args = GradioAppArgs(
        scenario_dir=None,
        user_id=0,
        policy="graph",
        locale=Locale.EN.value,
        debug=True,
        verbose=False,
        random_scenario=False,
    )
    demo = _LaunchRecorderDemo()

    _ = _launch_gradio_app(demo=demo, args=args)

    assert demo.launch_kwargs is not None
    assert demo.launch_kwargs["debug"] is True
    assert demo.launch_kwargs["share"] is True
    assert demo.launch_kwargs["inbrowser"] is True


def test_run_gradio_app_returns_immediately_with_exit_on_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--exit-on-launch` should skip the process hold loop."""
    args = GradioAppArgs(
        scenario_dir=None,
        user_id=0,
        policy="graph",
        locale=Locale.EN.value,
        debug=False,
        random_scenario=False,
        exit_on_launch=True,
    )
    demo = _CloseRecorderDemo()

    monkeypatch.setattr(gradio_app, "build_demo", lambda **_: demo)
    monkeypatch.setattr(
        gradio_app,
        "_launch_gradio_app",
        lambda **_: (object(), "http://127.0.0.1:7860", "https://example.gradio.live"),
    )

    code = run_gradio_app(session_manager=_FakeSessionManager(), args=args)

    assert code == 0
    assert demo.closed is False


def test_run_gradio_app_returns_130_on_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One Ctrl-C during the hold loop should stop Gradio with exit code 130."""
    args = GradioAppArgs(
        scenario_dir=None,
        user_id=0,
        policy="graph",
        locale=Locale.EN.value,
        debug=False,
        random_scenario=False,
    )
    demo = _CloseRecorderDemo()

    monkeypatch.setattr(gradio_app, "build_demo", lambda **_: demo)
    monkeypatch.setattr(
        gradio_app,
        "_launch_gradio_app",
        lambda **_: (object(), "http://127.0.0.1:7860", None),
    )

    def _raise_interrupt(_seconds: int) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(gradio_app.time, "sleep", _raise_interrupt)

    code = run_gradio_app(session_manager=_FakeSessionManager(), args=args)

    assert code == 130
    assert demo.closed is True


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
    assert all(step.interactive is True for step in step_blocks)
    assert len(scenario_briefs) == 1
    assert len(guided_chatbots) == 1
    assert all(chatbot.container is False for chatbot in guided_chatbots)
    assert all(chatbot.buttons == [] for chatbot in guided_chatbots)
    assert html_blocks == []


def test_build_guided_demo_with_consent_adds_preliminary_step(tmp_path: Path) -> None:
    """Consent-enabled guided mode should add a consent step before scenario 1."""
    scenario_names = ("Scenario_01.md", "Scenario_02.md")
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
        require_consent=True,
    )
    demo = build_demo(session_manager=_FakeSessionManager(), args=args)

    step_blocks = [block for block in demo.blocks.values() if isinstance(block, gr.Step)]
    checkboxes = [block for block in demo.blocks.values() if isinstance(block, gr.Checkbox)]
    continue_buttons = [
        block
        for block in demo.blocks.values()
        if isinstance(block, gr.Button)
        and getattr(block, "value", None) == "Review Scenario"
    ]

    assert len(step_blocks) == len(scenario_names) + 2
    assert len(checkboxes) == 1
    assert continue_buttons
    assert any(button.interactive is False for button in continue_buttons)
    assert step_blocks[0].label == "Consent"
    assert step_blocks[1].label == "Scenario 1"


@pytest.mark.asyncio
async def test_stream_backend_events_appends_and_streams_assistant_message() -> None:
    """Streaming should follow append-and-mutate assistant pattern."""
    history = [gr.ChatMessage(role="user", content="Hello")]
    events = [BackendEvent(kind=BackendEventKind.MESSAGE, text="Hello there")]

    snapshots = [
        [
            item.content if isinstance(item, gr.ChatMessage) else item.get("content")
            for item in updated_history
        ]
        async for updated_history in stream_backend_events(history=history, events=events)
    ]

    assert snapshots
    assert snapshots[0] == ["Hello", "H"]
    assert snapshots[-1] == ["Hello", "Hello there"]


@pytest.mark.asyncio
async def test_standard_bot_respond_uses_static_response_when_policy_calls_are_skipped() -> None:
    """Debug sessions should bypass policy handle_input and complete after static reply."""
    session_data = SessionRef(
        user_id=str(uuid4()),
        session_id=str(uuid4()),
        policy_name="graph",
        scenario_name="Scenario_01.md",
        experiment_name="",
    )

    snapshots = [
        (
            [
                item.content if isinstance(item, gr.ChatMessage) else item.get("content")
                for item in updated_history
            ],
            completed,
        )
        async for updated_history, completed in bot_respond(
            history=[],
            session_data=session_data,
            user_msg="hello",
            session_manager=_FakeSessionManager(),
            skip_policy_calls=True,
        )
    ]

    assert snapshots
    assert snapshots[-1] == (
        [DEBUG_STATIC_REPLY, COMPLETION_MESSAGE.content],
        True,
    )


@pytest.mark.asyncio
async def test_reset_standard_session_returns_expected_outputs(tmp_path: Path) -> None:
    """Standard reset helper should return a stable output shape."""
    (tmp_path / "Scenario_01.md").write_text("# Scenario 01\n", encoding="utf-8")
    args = GradioAppArgs(
        scenario_dir=str(tmp_path),
        user_id=0,
        policy="graph",
        locale=Locale.EN.value,
        debug=False,
        random_scenario=False,
    )
    result = await reset_standard_session(
        SessionRef(
            user_id=str(uuid4()),
            session_id=str(uuid4()),
            policy_name="graph",
            scenario_name="Scenario_01.md",
            experiment_name="",
        ),
        user_id=str(uuid4()),
        scenario_name="Scenario_01.md",
        session_manager=_FakeSessionManager(),
        args=args,
    )

    assert len(result) == 17
    assert isinstance(result[-2], dict)
    assert result[-2]["value"] == "Start New Session ->"
    assert result[-2]["visible"] == "hidden"


@pytest.mark.asyncio
async def test_save_standard_survey_shows_restart_controls() -> None:
    """Standard survey submit should reveal thank-you copy and restart button."""
    result = await save_standard_survey(
        SessionRef(
            user_id=str(uuid4()),
            session_id=str(uuid4()),
            policy_name="graph",
            scenario_name="Scenario_01.md",
            experiment_name="",
        ),
        "thanks",
        *([5] * len(DEFAULT_SURVEY.questions)),
        session_manager=_FakeSessionManager(),
        skip_policy_calls=False,
    )

    thanks_update = result[-2]
    restart_update = result[-1]

    assert isinstance(thanks_update, dict)
    assert thanks_update["visible"] is True
    assert thanks_update["value"] == "### ✓ Thank you for your feedback!"
    assert isinstance(restart_update, dict)
    assert restart_update["visible"] is True
    assert restart_update["value"] == "Start New Session ->"


def test_restore_standard_controls_after_completion_shows_continue_to_survey() -> None:
    """A completed standard chat should expose continue-to-survey button."""
    updates = restore_standard_controls_after_response(
        True,
        picker_interactive=True,
    )

    continue_button_update = updates[-1]

    assert isinstance(continue_button_update, dict)
    assert continue_button_update["visible"] is True
    assert continue_button_update["interactive"] is True
