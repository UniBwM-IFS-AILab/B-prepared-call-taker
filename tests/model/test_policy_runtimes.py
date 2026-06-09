"""Tests for policy runtime builders and adapter behavior."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic_graph import End

import ems_prepared.policies.llm_only.runtime as agent_runtime
import ems_prepared.policies.pydantic_graph.emergency_main_graph as emergency_main_graph
import ems_prepared.policies.pydantic_graph.runtime as graph_runtime
import ems_prepared.policies.pydantic_graph.tcpr_subgraph as tcpr_subgraph
from ems_prepared.model.context import Locale, Settings
from ems_prepared.model.contracts import (
    BackendEvent,
    BackendEventKind,
    SessionResumeError,
)
from ems_prepared.policies.llm_only.runtime import (
    AgentConversationPolicy,
    build_agent_policy,
)
from ems_prepared.policies.pydantic_graph.runtime import (
    GraphConversationPolicy,
    build_graph_policy,
)
from ems_prepared.policies.pydantic_graph.tcpr_subgraph import EMSArrived


@pytest.mark.asyncio
async def test_build_graph_policy_uses_graph_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Graph policy construction should use the graph builder helper."""
    deps = Settings(name="tests_graph", policy_name="graph")
    sentinel_graph: Any = object()

    async def fake_build_graph():
        return sentinel_graph

    monkeypatch.setattr(graph_runtime, "build_graph", fake_build_graph)

    policy = await build_graph_policy(deps)

    assert isinstance(policy, GraphConversationPolicy)
    assert policy.graph is sentinel_graph
    assert policy.deps is deps


@pytest.mark.asyncio
async def test_build_agent_policy_uses_agent_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Agent policy construction should use the agent builder helper."""
    deps = Settings(name="tests_agent", policy_name="agent")
    sentinel_agent: Any = object()

    def fake_build_emergency_agent(_: Settings):
        return sentinel_agent

    monkeypatch.setattr(
        agent_runtime,
        "build_emergency_agent",
        fake_build_emergency_agent,
    )

    policy = await build_agent_policy(deps)

    assert isinstance(policy, AgentConversationPolicy)
    assert policy.policy.agent is sentinel_agent
    assert policy.policy.deps is deps
    assert policy.policy.history == []


@pytest.mark.asyncio
async def test_build_graph_policy_does_not_probe_resume_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Graph policy construction should not execute graph turns eagerly."""
    monkeypatch.chdir(tmp_path)
    deps = Settings(name="tests_graph_resume", policy_name="graph")
    sentinel_graph: Any = object()

    async def fake_build_graph():
        return sentinel_graph

    monkeypatch.setattr(graph_runtime, "build_graph", fake_build_graph)
    monkeypatch.setattr(
        graph_runtime,
        "run_graph",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected")),
    )

    policy = await build_graph_policy(deps)

    assert isinstance(policy, GraphConversationPolicy)
    assert policy.graph is sentinel_graph
    assert policy.deps is deps


@pytest.mark.asyncio
async def test_build_agent_policy_does_not_require_snapshot_at_build_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Agent policy construction should not probe persisted snapshots eagerly."""
    deps = Settings(name="tests_agent_resume", policy_name="agent")
    sentinel_agent: Any = object()

    def fake_build_emergency_agent(_: Settings):
        return sentinel_agent

    monkeypatch.setattr(
        agent_runtime,
        "build_emergency_agent",
        fake_build_emergency_agent,
    )
    deps.load_agent_snapshot = lambda: (_ for _ in ()).throw(
        AssertionError("unexpected snapshot probe")
    )

    policy = await build_agent_policy(deps)

    assert isinstance(policy, AgentConversationPolicy)
    assert policy.policy.agent is sentinel_agent


@pytest.mark.asyncio
async def test_graph_policy_records_completion_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Graph runtime should record completion artifacts via callback wiring."""
    deps = Settings(name="tests_graph_completion", policy_name="graph")
    completion_calls: list[tuple[Any, list[Any]]] = []
    deps.record_completion_artifacts = lambda state, history: completion_calls.append(
        (state, history)
    )

    async def fake_run_graph(
        _graph: object,
        _deps: Settings,
        _text: str | None,
        on_complete=None,
    ):
        assert on_complete is not None
        on_complete({"finished": True}, [{"speaker": "operator"}])
        return End({"finished": True})

    monkeypatch.setattr(graph_runtime, "run_graph", fake_run_graph)

    graph_placeholder: Any = object()
    policy = GraphConversationPolicy(graph=graph_placeholder, deps=deps)
    events = await policy.start()

    assert len(completion_calls) == 1
    assert completion_calls[0][0] == {"finished": True}
    assert completion_calls[0][1] == [{"speaker": "operator"}]
    assert len(events) == 1
    assert events[0] == BackendEvent(
        kind=BackendEventKind.COMPLETED,
        text="The emergency call has been processed.",
        payload={"finished": True},
    )


@pytest.mark.asyncio
async def test_graph_policy_handle_input_requires_existing_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Existing-session graph turns should demand persisted runtime state."""
    deps = Settings(name="tests_graph_handle_input", policy_name="graph")
    seen_runner: list[str] = []

    async def fake_resume_existing_graph(
        _graph: object,
        _deps: Settings,
        _text: str | None,
        on_complete=None,
    ):
        _ = on_complete
        seen_runner.append("resume")
        return End({"finished": True})

    monkeypatch.setattr(
        graph_runtime,
        "resume_existing_graph",
        fake_resume_existing_graph,
    )

    policy = GraphConversationPolicy(graph=object(), deps=deps)
    _ = await policy.handle_input("hello")

    assert seen_runner == ["resume"]


@pytest.mark.asyncio
async def test_resume_existing_graph_forwards_answer_to_main_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Main-graph resume should process the caller answer instead of dropping it."""
    deps = Settings(name="tests_graph_resume_answer", policy_name="graph")
    captured_answer: list[str | None] = []

    async def fake_tcpr_run_graph(*, deps: Settings, answer: str | None = None):
        _ = (deps, answer)
        return None

    async def fake_resume_existing_from_persistence(_deps: Settings, _graph: object):
        return object(), object()

    async def fake_run_loaded_graph(
        _graph: object,
        _deps: Settings,
        _node: object,
        _persistence: object,
        answer: str | None = None,
        on_complete=None,
    ):
        _ = on_complete
        captured_answer.append(answer)
        return End({"finished": True})

    monkeypatch.setattr(
        tcpr_subgraph,
        "run_graph",
        fake_tcpr_run_graph,
    )
    monkeypatch.setattr(
        emergency_main_graph,
        "resume_existing_from_persistence",
        fake_resume_existing_from_persistence,
    )
    monkeypatch.setattr(
        emergency_main_graph,
        "_run_loaded_graph",
        fake_run_loaded_graph,
    )

    _ = await emergency_main_graph.resume_existing_graph(
        graph=object(),
        deps=deps,
        answer="At home",
    )

    assert captured_answer == ["At home"]


@pytest.mark.asyncio
async def test_graph_policy_emits_tcpr_handover_message_before_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TCPR completion should surface its final handover line before the terminal event."""
    deps = Settings(name="tests_graph_tcpr_completion", policy_name="graph")
    results = iter(
        [
            EMSArrived(),
            End({"finished": True}),
        ]
    )

    async def fake_resume_existing_graph(
        _graph: object,
        _deps: Settings,
        _text: str | None,
        on_complete=None,
    ):
        _ = on_complete
        return next(results)

    monkeypatch.setattr(
        graph_runtime,
        "resume_existing_graph",
        fake_resume_existing_graph,
    )

    policy = GraphConversationPolicy(graph=object(), deps=deps)
    events = await policy.handle_input("they are here")

    assert events == [
        BackendEvent(
            kind=BackendEventKind.MESSAGE,
            text="Please hand over to the paramedics",
        ),
        BackendEvent(
            kind=BackendEventKind.COMPLETED,
            text="The emergency call has been processed.",
            payload={"finished": True},
        ),
    ]


@pytest.mark.asyncio
async def test_graph_policy_accumulates_message_events_before_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Graph runtime should stream TCPR message nodes before the next question."""
    deps = Settings(name="tests_graph_tcpr_messages", policy_name="graph")
    results = iter(
        [
            tcpr_subgraph.TCPRMessage(
                messages={Locale.EN: "Step one", Locale.DE: "Schritt eins"},
                phase="instruct",
                step_index=0,
            ),
            tcpr_subgraph.TCPRMessage(
                messages={Locale.EN: "Step two", Locale.DE: "Schritt zwei"},
                phase="instruct",
                step_index=1,
            ),
            tcpr_subgraph.TCPRQuestion(
                question="Please confirm when done.",
                phase="instruct",
                step_index=1,
            ),
        ]
    )

    async def fake_run_graph(
        _graph: object,
        _deps: Settings,
        _text: str | None,
        on_complete=None,
    ):
        _ = on_complete
        return next(results)

    monkeypatch.setattr(graph_runtime, "run_graph", fake_run_graph)

    policy = GraphConversationPolicy(graph=object(), deps=deps)
    events = await policy.start()

    assert events == [
        BackendEvent(kind=BackendEventKind.MESSAGE, text="Step one"),
        BackendEvent(kind=BackendEventKind.MESSAGE, text="Step two"),
        BackendEvent(
            kind=BackendEventKind.QUESTION,
            text="Please confirm when done.",
        ),
    ]


@pytest.mark.asyncio
async def test_agent_policy_records_completion_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Agent runtime should record completion artifacts via shared helper."""
    deps = Settings(name="tests_agent_completion", policy_name="agent")
    completion_calls: list[tuple[Any, list[Any]]] = []
    deps.record_completion_artifacts = lambda state, history: completion_calls.append(
        (state, history)
    )

    class FakeResult:
        output = object()

        @staticmethod
        def usage() -> dict[str, int]:
            return {}

    async def fake_run_agent_with_capture(*args, **kwargs):
        return FakeResult(), []

    monkeypatch.setattr(
        agent_runtime,
        "run_agent_with_capture",
        fake_run_agent_with_capture,
    )
    monkeypatch.setattr(
        agent_runtime,
        "extract_last_model_response",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        agent_runtime,
        "update_history_and_merge_state",
        lambda *args, **kwargs: (True, None),
    )
    agent_placeholder: Any = object()

    policy = AgentConversationPolicy(
        policy=agent_runtime.AgentPolicy(
            agent=agent_placeholder,
            # policy state can be mocked in this test; runtime path under test
            # is completion recording, not state validation.
            state={"finished": True},  # type: ignore[arg-type]
            history=[],
            deps=deps,
        ),
        deps=deps,
    )
    events = await policy.start()

    assert len(completion_calls) == 1
    assert completion_calls[0][0] == {"finished": True}
    assert completion_calls[0][1] == []
    assert len(events) == 1
    assert events[0] == BackendEvent(
        kind=BackendEventKind.COMPLETED,
        text="The emergency call has been processed.",
        payload={"finished": True},
    )


@pytest.mark.asyncio
async def test_agent_policy_handle_input_requires_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Existing-session agent turns should fail loudly without a snapshot."""
    deps = Settings(name="tests_agent_handle_input", policy_name="agent")
    sentinel_agent: Any = object()

    def fake_build_emergency_agent(_: Settings):
        return sentinel_agent

    monkeypatch.setattr(
        agent_runtime,
        "build_emergency_agent",
        fake_build_emergency_agent,
    )
    deps.load_agent_snapshot = lambda: None

    policy = await build_agent_policy(deps)

    with pytest.raises(SessionResumeError):
        await policy.handle_input("hello")
