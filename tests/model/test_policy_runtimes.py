"""Tests for policy runtime builders and adapter behavior."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic_graph.nodes import End

import ems_prepared.policies.llm_only.runtime as agent_runtime
import ems_prepared.policies.pydantic_graph.runtime as graph_runtime
from ems_prepared.model.context import Settings
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
    """Resume-mode graph builds should not validate persistence eagerly."""
    monkeypatch.chdir(tmp_path)
    deps = Settings(name="tests_graph_resume", policy_name="graph", resume_expected=True)
    sentinel_graph: Any = object()

    async def fake_build_graph():
        return sentinel_graph

    monkeypatch.setattr(graph_runtime, "build_graph", fake_build_graph)

    policy = await build_graph_policy(deps)

    assert isinstance(policy, GraphConversationPolicy)
    assert policy.graph is sentinel_graph
    assert policy.deps is deps


@pytest.mark.asyncio
async def test_build_agent_policy_requires_snapshot_when_resume_expected() -> None:
    """Resume-mode agent builds should fail loudly without a snapshot."""
    deps = Settings(name="tests_agent_resume", policy_name="agent", resume_expected=True)

    with pytest.raises(SessionResumeError):
        await build_agent_policy(deps)


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
