from __future__ import annotations

import asyncio

import pytest

from ems_prepared.model.context import Locale, Settings
from experiments.nlu_eval.common.runner import run_full_agent_once


def test_run_full_agent_once_propagates_agent_exceptions(
    monkeypatch,
) -> None:
    async def raise_failure(**kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(
        "experiments.nlu_eval.common.runner.run_state_fill",
        raise_failure,
    )

    deps = Settings(
        name="nlu_eval_test",
        locale=Locale.EN,
        experiment_name="nlu_eval/test",
        scenario_name="item-1",
        policy_name="full_agent",
    )

    with pytest.raises(RuntimeError, match="synthetic failure"):
        asyncio.run(
            run_full_agent_once(
                deps=deps,
                experiment_id="experiment2",
                item_id="item-1",
                repeat_index=0,
                operator_question="What has just happened acutely?",
                caller_utterance="She suddenly cannot speak properly.",
            )
        )
