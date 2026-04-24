"""Tests for reusable Gradio view components."""

from __future__ import annotations

import gradio as gr

from ems_prepared.adapters.gradio.components import (
    build_chat_section,
    build_context_panel,
    build_intro_section,
    build_scenario_flow,
    build_survey_section,
)
from ems_prepared.adapters.gradio.survey import DEFAULT_SURVEY


def test_context_panel_updates_shared_outputs() -> None:
    """The context panel should expose one update surface for the scenario brief."""
    with gr.Blocks():
        panel = build_context_panel(
            scenario_markdown="# Scenario\n",
            overview_html="<p>Overview</p>",
        )

    updates = panel.updates(
        scenario_markdown="# Updated\n",
        overview_html="<p>Updated</p>",
    )

    assert panel.outputs() == [panel.overview_display, panel.scenario_markdown]
    assert updates[panel.scenario_markdown]["value"] == "# Updated\n"
    assert updates[panel.overview_display]["value"] == "<p>Updated</p>"


def test_call_step_control_updates_toggle_manual_actions() -> None:
    """The shared call pane should centralize control state updates."""
    with gr.Blocks():
        call = build_chat_section(
            reset_label="Restart",
            continue_label="Continue",
            finish_label="Finish",
        )

    assert call.continue_button is not None
    assert call.finish_button is not None

    updates = call.control_updates(
        input_enabled=False,
        reset_enabled=True,
        continue_visible=True,
        finish_visible=False,
    )

    assert updates[call.input_box]["interactive"] is False
    assert updates[call.reset]["interactive"] is True
    assert updates[call.continue_button]["visible"] is True
    assert updates[call.finish_button]["visible"] is False


def test_scenario_flow_tracks_phase_container_states() -> None:
    """Guided scenario flows should keep one call pane mounted through completion."""
    with gr.Blocks():
        flow = build_scenario_flow(
            intro_markdown="### Intro",
            start_label="Start",
            reset_label="Restart",
            completion_markdown="### Done",
            advance_label="Continue",
        )

    brief_updates = flow.brief_updates()
    starting_updates = flow.starting_updates()
    processing_updates = flow.processing_updates()
    streaming_updates = flow.streaming_updates()
    complete_updates = flow.complete_updates()

    assert brief_updates[flow.brief_group].visible is True
    assert brief_updates[flow.call_group].visible is False
    assert starting_updates[flow.call_group].visible is True
    assert starting_updates[flow.call.status_markdown]["visible"] is False
    assert processing_updates[flow.call.status_markdown]["visible"] is False
    assert processing_updates[flow.call.status_markdown]["value"] == ""
    assert streaming_updates[flow.call.status_markdown]["visible"] is False
    assert streaming_updates[flow.call.input_box]["interactive"] is False
    assert flow.call.chatbot.show_label is False
    assert flow.call.chatbot.container is False
    assert flow.call.chatbot.buttons == []
    assert flow.call.chatbot.group_consecutive_messages is False
    assert complete_updates[flow.call_group].visible is True
    assert complete_updates[flow.call.input_group].visible is False
    assert complete_updates[flow.call.completion_markdown].visible is True
    assert complete_updates[flow.call.continue_button].visible is True
    assert complete_updates[flow.call.continue_button].interactive is True


def test_intro_and_survey_views_render_inside_blocks_context() -> None:
    """Intro and survey sections should render as reusable views without context errors."""
    with gr.Blocks():
        intro = build_intro_section(
            intro_markdown="### Intro",
            start_label="Start",
        )
        survey = build_survey_section(
            survey=DEFAULT_SURVEY,
            intro_markdown="### Survey",
            feedback_label="Feedback",
            feedback_placeholder="Share feedback",
            submit_label="Submit",
            thanks_markdown="### Thanks",
            restart_label="Restart",
        )

    assert intro.start_button.value == "Start"
    assert len(survey.survey_radios) == len(DEFAULT_SURVEY.questions)
    assert survey.restart_button is not None
    reset_updates = survey.reset_updates()
    assert reset_updates[survey.completion_group].visible is False
    assert reset_updates[survey.survey_submit]["interactive"] is False
