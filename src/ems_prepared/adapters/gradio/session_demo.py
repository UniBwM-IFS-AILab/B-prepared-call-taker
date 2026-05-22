"""Shared outer Gradio session wrapper for both standard and guided modes."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from functools import partial

import gradio as gr
from gradio import ChatMessage

from ems_prepared.adapters.gradio.args import GradioAppArgs
from ems_prepared.adapters.gradio.asr import (
    AsrPane,
    StreamingAsr,
    wire_asr_to_chat_input,
)
from ems_prepared.adapters.gradio.chat_stream import stream_backend_events
from ems_prepared.adapters.gradio.components import (
    ConsentStepView,
    ContextPanelView,
    ScenarioFlowView,
    SelectionStepView,
    SurveyStepView,
)
from ems_prepared.adapters.gradio.consent import (
    DEFAULT_CONSENT_CHECKBOX_LABEL,
    DEFAULT_CONSENT_CONTEXT_MARKDOWN,
    resolve_consent_markdown,
)
from ems_prepared.adapters.gradio.feedback import persist_chat_feedback
from ems_prepared.adapters.gradio.flow import (
    advance_to_next_scenario,
    archive_guided_runtime_artifacts,
    begin_guided_start,
    capture_guided_user_submit,
    enter_guided_survey,
    finalize_guided_turn,
    finalize_started_call,
    restart_current_call,
    save_guided_survey,
    start_failure_state,
    start_guided_session,
    stream_guided_turn,
)
from ems_prepared.adapters.gradio.scenarios import (
    construct_scenario_desc,
    get_random_scenario,
    list_md_files,
)
from ems_prepared.adapters.gradio.session_runtime import (
    end_active_session,
    log_consent_acceptance,
    resolve_or_create_user_id,
    session_info_text,
)
from ems_prepared.adapters.gradio.state import GuidedState
from ems_prepared.adapters.gradio.survey import (
    DEFAULT_SURVEY,
    check_all_survey_answers,
    submit_survey_for_gradio,
)
from ems_prepared.adapters.gradio.walkthrough import (
    WalkthroughController,
    ordered_component_updates,
)
from ems_prepared.model.contracts import SessionManager

logger = logging.getLogger(__name__)

REVIEW_SCENARIO_LABEL = "Review Scenario"
START_SCENARIO_LABEL = "Start Scenario"
START_SESSION_LABEL = "Start Session"
RESTART_CURRENT_CALL_LABEL = "Restart Current Call"
RESTART_SESSION_LABEL = "Restart Session"
CONTINUE_TO_NEXT_SCENARIO_LABEL = "Continue to Next Scenario"
CONTINUE_TO_SURVEY_LABEL = "Continue to Survey"
START_NEW_GUIDED_SESSION_LABEL = "Start New Guided Session"
START_NEW_SESSION_LABEL = "Start New Session"


def _mounted_visibility(visible: bool) -> bool | str:
    """Keep components mounted while hidden to preserve client-side bindings."""
    return True if visible else "hidden"


@dataclass(slots=True)
class SessionShellUI:
    """Shared session shell used by standard and guided modes."""

    demo: gr.Blocks
    walkthrough: gr.Walkthrough
    steps: list[gr.Step]
    state: gr.State
    user_id_state: gr.State
    session_info_display: gr.Markdown
    context: ContextPanelView
    consent: ConsentStepView | None
    selection: SelectionStepView | None
    scenario_flow: ScenarioFlowView
    survey: SurveyStepView
    asr_pane: AsrPane | None


def _resolve_loaded_scenarios(
    args: GradioAppArgs,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return available directory scenarios and loaded session scenarios."""
    available = tuple(list_md_files(scenario_dir=args.scenario_dir))
    if not available:
        raise ValueError("At least one scenario file is required.")
    if args.ui == "guided":
        loaded = tuple(args.guided_scenarios) or available
        missing = [name for name in loaded if name not in available]
        if missing:
            raise ValueError(f"Unknown guided scenarios: {', '.join(missing)}")
        return available, loaded
    return available, (available[0],)


def _initial_phase(*, args: GradioAppArgs, consent_required: bool) -> str:
    if consent_required:
        return "consent"
    if args.ui == "standard":
        return "selection"
    return "brief"


def _build_initial_state(
    *,
    args: GradioAppArgs,
    loaded_scenarios: tuple[str, ...],
    consent_required: bool,
) -> GuidedState:
    return GuidedState(
        scenario_names=loaded_scenarios,
        phase=_initial_phase(args=args, consent_required=consent_required),
    )


def _current_scenario_name(state: GuidedState) -> str | None:
    if not state.scenario_names:
        return None
    index = min(state.active_index, len(state.scenario_names) - 1)
    return state.scenario_names[index]


def _is_scenario_phase(phase: str) -> bool:
    return phase in {"brief", "starting", "call", "complete"}


def _context_markdown(
    state: GuidedState,
    *,
    args: GradioAppArgs,
    consent_required: bool,
) -> str:
    if consent_required and state.phase == "consent":
        return DEFAULT_CONSENT_CONTEXT_MARKDOWN
    if state.phase == "survey":
        if args.ui == "guided":
            total = len(state.scenario_names)
            if state.survey_submitted:
                return (
                    "### Session Complete\n\n"
                    f"You completed all {total} scenarios and submitted the survey."
                )
            return (
                "### Final Survey\n\n"
                f"Please answer the survey for the full run across all {total} scenarios."
            )
        return "### Final Survey\n\nPlease answer the survey for this session."
    return construct_scenario_desc(
        _current_scenario_name(state),
        scenario_dir=args.scenario_dir,
    )


def _session_info_update(state: GuidedState, *, args: GradioAppArgs):
    active_session = (
        state.active_session if state.phase in {"call", "complete", "survey"} else None
    )
    scenario_name = (
        active_session.scenario_name
        if active_session
        else _current_scenario_name(state)
    )
    return gr.update(
        value=session_info_text(
            args,
            user_id=state.user_id or None,
            session_id=active_session.session_id if active_session else None,
            scenario=scenario_name,
            experiment_name=active_session.experiment_name if active_session else None,
            policy=active_session.policy_name if active_session else None,
        ),
        visible=args.is_debug_enabled(),
    )


def _context_updates(
    state: GuidedState,
    *,
    context: ContextPanelView,
    session_info_display: gr.Markdown,
    args: GradioAppArgs,
    consent_required: bool,
) -> dict[object, object]:
    updates = context.updates(
        scenario_markdown=_context_markdown(
            state,
            args=args,
            consent_required=consent_required,
        )
    )
    updates[session_info_display] = _session_info_update(state, args=args)
    return updates


def _build_scenario_intro_markdown() -> str:
    return (
        "### Ready\n\n"
        "Read the scenario brief on the right. When you are ready, start the call."
    )


def _build_scenario_complete_markdown(
    *,
    args: GradioAppArgs,
    is_final: bool,
) -> str:
    if args.ui == "guided":
        if is_final:
            return "### Call Complete\n\nContinue to the survey, or restart the current call."
        return "### Call Complete\n\nContinue to the next scenario, or restart the current call."
    return "### Call Complete\n\nContinue to the survey, or restart the session."


def _build_survey_intro_markdown(total_scenarios: int) -> str:
    if total_scenarios == 1:
        return (
            "### Final Survey\n\nPlease rate this session. All questions are required."
        )
    return (
        "### Final Survey\n\n"
        f"Please rate the full run across all {total_scenarios} scenarios. All questions are required."
    )


def _selection_step_intro() -> str:
    return (
        "### Choose Scenario\n\n"
        "Select a scenario, review the brief, and then start the session."
    )


def _selection_to_scenario_name(
    selected_scenario: str | None,
    *,
    args: GradioAppArgs,
) -> str:
    if args.random_scenario:
        return get_random_scenario(scenario_dir=args.scenario_dir)
    if selected_scenario is None:
        raise ValueError("A scenario must be selected.")
    return selected_scenario


def _apply_standard_selection(
    state: GuidedState,
    selected_scenario: str | None,
    *,
    args: GradioAppArgs,
) -> GuidedState:
    scenario_name = _selection_to_scenario_name(selected_scenario, args=args)
    return replace(
        state,
        scenario_names=(scenario_name,),
        active_index=0,
        active_session=None,
        phase="selection" if state.phase == "selection" else state.phase,
    )


def _after_consent_phase(*, args: GradioAppArgs) -> str:
    return "selection" if args.ui == "standard" else "brief"


def _reset_full_state(
    state: GuidedState,
    user_id: str,
    selected_scenario: str | None,
    *,
    args: GradioAppArgs,
    consent_required: bool,
) -> GuidedState:
    base_scenario = (
        (_selection_to_scenario_name(selected_scenario, args=args),)
        if args.ui == "standard"
        else state.scenario_names
    )
    return GuidedState(
        scenario_names=base_scenario,
        user_id=user_id or state.user_id,
        phase=_initial_phase(args=args, consent_required=consent_required),
    )


async def _save_standard_survey(
    state: GuidedState,
    feedback: str,
    *responses,
    session_manager: SessionManager,
    skip_policy_calls: bool,
):
    """Save standard-mode survey using the shared state model."""
    active_session = state.active_session
    if active_session is None:
        return (
            state,
            gr.update(),
            gr.update(),
            *[gr.update() for _ in responses],
            gr.update(),
            gr.update(),
            gr.update(visible=True),
            gr.update(
                value="### ⚠️ Session expired. Please restart the session.",
                visible=True,
            ),
            gr.update(
                value=START_NEW_SESSION_LABEL,
                interactive=True,
                visible=True,
            ),
        )

    if skip_policy_calls:
        updated_state = replace(state, survey_submitted=True, phase="survey")
        return (
            updated_state,
            gr.update(),
            gr.update(),
            *[gr.update(interactive=False) for _ in responses],
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(visible=True),
            gr.update(value="### ✓ Thank you for your feedback!", visible=True),
            gr.update(
                value=START_NEW_SESSION_LABEL,
                interactive=True,
                visible=True,
            ),
        )

    status = await submit_survey_for_gradio(
        session_manager=session_manager,
        session_id=active_session.session_id,
        survey=DEFAULT_SURVEY,
        feedback=feedback,
        responses=responses,
        metadata={
            "ui_mode": "standard",
            "scenario_name": active_session.scenario_name,
        },
    )
    if status == "expired":
        return (
            state,
            gr.update(),
            gr.update(),
            *[gr.update() for _ in responses],
            gr.update(),
            gr.update(),
            gr.update(visible=True),
            gr.update(
                value="### ⚠️ Session expired. Please restart the session.",
                visible=True,
            ),
            gr.update(
                value=START_NEW_SESSION_LABEL,
                interactive=True,
                visible=True,
            ),
        )

    updated_state = replace(state, survey_submitted=True, phase="survey")
    return (
        updated_state,
        gr.update(),
        gr.update(),
        *[gr.update(interactive=False) for _ in responses],
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(visible=True),
        gr.update(value="### ✓ Thank you for your feedback!", visible=True),
        gr.update(
            value=START_NEW_SESSION_LABEL,
            interactive=True,
            visible=True,
        ),
    )


def _initialize_user_id(
    stored_user_id: str,
    state: GuidedState,
    request: gr.Request | None = None,
):
    query_params = dict(getattr(request, "query_params", {}) or {}) if request else {}
    requested_user_id = str(query_params.get("user_id") or "")
    user_id = str(resolve_or_create_user_id(requested_user_id or stored_user_id))
    return replace(state, user_id=user_id), user_id


def _update_consent_button(consent_checked: bool, state: GuidedState):
    return gr.update(interactive=bool(consent_checked or state.consent_accepted))


def _start_label(args: GradioAppArgs) -> str:
    return START_SCENARIO_LABEL if args.ui == "guided" else REVIEW_SCENARIO_LABEL


def _restart_label(args: GradioAppArgs) -> str:
    return (
        RESTART_CURRENT_CALL_LABEL
        if args.ui == "guided"
        else RESTART_CURRENT_CALL_LABEL + " / Back to Scenario Selection"
    )


def _final_restart_label(args: GradioAppArgs) -> str:
    return (
        START_NEW_GUIDED_SESSION_LABEL
        if args.ui == "guided"
        else START_NEW_SESSION_LABEL
    )


def _continue_label(args: GradioAppArgs, *, is_final: bool) -> str:
    if is_final:
        return CONTINUE_TO_SURVEY_LABEL
    return CONTINUE_TO_NEXT_SCENARIO_LABEL


def _selected_step_id(
    state: GuidedState,
    *,
    consent_required: bool,
    has_selection: bool,
    selection_step_id: int,
    first_scenario_step_id: int,
    survey_step_id: int,
) -> int:
    if consent_required and state.phase == "consent":
        return 1
    if has_selection and state.phase == "selection":
        return selection_step_id
    if state.phase == "survey":
        return survey_step_id
    return first_scenario_step_id + state.active_index


def build_session_demo(
    session_manager: SessionManager,
    args: GradioAppArgs,
) -> gr.Blocks:
    """Build one shared outer session wrapper for standard and guided mode."""
    available_scenarios, loaded_scenarios = _resolve_loaded_scenarios(args)
    consent_markdown = resolve_consent_markdown(args)
    consent_required = consent_markdown is not None
    initial_state = _build_initial_state(
        args=args,
        loaded_scenarios=loaded_scenarios,
        consent_required=consent_required,
    )

    has_selection = args.ui == "standard"
    selection_step_id = 2 if consent_required else 1
    first_scenario_step_id = selection_step_id + (1 if has_selection else 0)
    survey_step_id = first_scenario_step_id + len(loaded_scenarios)

    demo = gr.Blocks(title="Emergency Call Simulator", fill_height=True)
    with demo:
        gr.Markdown("## Emergency Call Simulator")
        state_store = gr.State(value=initial_state)
        user_id_state = gr.State(value="")
        initial_events_state = gr.State(value=[])
        initial_completion_state = gr.State(value=False)
        turn_message_state = gr.State(value="")
        turn_completion_state = gr.State(value=False)

        with gr.Walkthrough(
            selected=1,
            elem_id="session_walkthrough",
            elem_classes=["guided-outer-walkthrough"],
        ) as walkthrough:
            steps: list[gr.Step] = []
            if consent_required:
                with gr.Step("Consent", id=1) as consent_step:
                    steps.append(consent_step)
                    gr.Markdown("", visible=False)

            if has_selection:
                with gr.Step(
                    "Scenario Selection", id=selection_step_id
                ) as selection_step:
                    steps.append(selection_step)
                    gr.Markdown("", visible=False)

            for index in range(len(loaded_scenarios)):
                with gr.Step(
                    f"Scenario {index + 1}", id=first_scenario_step_id + index
                ) as scenario_step:
                    steps.append(scenario_step)
                    gr.Markdown("", visible=False)

            with gr.Step("Survey", id=survey_step_id) as survey_step:
                steps.append(survey_step)
                gr.Markdown("", visible=False)

        with gr.Row(equal_height=False):
            with gr.Column(scale=8):
                consent = (
                    ConsentStepView.render(
                        consent_markdown=consent_markdown,
                        checkbox_label=DEFAULT_CONSENT_CHECKBOX_LABEL,
                        continue_label=REVIEW_SCENARIO_LABEL,
                        visible=_mounted_visibility(initial_state.phase == "consent"),
                    )
                    if consent_required
                    else None
                )

                selection = (
                    SelectionStepView.render(
                        intro_markdown=_selection_step_intro(),
                        scenario_choices=list(available_scenarios),
                        scenario_value=loaded_scenarios[0],
                        picker_interactive=not args.random_scenario,
                        next_label=REVIEW_SCENARIO_LABEL,
                        visible=_mounted_visibility(initial_state.phase == "selection"),
                    )
                    if has_selection
                    else None
                )

                scenario_flow = ScenarioFlowView.render(
                    intro_markdown=_build_scenario_intro_markdown(),
                    start_label=_start_label(args),
                    reset_label=_restart_label(args),
                    completion_markdown=_build_scenario_complete_markdown(
                        args=args,
                        is_final=len(loaded_scenarios) == 1,
                    ),
                    advance_label=_continue_label(
                        args,
                        is_final=len(loaded_scenarios) == 1,
                    ),
                    start_interactive=not consent_required,
                    visible=_mounted_visibility(
                        _is_scenario_phase(initial_state.phase)
                    ),
                )

                if args.enable_asr:
                    with scenario_flow.call_group:
                        asr_pane = AsrPane.render()
                else:
                    asr_pane = None

                survey = SurveyStepView.render(
                    survey=DEFAULT_SURVEY,
                    intro_markdown=_build_survey_intro_markdown(len(loaded_scenarios)),
                    feedback_label="Additional feedback (optional)",
                    feedback_placeholder="Share any comments about this session...",
                    submit_label="Submit Survey",
                    thanks_markdown="### ✓ Thank you!",
                    restart_label=_final_restart_label(args),
                    visible=initial_state.phase == "survey",
                )

            with gr.Column(scale=4):
                context = ContextPanelView.render(
                    scenario_markdown=_context_markdown(
                        initial_state,
                        args=args,
                        consent_required=consent_required,
                    ),
                )

        session_info_display = gr.Markdown(
            session_info_text(args, scenario=_current_scenario_name(initial_state)),
            visible=args.is_debug_enabled(),
            elem_id="session_info_display",
        )

    ui = SessionShellUI(
        demo=demo,
        walkthrough=walkthrough,
        steps=steps,
        state=state_store,
        user_id_state=user_id_state,
        session_info_display=session_info_display,
        context=context,
        consent=consent,
        selection=selection,
        scenario_flow=scenario_flow,
        survey=survey,
        asr_pane=asr_pane,
    )
    walkthrough_controller = WalkthroughController(ui.walkthrough)
    context_outputs = [*ui.context.outputs(), ui.session_info_display]
    survey_outputs = ui.survey.outputs()
    phase_outputs = [
        ui.state,
        *walkthrough_controller.outputs(),
        *context_outputs,
        *(ui.consent.outputs() if ui.consent is not None else []),
        *(ui.selection.outputs() if ui.selection is not None else []),
        *ui.scenario_flow.outputs(),
        *survey_outputs,
    ]
    phase_outputs_without_state = phase_outputs[1:]

    on_context_update = partial(
        _context_updates,
        context=ui.context,
        session_info_display=ui.session_info_display,
        args=args,
        consent_required=consent_required,
    )

    def render_phase_updates(state: GuidedState) -> dict[object, object]:
        is_final = state.active_index >= len(state.scenario_names) - 1
        updates = on_context_update(state)
        updates |= dict(
            zip(
                walkthrough_controller.outputs(),
                walkthrough_controller.walkthrough_update(
                    selected_step_id=_selected_step_id(
                        state,
                        consent_required=consent_required,
                        has_selection=has_selection,
                        selection_step_id=selection_step_id,
                        first_scenario_step_id=first_scenario_step_id,
                        survey_step_id=survey_step_id,
                    ),
                ),
                strict=True,
            )
        )

        if ui.consent is not None:
            consent_visible = state.phase == "consent"
            updates[ui.consent.container] = gr.update(
                visible=_mounted_visibility(consent_visible)
            )
            updates[ui.consent.consent_checkbox] = gr.update(
                value=state.consent_accepted,
                interactive=consent_visible and not state.consent_accepted,
            )
            updates[ui.consent.continue_button] = gr.update(
                value=REVIEW_SCENARIO_LABEL,
                interactive=consent_visible and state.consent_accepted,
            )

        if ui.selection is not None:
            selection_visible = state.phase == "selection"
            updates[ui.selection.container] = gr.update(
                visible=_mounted_visibility(selection_visible)
            )
            updates[ui.selection.intro_markdown] = gr.update(
                value=_selection_step_intro(),
            )
            updates[ui.selection.scenario_picker] = gr.update(
                value=_current_scenario_name(state),
                interactive=selection_visible and not args.random_scenario,
            )
            updates[ui.selection.next_button] = gr.update(
                value=REVIEW_SCENARIO_LABEL,
                interactive=selection_visible,
            )

        if _is_scenario_phase(state.phase):
            common_flow_kwargs = {
                "intro_markdown": _build_scenario_intro_markdown(),
                "start_label": _start_label(args),
                "continue_label": _continue_label(args, is_final=is_final),
                "restart_label": _restart_label(args),
            }
            if state.phase == "brief":
                updates |= ui.scenario_flow.brief_updates(**common_flow_kwargs)
            elif state.phase == "starting":
                updates |= ui.scenario_flow.starting_updates(**common_flow_kwargs)
            elif state.phase == "call":
                updates |= ui.scenario_flow.call_updates(
                    continue_label=common_flow_kwargs["continue_label"],
                    restart_label=common_flow_kwargs["restart_label"],
                )
            else:
                updates |= ui.scenario_flow.complete_updates(
                    completion_markdown=_build_scenario_complete_markdown(
                        args=args,
                        is_final=is_final,
                    ),
                    continue_label=common_flow_kwargs["continue_label"],
                    restart_label=common_flow_kwargs["restart_label"],
                )
        else:
            updates |= ui.scenario_flow.hide_updates()

        updates |= ui.survey.phase_updates(visible=state.phase == "survey")
        return updates

    def render_phase_tuple(
        state: GuidedState,
        output_components: list[object],
    ) -> tuple[object, ...]:
        updates = {ui.state: state}
        updates |= render_phase_updates(state)
        return ordered_component_updates(output_components, updates)

    with ui.demo:
        ui.demo.load(
            _initialize_user_id,
            inputs=[ui.user_id_state, ui.state],
            outputs=[ui.state, ui.user_id_state],
        ).then(
            lambda state: ordered_component_updates(
                phase_outputs_without_state,
                render_phase_updates(state),
            ),
            inputs=[ui.state],
            outputs=phase_outputs_without_state,
            queue=False,
            show_progress="hidden",
        )

        if ui.consent is not None:
            ui.consent.consent_checkbox.change(
                _update_consent_button,
                inputs=[ui.consent.consent_checkbox, ui.state],
                outputs=[ui.consent.continue_button],
                queue=False,
            )

            def on_consent_continue(
                state: GuidedState,
            ) -> tuple[object, ...]:
                updated = replace(
                    state,
                    phase=_after_consent_phase(args=args),
                    consent_accepted=True,
                )
                return render_phase_tuple(updated, phase_outputs)

            ui.consent.continue_button.click(
                on_consent_continue,
                inputs=[ui.state],
                outputs=phase_outputs,
                queue=False,
                show_progress="hidden",
            )

        if ui.selection is not None:

            def on_selection_change(
                selected_scenario: str | None,
                state: GuidedState,
            ) -> tuple[object, ...]:
                updated_state = _apply_standard_selection(
                    state, selected_scenario, args=args
                )
                return render_phase_tuple(updated_state, phase_outputs)

            ui.selection.scenario_picker.input(
                on_selection_change,
                inputs=[ui.selection.scenario_picker, ui.state],
                outputs=phase_outputs,
                queue=False,
                show_progress="hidden",
            )

            def on_selection_continue(
                state: GuidedState,
                selected_scenario: str | None,
            ) -> tuple[object, ...]:
                updated_state = replace(
                    _apply_standard_selection(state, selected_scenario, args=args),
                    phase="starting",
                )
                return render_phase_tuple(updated_state, phase_outputs)

            selection_start_event = ui.selection.next_button.click(
                on_selection_continue,
                inputs=[ui.state, ui.selection.scenario_picker],
                outputs=phase_outputs,
                queue=False,
                show_progress="hidden",
            )

        ui.scenario_flow.call.bind_send_interactivity()

        def on_prepare_start(
            state: GuidedState,
            consent_checked: bool = False,
        ) -> tuple[object, ...]:
            updated_state = begin_guided_start(
                state,
                scenario_index=state.active_index,
                consent_checked=consent_checked if state.active_index == 0 else False,
            )
            return render_phase_tuple(updated_state, phase_outputs)

        async def on_start_failure(
            state: GuidedState,
        ) -> tuple[object, ...]:
            updated_state = start_failure_state(state)
            return render_phase_tuple(updated_state, phase_outputs)

        async def on_start_scenario(
            state: GuidedState,
            stored_user_id: str,
        ):
            previous_run_session_id = state.run_session_id
            (
                updated_state,
                events,
                completion,
                resolved_user_id,
            ) = await start_guided_session(
                state,
                stored_user_id,
                scenario_index=state.active_index,
                session_manager=session_manager,
                args=args,
            )
            if (
                consent_required
                and updated_state.active_session is not None
                and updated_state.consent_accepted
                and not args.should_skip_policy_calls()
                and (args.ui == "standard" or previous_run_session_id is None)
            ):
                await log_consent_acceptance(
                    session_manager=session_manager,
                    session_id=updated_state.active_session.session_id,
                    ui_mode=args.ui,
                    scenario_name=updated_state.active_session.scenario_name,
                )
            return {
                ui.state: updated_state,
                ui.user_id_state: resolved_user_id,
                initial_events_state: events,
                initial_completion_state: completion,
            } | on_context_update(updated_state)

        def on_finalize_start(
            state: GuidedState,
            is_complete: bool,
        ) -> tuple[object, ...]:
            updated_state = finalize_started_call(state, is_complete=is_complete)
            return render_phase_tuple(updated_state, phase_outputs)

        def bind_start_chain(trigger_event) -> None:
            start_session_event = trigger_event.then(
                on_start_scenario,
                inputs=[ui.state, ui.user_id_state],
                outputs=[
                    ui.state,
                    ui.user_id_state,
                    initial_events_state,
                    initial_completion_state,
                    *context_outputs,
                ],
                show_progress="hidden",
            )
            start_session_event.failure(
                on_start_failure,
                inputs=[ui.state],
                outputs=phase_outputs,
                queue=False,
                show_progress="hidden",
            )
            start_greeting_event = start_session_event.then(
                stream_backend_events,
                inputs=[ui.scenario_flow.call.chatbot, initial_events_state],
                outputs=[ui.scenario_flow.call.chatbot],
            )
            start_greeting_event.then(
                lambda: [],
                outputs=[initial_events_state],
                queue=False,
                show_progress="hidden",
            )
            start_greeting_event.then(
                on_finalize_start,
                inputs=[ui.state, initial_completion_state],
                outputs=phase_outputs,
                queue=False,
                show_progress="hidden",
            ).then(
                lambda: False,
                outputs=[initial_completion_state],
                queue=False,
                show_progress="hidden",
            )

        start_prepare_event = ui.scenario_flow.intro.start_button.click(
            on_prepare_start,
            inputs=[
                ui.state,
                *([ui.consent.consent_checkbox] if ui.consent is not None else []),
            ],
            outputs=phase_outputs,
            queue=False,
            show_progress="hidden",
        )
        bind_start_chain(start_prepare_event)
        if ui.selection is not None:
            bind_start_chain(selection_start_event)

        capture_turn_outputs = [
            turn_message_state,
            ui.scenario_flow.call.chatbot,
            ui.scenario_flow.call.input_box,
            ui.scenario_flow.call.send,
            ui.scenario_flow.call.restart_button,
            ui.scenario_flow.call.status_markdown,
        ]

        submit_event = gr.on(
            triggers=ui.scenario_flow.call.submit_triggers(),
            fn=capture_guided_user_submit,
            inputs=[ui.scenario_flow.call.input_box, ui.scenario_flow.call.chatbot],
            outputs=capture_turn_outputs,
            queue=False,
            show_progress="hidden",
        )

        async def on_stream_turn(
            history: list[ChatMessage],
            user_message: str,
            state: GuidedState,
        ):
            async for payload in stream_guided_turn(
                history,
                user_message,
                state,
                scenario_index=state.active_index,
                session_manager=session_manager,
                skip_policy_calls=args.should_skip_policy_calls(),
            ):
                yield payload

        stream_turn_event = submit_event.then(
            on_stream_turn,
            inputs=[ui.scenario_flow.call.chatbot, turn_message_state, ui.state],
            outputs=[ui.scenario_flow.call.chatbot, turn_completion_state],
        )
        stream_turn_event.then(
            lambda state, is_complete: render_phase_tuple(
                finalize_guided_turn(state, is_complete=is_complete),
                phase_outputs,
            ),
            inputs=[ui.state, turn_completion_state],
            outputs=phase_outputs,
            queue=False,
            show_progress="hidden",
        ).then(
            lambda: "",
            outputs=[turn_message_state],
            queue=False,
            show_progress="hidden",
        ).then(
            lambda: False,
            outputs=[turn_completion_state],
            queue=False,
            show_progress="hidden",
        )

        async def on_like(
            history: list[ChatMessage],
            state: GuidedState,
            like_data: gr.LikeData,
        ) -> None:
            await persist_chat_feedback(
                history,
                state.active_session,
                like_data,
                session_manager=session_manager,
                skip_policy_calls=args.should_skip_policy_calls(),
            )

        ui.scenario_flow.call.chatbot.like(
            on_like,
            inputs=[ui.scenario_flow.call.chatbot, ui.state],
            outputs=[],
            queue=False,
            show_progress="hidden",
        )

        async def on_restart_scenario(
            state: GuidedState,
        ) -> tuple[object, ...]:
            await end_active_session(
                state.active_session,
                session_manager=session_manager,
                skip_policy_calls=args.should_skip_policy_calls(),
            )
            if args.ui == "guided" and state.active_session is not None:
                archive_guided_runtime_artifacts(
                    session_manager=session_manager,
                    run_session_id=state.run_session_id,
                    scenario_index=state.active_index,
                    scenario_name=state.active_session.scenario_name,
                )
            if args.ui == "guided":
                updated_state = restart_current_call(state)
            else:
                updated_state = replace(
                    state,
                    active_session=None,
                    phase="selection",
                )
            return render_phase_tuple(updated_state, phase_outputs)

        ui.scenario_flow.call.restart_button.click(
            on_restart_scenario,
            inputs=[ui.state],
            outputs=phase_outputs,
            queue=False,
            show_progress="hidden",
        )

        async def on_advance_scenario(
            state: GuidedState,
        ) -> tuple[object, ...]:
            if state.phase != "complete":
                return render_phase_tuple(state, phase_outputs)

            is_final = state.active_index >= len(state.scenario_names) - 1
            if is_final:
                if args.ui == "guided" and state.active_session is not None:
                    archive_guided_runtime_artifacts(
                        session_manager=session_manager,
                        run_session_id=state.run_session_id,
                        scenario_index=state.active_index,
                        scenario_name=state.active_session.scenario_name,
                    )
                updated_state = enter_guided_survey(state)
                return render_phase_tuple(updated_state, phase_outputs)

            await end_active_session(
                state.active_session,
                session_manager=session_manager,
                skip_policy_calls=args.should_skip_policy_calls(),
            )
            if args.ui == "guided" and state.active_session is not None:
                archive_guided_runtime_artifacts(
                    session_manager=session_manager,
                    run_session_id=state.run_session_id,
                    scenario_index=state.active_index,
                    scenario_name=state.active_session.scenario_name,
                )
            updated_state = advance_to_next_scenario(state)
            return render_phase_tuple(updated_state, phase_outputs)

        continue_event = ui.scenario_flow.call.continue_button.click(
            on_advance_scenario,
            inputs=[ui.state],
            outputs=phase_outputs,
            queue=False,
            show_progress="hidden",
        )

        def on_refresh_survey_controls(state: GuidedState) -> tuple[object, ...]:
            if state.phase != "survey":
                return ordered_component_updates(survey_outputs, {})
            return ordered_component_updates(survey_outputs, ui.survey.reset_updates())

        continue_event.then(
            on_refresh_survey_controls,
            inputs=[ui.state],
            outputs=survey_outputs,
            queue=False,
            show_progress="hidden",
        )

        if ui.asr_pane is not None:
            wire_asr_to_chat_input(
                pane=ui.asr_pane,
                input_box=ui.scenario_flow.call.input_box,
                send_button=ui.scenario_flow.call.send,
                runtime=StreamingAsr(model_name=args.asr_model),
                concurrency_id=f"{args.ui}_asr_stream",
            )

        ui.survey.bind_validation(
            fn=check_all_survey_answers,
            concurrency_id=f"{args.ui}_survey_validation",
        )

        save_survey = partial(
            save_guided_survey if args.ui == "guided" else _save_standard_survey,
            session_manager=session_manager,
            skip_policy_calls=args.should_skip_policy_calls(),
        )
        survey_submit_event = ui.survey.survey_submit.click(
            save_survey,
            inputs=[ui.state, ui.survey.survey_feedback, *ui.survey.survey_radios],
            outputs=[ui.state, *survey_outputs],
            show_progress="hidden",
        )

        assert ui.survey.restart_button is not None

        survey_submit_event.then(
            lambda: (
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True, interactive=True),
            ),
            outputs=[
                ui.survey.completion_group,
                ui.survey.survey_thanks,
                ui.survey.restart_button,
            ],
            queue=False,
            show_progress="hidden",
        ).then(
            on_context_update,
            inputs=[ui.state],
            outputs=context_outputs,
            queue=False,
            show_progress="hidden",
        )

        async def on_restart_session(
            state: GuidedState,
            stored_user_id: str,
            selected_scenario: str | None = None,
        ) -> tuple[object, ...]:
            await end_active_session(
                state.active_session,
                session_manager=session_manager,
                skip_policy_calls=args.should_skip_policy_calls(),
            )
            updated_state = _reset_full_state(
                state,
                stored_user_id,
                selected_scenario,
                args=args,
                consent_required=consent_required,
            )
            updates = {ui.state: updated_state}
            updates |= render_phase_updates(updated_state)
            updates |= ui.survey.reset_updates(visible=False)
            return ordered_component_updates(phase_outputs, updates)

        ui.survey.restart_button.click(
            on_restart_session,
            inputs=[
                ui.state,
                ui.user_id_state,
                *([ui.selection.scenario_picker] if ui.selection is not None else []),
            ],
            outputs=phase_outputs,
            queue=False,
            show_progress="hidden",
        )

    return ui.demo
