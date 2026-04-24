"""Reusable Gradio view objects for the adapter."""

from __future__ import annotations

from dataclasses import dataclass

import gradio as gr

from ems_prepared.adapters.gradio.survey import SurveyConfig

_UNSET = object()


@dataclass(slots=True)
class ContextPanelView:
    """Persistent context panel shared across the app shell."""

    overview_display: gr.HTML | None
    scenario_markdown: gr.Markdown

    @classmethod
    def render(
        cls,
        *,
        scenario_markdown: str,
        heading_markdown: str = "### Scenario Brief",
        overview_html: str | None = None,
    ) -> ContextPanelView:
        """Render the shared context panel."""
        overview_display = gr.HTML(overview_html) if overview_html is not None else None
        gr.Markdown(heading_markdown)
        scenario_view = gr.Markdown(scenario_markdown)
        return cls(
            overview_display=overview_display,
            scenario_markdown=scenario_view,
        )

    def outputs(self) -> list[object]:
        """Return the mutable context components."""
        outputs: list[object] = [self.scenario_markdown]
        if self.overview_display is not None:
            outputs.insert(0, self.overview_display)
        return outputs

    def updates(
        self,
        *,
        scenario_markdown: str | object = _UNSET,
        overview_html: str | object = _UNSET,
    ) -> dict[object, object]:
        """Build component-keyed updates for the context panel."""
        updates: dict[object, object] = {}
        if scenario_markdown is not _UNSET:
            updates[self.scenario_markdown] = gr.update(value=scenario_markdown)
        if self.overview_display is not None and overview_html is not _UNSET:
            updates[self.overview_display] = gr.update(value=overview_html)
        return updates


@dataclass(slots=True)
class IntroStepView:
    """Intro step view for either standard or guided mode."""

    intro_markdown: gr.Markdown
    start_button: gr.Button

    @classmethod
    def render(
        cls,
        *,
        intro_markdown: str,
        start_label: str,
        button_variant: str = "primary",
    ) -> IntroStepView:
        """Render the intro step contents."""
        intro_view = gr.Markdown(intro_markdown)
        start_button = gr.Button(start_label, variant=button_variant, size="lg")
        return cls(
            intro_markdown=intro_view,
            start_button=start_button,
        )

    def outputs(self) -> list[object]:
        """Return the mutable intro controls."""
        return [self.intro_markdown, self.start_button]


@dataclass(slots=True)
class CallStepView:
    """Reusable chat/call pane."""

    status_markdown: gr.Markdown
    chatbot: gr.Chatbot
    input_box: gr.Textbox
    send: gr.Button
    reset: gr.Button
    continue_button: gr.Button | None
    finish_button: gr.Button | None

    @classmethod
    def render(
        cls,
        *,
        reset_label: str,
        continue_label: str | None = None,
        finish_label: str | None = None,
    ) -> CallStepView:
        """Render the reusable chat pane."""
        status_markdown = gr.Markdown(
            "### Starting call...\n\nPlease wait while the caller connects.",
            visible=False,
            elem_classes=["guided-call-status"],
        )
        chatbot = gr.Chatbot(
            height="52vh",
            label="112",
            group_consecutive_messages=False,
        )
        with gr.Row():
            input_box = gr.Textbox(
                placeholder="Type your message and press Enter...",
                show_label=False,
                scale=4,
                max_lines=5,
                interactive=False,
            )
            send = gr.Button(
                "Send",
                variant="primary",
                scale=1,
                interactive=False,
            )
        with gr.Row():
            reset = gr.Button(reset_label, variant="secondary")
            continue_button = (
                gr.Button(continue_label, variant="primary", visible=False)
                if continue_label is not None
                else None
            )
            finish_button = (
                gr.Button(finish_label, variant="primary", visible=False)
                if finish_label is not None
                else None
            )
        return cls(
            status_markdown=status_markdown,
            chatbot=chatbot,
            input_box=input_box,
            send=send,
            reset=reset,
            continue_button=continue_button,
            finish_button=finish_button,
        )

    def submit_triggers(self) -> list[object]:
        """Return the standard text submit triggers for the pane."""
        return [self.input_box.submit, self.send.click]

    def bind_send_interactivity(self) -> None:
        """Enable send once the user has entered text."""
        self.input_box.input(
            lambda text: gr.update(interactive=bool(text and text.strip())),
            inputs=[self.input_box],
            outputs=[self.send],
        )

    def outputs(self) -> list[object]:
        """Return the mutable call controls for event wiring."""
        outputs: list[object] = [
            self.chatbot,
            self.input_box,
            self.send,
            self.reset,
        ]
        if self.continue_button is not None:
            outputs.append(self.continue_button)
        if self.finish_button is not None:
            outputs.append(self.finish_button)
        return outputs

    def control_updates(
        self,
        *,
        input_enabled: bool,
        reset_enabled: bool,
        continue_visible: bool = False,
        finish_visible: bool = False,
    ) -> dict[object, object]:
        """Build common control updates for the pane."""
        updates: dict[object, object] = {
            self.input_box: gr.update(
                interactive=input_enabled,
                value="",
                autofocus=input_enabled,
            ),
            self.send: gr.update(interactive=False),
            self.reset: gr.update(interactive=reset_enabled),
        }
        if self.continue_button is not None:
            updates[self.continue_button] = gr.update(
                visible=continue_visible,
                interactive=continue_visible,
            )
        if self.finish_button is not None:
            updates[self.finish_button] = gr.update(
                visible=finish_visible,
                interactive=finish_visible,
            )
        return updates


@dataclass(slots=True)
class GuidedCallView:
    """Guided-mode chat pane with low-chrome presentation."""

    status_markdown: gr.Markdown
    chatbot: gr.Chatbot
    input_group: gr.Column
    input_box: gr.Textbox
    send: gr.Button
    completion_text: str
    completion_markdown: gr.Markdown
    continue_button: gr.Button
    restart_button: gr.Button

    @classmethod
    def render(
        cls,
        *,
        reset_label: str,
        completion_markdown: str,
        continue_label: str,
    ) -> GuidedCallView:
        """Render the guided chat pane."""
        status_markdown = gr.Markdown(
            "### Starting call...\n\nPlease wait while the caller connects.",
            visible=False,
            elem_classes=["guided-call-status"],
        )
        chatbot = gr.Chatbot(
            height="52vh",
            show_label=False,
            container=False,
            buttons=[],
            group_consecutive_messages=False,
        )
        with gr.Column() as input_group:
            with gr.Row():
                input_box = gr.Textbox(
                    placeholder="Type your message and press Enter...",
                    show_label=False,
                    scale=4,
                    max_lines=5,
                    interactive=False,
                )
                send = gr.Button(
                    "Send",
                    variant="primary",
                    scale=1,
                    interactive=False,
                )
        completion_view = gr.Markdown("")
        continue_button = gr.Button(
            continue_label,
            variant="primary",
            interactive=False,
            elem_classes=["guided-continue-button"],
        )
        restart_button = gr.Button(reset_label, variant="secondary", interactive=False)
        return cls(
            status_markdown=status_markdown,
            chatbot=chatbot,
            input_group=input_group,
            input_box=input_box,
            send=send,
            completion_text=completion_markdown,
            completion_markdown=completion_view,
            continue_button=continue_button,
            restart_button=restart_button,
        )

    def submit_triggers(self) -> list[object]:
        """Return the standard text submit triggers for the pane."""
        return [self.input_box.submit, self.send.click]

    def bind_send_interactivity(self) -> None:
        """Enable send once the user has entered text."""
        self.input_box.input(
            lambda text: gr.update(interactive=bool(text and text.strip())),
            inputs=[self.input_box],
            outputs=[self.send],
        )

    def outputs(self) -> list[object]:
        """Return the mutable guided chat controls."""
        return [
            self.status_markdown,
            self.chatbot,
            self.input_group,
            self.input_box,
            self.send,
            self.completion_markdown,
            self.continue_button,
            self.restart_button,
        ]


@dataclass(slots=True)
class ScenarioFlowView:
    """Per-scenario guided flow with a brief pane and one persistent call pane."""

    brief_group: gr.Column
    call_group: gr.Column
    intro: IntroStepView
    call: GuidedCallView

    @classmethod
    def render(
        cls,
        *,
        intro_markdown: str,
        start_label: str,
        reset_label: str,
        completion_markdown: str,
        advance_label: str,
    ) -> ScenarioFlowView:
        """Render one guided scenario flow."""
        with gr.Column(
            visible=True,
            elem_classes=["guided-phase", "guided-phase-brief"],
        ) as brief_group:
            intro = IntroStepView.render(
                intro_markdown=intro_markdown,
                start_label=start_label,
            )
        with gr.Column(
            visible=False,
            elem_classes=["guided-phase", "guided-phase-call"],
        ) as call_group:
            call = GuidedCallView.render(
                reset_label=reset_label,
                completion_markdown=completion_markdown,
                continue_label=advance_label,
            )
        return cls(
            brief_group=brief_group,
            call_group=call_group,
            intro=intro,
            call=call,
        )

    def outputs(self) -> list[object]:
        """Return the mutable guided scenario components."""
        return [
            self.brief_group,
            self.call_group,
            *self.intro.outputs(),
            *self.call.outputs(),
        ]

    def brief_updates(self) -> dict[object, object]:
        """Reset the flow to the briefing phase."""
        return {
            self.brief_group: gr.Column(visible=True),
            self.call_group: gr.Column(visible=False),
            self.intro.start_button: gr.update(interactive=True),
            self.call.status_markdown: gr.update(visible=False, value=""),
            self.call.chatbot: [],
            self.call.input_group: gr.Column(visible=True),
            self.call.input_box: gr.update(
                interactive=False,
                value="",
                autofocus=False,
            ),
            self.call.send: gr.update(interactive=False),
            self.call.completion_markdown: gr.Markdown(value=""),
            self.call.restart_button: gr.update(interactive=False),
            self.call.continue_button: gr.Button(
                value=self.call.continue_button.value,
                interactive=False,
            ),
        }

    def starting_updates(self) -> dict[object, object]:
        """Show the call phase immediately while the backend session starts."""
        return {
            self.brief_group: gr.Column(visible=False),
            self.call_group: gr.Column(visible=True),
            self.intro.start_button: gr.update(interactive=False),
            self.call.status_markdown: gr.update(visible=False, value=""),
            self.call.chatbot: [],
            self.call.input_group: gr.Column(visible=True),
            self.call.input_box: gr.update(
                interactive=False,
                value="",
                autofocus=False,
            ),
            self.call.send: gr.update(interactive=False),
            self.call.completion_markdown: gr.Markdown(value=""),
            self.call.restart_button: gr.update(interactive=False),
            self.call.continue_button: gr.Button(
                value=self.call.continue_button.value,
                interactive=False,
            ),
        }

    def call_updates(self) -> dict[object, object]:
        """Show the live call phase with active input controls."""
        return {
            self.brief_group: gr.Column(visible=False),
            self.call_group: gr.Column(visible=True),
            self.call.status_markdown: gr.update(visible=False, value=""),
            self.call.input_group: gr.Column(visible=True),
            self.call.input_box: gr.update(interactive=True, value="", autofocus=True),
            self.call.send: gr.update(interactive=False),
            self.call.completion_markdown: gr.Markdown(value=""),
            self.call.restart_button: gr.update(interactive=True),
            self.call.continue_button: gr.Button(
                value=self.call.continue_button.value,
                interactive=False,
            ),
        }

    def processing_updates(self) -> dict[object, object]:
        """Disable chat controls while a backend turn is running."""
        return {
            self.brief_group: gr.Column(visible=False),
            self.call_group: gr.Column(visible=True),
            self.call.status_markdown: gr.update(visible=False, value=""),
            self.call.input_group: gr.Column(visible=True),
            self.call.input_box: gr.update(
                interactive=False,
                value="",
                autofocus=False,
            ),
            self.call.send: gr.update(interactive=False),
            self.call.completion_markdown: gr.Markdown(value=""),
            self.call.restart_button: gr.update(interactive=False),
            self.call.continue_button: gr.Button(
                value=self.call.continue_button.value,
                interactive=False,
            ),
        }

    def streaming_updates(self) -> dict[object, object]:
        """Hide the waiting indicator while streamed caller output is rendering."""
        return {
            self.brief_group: gr.Column(visible=False),
            self.call_group: gr.Column(visible=True),
            self.call.status_markdown: gr.update(visible=False, value=""),
            self.call.input_group: gr.Column(visible=True),
            self.call.input_box: gr.update(
                interactive=False,
                value="",
                autofocus=False,
            ),
            self.call.send: gr.update(interactive=False),
            self.call.completion_markdown: gr.Markdown(value=""),
            self.call.restart_button: gr.update(interactive=False),
            self.call.continue_button: gr.Button(
                value=self.call.continue_button.value,
                interactive=False,
            ),
        }

    def call_error_updates(self) -> dict[object, object]:
        """Keep the call visible but require a restart when the session fails."""
        return {
            self.brief_group: gr.Column(visible=False),
            self.call_group: gr.Column(visible=True),
            self.call.status_markdown: gr.update(visible=False, value=""),
            self.call.input_group: gr.Column(visible=False),
            self.call.input_box: gr.update(interactive=False, value="", autofocus=False),
            self.call.send: gr.update(interactive=False),
            self.call.completion_markdown: gr.Markdown(value=""),
            self.call.restart_button: gr.update(interactive=True),
            self.call.continue_button: gr.Button(
                value=self.call.continue_button.value,
                interactive=False,
            ),
        }

    def complete_updates(self) -> dict[object, object]:
        """Show completion controls while keeping the chat transcript visible."""
        return {
            self.brief_group: gr.Column(visible=False),
            self.call_group: gr.Column(visible=True),
            self.call.status_markdown: gr.update(visible=False, value=""),
            self.call.input_group: gr.Column(visible=False),
            self.call.input_box: gr.update(interactive=False, value="", autofocus=False),
            self.call.send: gr.update(interactive=False),
            self.call.completion_markdown: gr.Markdown(value=self.call.completion_text),
            self.call.restart_button: gr.update(interactive=True),
            self.call.continue_button: gr.Button(
                value=self.call.continue_button.value,
                interactive=True,
            ),
        }


@dataclass(slots=True)
class SurveyStepView:
    """Reusable survey pane."""

    container: gr.Column
    survey_radios: list[gr.Radio]
    survey_feedback: gr.Textbox
    survey_submit: gr.Button
    completion_group: gr.Column
    survey_thanks: gr.Markdown
    restart_button: gr.Button | None

    @classmethod
    def render(
        cls,
        *,
        survey: SurveyConfig,
        intro_markdown: str,
        feedback_label: str,
        feedback_placeholder: str,
        submit_label: str,
        thanks_markdown: str,
        restart_label: str | None = None,
    ) -> SurveyStepView:
        """Render the survey pane."""
        with gr.Column() as container:
            gr.Markdown(intro_markdown)
            survey_radios: list[gr.Radio] = []
            for question in survey.questions:
                survey_radios.append(
                    gr.Radio(
                        choices=survey.get_choices(),
                        label=question.text,
                        type="value",
                        interactive=True,
                    )
                )

            survey_feedback = gr.Textbox(
                label=feedback_label,
                placeholder=feedback_placeholder,
                lines=2,
                max_lines=5,
                interactive=True,
            )
            survey_submit = gr.Button(submit_label, variant="primary", interactive=False)
            with gr.Column(visible=False) as completion_group:
                survey_thanks = gr.Markdown(thanks_markdown)
                restart_button = (
                    gr.Button(restart_label, variant="primary")
                    if restart_label is not None
                    else None
                )
        return cls(
            container=container,
            survey_radios=survey_radios,
            survey_feedback=survey_feedback,
            survey_submit=survey_submit,
            completion_group=completion_group,
            survey_thanks=survey_thanks,
            restart_button=restart_button,
        )

    def bind_validation(self, *, fn, concurrency_id: str) -> None:
        """Attach shared survey answer validation."""
        gr.on(
            triggers=[radio.change for radio in self.survey_radios],
            fn=fn,
            inputs=self.survey_radios,
            outputs=self.survey_submit,
            queue=False,
            trigger_mode="always_last",
            concurrency_limit=1,
            concurrency_id=concurrency_id,
        )

    def outputs(self) -> list[object]:
        """Return all mutable survey controls."""
        outputs: list[object] = [
            *self.survey_radios,
            self.survey_feedback,
            self.survey_submit,
            self.completion_group,
            self.survey_thanks,
        ]
        if self.restart_button is not None:
            outputs.append(self.restart_button)
        return outputs

    def reset_updates(self) -> dict[object, object]:
        """Reset the survey to its initial interactive state."""
        updates: dict[object, object] = {
            self.survey_feedback: gr.update(value="", interactive=True),
            self.survey_submit: gr.update(interactive=False),
            self.completion_group: gr.Column(visible=False),
        }
        for radio in self.survey_radios:
            updates[radio] = gr.update(value=None, interactive=True)
        return updates


def build_context_panel(
    *,
    scenario_markdown: str,
    heading_markdown: str = "### Scenario Brief",
    overview_html: str | None = None,
) -> ContextPanelView:
    """Compatibility wrapper for rendering the shared context panel."""
    return ContextPanelView.render(
        scenario_markdown=scenario_markdown,
        heading_markdown=heading_markdown,
        overview_html=overview_html,
    )



def build_intro_section(
    *,
    intro_markdown: str,
    start_label: str,
    button_variant: str = "primary",
) -> IntroStepView:
    """Compatibility wrapper for rendering an intro step."""
    return IntroStepView.render(
        intro_markdown=intro_markdown,
        start_label=start_label,
        button_variant=button_variant,
    )



def build_chat_section(
    *,
    reset_label: str,
    continue_label: str | None = None,
    finish_label: str | None = None,
) -> CallStepView:
    """Compatibility wrapper for rendering the shared call pane."""
    return CallStepView.render(
        reset_label=reset_label,
        continue_label=continue_label,
        finish_label=finish_label,
    )



def build_scenario_flow(
    *,
    intro_markdown: str,
    start_label: str,
    reset_label: str,
    completion_markdown: str,
    advance_label: str,
) -> ScenarioFlowView:
    """Compatibility wrapper for rendering one guided scenario flow."""
    return ScenarioFlowView.render(
        intro_markdown=intro_markdown,
        start_label=start_label,
        reset_label=reset_label,
        completion_markdown=completion_markdown,
        advance_label=advance_label,
    )



def build_survey_section(
    *,
    survey: SurveyConfig,
    intro_markdown: str,
    feedback_label: str,
    feedback_placeholder: str,
    submit_label: str,
    thanks_markdown: str,
    restart_label: str | None = None,
) -> SurveyStepView:
    """Compatibility wrapper for rendering a survey step."""
    return SurveyStepView.render(
        survey=survey,
        intro_markdown=intro_markdown,
        feedback_label=feedback_label,
        feedback_placeholder=feedback_placeholder,
        submit_label=submit_label,
        thanks_markdown=thanks_markdown,
        restart_label=restart_label,
    )
