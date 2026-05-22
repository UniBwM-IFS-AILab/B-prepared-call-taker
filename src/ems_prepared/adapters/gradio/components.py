"""Reusable Gradio view objects for guided and standard UIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import gradio as gr

from ems_prepared.adapters.gradio.survey import SurveyConfig


def _mounted_visibility(visible: bool) -> bool | Literal["hidden"]:
    """Keep components mounted while hidden to preserve event bindings."""
    return True if visible else "hidden"


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
        scenario_markdown: str | None = None,
        overview_html: str | None = None,
    ) -> dict[object, object]:
        """Build component-keyed updates for the context panel."""
        updates: dict[object, object] = {}
        if scenario_markdown is not None:
            updates[self.scenario_markdown] = gr.update(value=scenario_markdown)
        if self.overview_display is not None and overview_html is not None:
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
        button_variant: Literal[
            "primary", "secondary", "stop", "huggingface"
        ] = "primary",
        start_interactive: bool = True,
    ) -> IntroStepView:
        """Render the intro step contents."""
        intro_view = gr.Markdown(intro_markdown)
        start_button = gr.Button(
            start_label,
            variant=button_variant,
            size="lg",
            interactive=start_interactive,
        )
        return cls(
            intro_markdown=intro_view,
            start_button=start_button,
        )

    def outputs(self) -> list[object]:
        """Return the mutable intro controls."""
        return [self.intro_markdown, self.start_button]


@dataclass(slots=True)
class ConsentStepView:
    """Reusable consent step content."""

    container: gr.Column
    consent_markdown: gr.Markdown
    consent_checkbox: gr.Checkbox
    continue_button: gr.Button

    @classmethod
    def render(
        cls,
        *,
        consent_markdown: str,
        checkbox_label: str,
        continue_label: str,
        visible: bool | Literal["hidden"] = True,
    ) -> ConsentStepView:
        """Render the consent step."""
        with gr.Column(visible=visible) as container:
            markdown_view = gr.Markdown(consent_markdown)
            checkbox = gr.Checkbox(label=checkbox_label, value=False)
            continue_button = gr.Button(
                continue_label,
                variant="primary",
                size="lg",
                interactive=False,
            )
        return cls(
            container=container,
            consent_markdown=markdown_view,
            consent_checkbox=checkbox,
            continue_button=continue_button,
        )

    def outputs(self) -> list[object]:
        """Return the mutable consent controls."""
        return [self.container, self.consent_checkbox, self.continue_button]


@dataclass(slots=True)
class SelectionStepView:
    """Scenario selection step used by standard mode."""

    container: gr.Column
    intro_markdown: gr.Markdown
    scenario_picker: gr.Dropdown
    next_button: gr.Button

    @classmethod
    def render(
        cls,
        *,
        intro_markdown: str,
        scenario_choices: list[str],
        scenario_value: str | None,
        picker_interactive: bool,
        next_label: str,
        visible: bool | Literal["hidden"] = True,
    ) -> SelectionStepView:
        """Render the scenario selection step."""
        with gr.Column(visible=visible) as container:
            intro_view = gr.Markdown(intro_markdown)
            picker = gr.Dropdown(
                choices=scenario_choices,
                value=scenario_value,
                label="Scenario selection",
                interactive=picker_interactive,
                allow_custom_value=False,
                filterable=False,
            )
            next_button = gr.Button(
                next_label,
                variant="primary",
                size="lg",
                interactive=True,
            )
        return cls(
            container=container,
            intro_markdown=intro_view,
            scenario_picker=picker,
            next_button=next_button,
        )

    def outputs(self) -> list[object]:
        """Return mutable selection step components."""
        return [self.container, self.intro_markdown, self.scenario_picker, self.next_button]


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
            feedback_value=[],
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
                gr.Button(continue_label, variant="primary", visible="hidden")
                if continue_label is not None
                else None
            )
            finish_button = (
                gr.Button(finish_label, variant="primary", visible="hidden")
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
                visible=True if continue_visible else "hidden",
                interactive=continue_visible,
            )
        if self.finish_button is not None:
            updates[self.finish_button] = gr.update(
                visible=True if finish_visible else "hidden",
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
    default_completion_markdown: str
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
            feedback_value=[],
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
            default_completion_markdown=completion_markdown,
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

    container: gr.Column
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
        start_interactive: bool = True,
        visible: bool | Literal["hidden"] = True,
    ) -> ScenarioFlowView:
        """Render one guided scenario flow."""
        with gr.Column(visible=visible) as container:
            with gr.Column(
                visible=True,
                elem_classes=["guided-phase", "guided-phase-brief"],
            ) as brief_group:
                intro = IntroStepView.render(
                    intro_markdown=intro_markdown,
                    start_label=start_label,
                    start_interactive=start_interactive,
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
            container=container,
            brief_group=brief_group,
            call_group=call_group,
            intro=intro,
            call=call,
        )

    def outputs(self) -> list[object]:
        """Return the mutable guided scenario components."""
        return [
            self.container,
            self.brief_group,
            self.call_group,
            *self.intro.outputs(),
            *self.call.outputs(),
        ]

    def hide_updates(self) -> dict[object, object]:
        """Hide the shared scenario flow when another pane is active."""
        return {self.container: gr.update(visible="hidden")}

    def brief_updates(
        self,
        *,
        intro_markdown: str | None = None,
        start_label: str | None = None,
        continue_label: str | None = None,
        restart_label: str | None = None,
    ) -> dict[object, object]:
        """Reset the flow to the briefing phase."""
        resolved_intro_markdown = intro_markdown or self.intro.intro_markdown.value
        resolved_start_label = start_label or self.intro.start_button.value
        resolved_continue_label = continue_label or self.call.continue_button.value
        resolved_restart_label = restart_label or self.call.restart_button.value
        return {
            self.container: gr.update(visible=True),
            self.brief_group: gr.update(visible=True),
            self.call_group: gr.update(visible=False),
            self.intro.intro_markdown: gr.update(value=resolved_intro_markdown),
            self.intro.start_button: gr.update(value=resolved_start_label, interactive=True),
            self.call.status_markdown: gr.update(
                visible=False,
                value=self.call.status_markdown.value,
            ),
            self.call.chatbot: [],
            self.call.input_group: gr.update(visible=True),
            self.call.input_box: gr.update(
                interactive=False,
                value="",
                autofocus=False,
            ),
            self.call.send: gr.update(interactive=False),
            self.call.completion_markdown: gr.update(value=""),
            self.call.restart_button: gr.update(
                value=resolved_restart_label,
                interactive=False,
            ),
            self.call.continue_button: gr.update(
                value=resolved_continue_label,
                interactive=False,
            ),
        }

    def starting_updates(
        self,
        *,
        intro_markdown: str | None = None,
        start_label: str | None = None,
        continue_label: str | None = None,
        restart_label: str | None = None,
    ) -> dict[object, object]:
        """Show the call phase immediately while the backend session starts."""
        resolved_intro_markdown = intro_markdown or self.intro.intro_markdown.value
        resolved_start_label = start_label or self.intro.start_button.value
        resolved_continue_label = continue_label or self.call.continue_button.value
        resolved_restart_label = restart_label or self.call.restart_button.value
        return {
            self.container: gr.update(visible=True),
            self.brief_group: gr.update(visible=False),
            self.call_group: gr.update(visible=True),
            self.intro.intro_markdown: gr.update(value=resolved_intro_markdown),
            self.intro.start_button: gr.update(value=resolved_start_label, interactive=False),
            self.call.status_markdown: gr.update(
                visible=True,
                value=self.call.status_markdown.value,
            ),
            self.call.chatbot: [],
            self.call.input_group: gr.update(visible=True),
            self.call.input_box: gr.update(
                interactive=False,
                value="",
                autofocus=False,
            ),
            self.call.send: gr.update(interactive=False),
            self.call.completion_markdown: gr.update(value=""),
            self.call.restart_button: gr.update(
                value=resolved_restart_label,
                interactive=False,
            ),
            self.call.continue_button: gr.update(
                value=resolved_continue_label,
                interactive=False,
            ),
        }

    def call_updates(
        self,
        *,
        continue_label: str | None = None,
        restart_label: str | None = None,
    ) -> dict[object, object]:
        """Show the live call phase with active input controls."""
        resolved_continue_label = continue_label or self.call.continue_button.value
        resolved_restart_label = restart_label or self.call.restart_button.value
        return {
            self.container: gr.update(visible=True),
            self.brief_group: gr.update(visible=False),
            self.call_group: gr.update(visible=True),
            self.call.status_markdown: gr.update(
                visible=False,
                value=self.call.status_markdown.value,
            ),
            self.call.input_group: gr.update(visible=True),
            self.call.input_box: gr.update(interactive=True, value="", autofocus=True),
            self.call.send: gr.update(interactive=False),
            self.call.completion_markdown: gr.update(value=""),
            self.call.restart_button: gr.update(
                value=resolved_restart_label,
                interactive=True,
            ),
            self.call.continue_button: gr.update(
                value=resolved_continue_label,
                interactive=False,
            ),
        }

    def processing_updates(self) -> dict[object, object]:
        """Disable chat controls while a backend turn is running."""
        return {
            self.container: gr.update(visible=True),
            self.brief_group: gr.update(visible=False),
            self.call_group: gr.update(visible=True),
            self.call.status_markdown: gr.update(visible=False, value=""),
            self.call.input_group: gr.update(visible=True),
            self.call.input_box: gr.update(
                interactive=False,
                value="",
                autofocus=False,
            ),
            self.call.send: gr.update(interactive=False),
            self.call.completion_markdown: gr.update(value=""),
            self.call.restart_button: gr.update(interactive=False),
            self.call.continue_button: gr.update(
                value=self.call.continue_button.value,
                interactive=False,
            ),
        }

    def streaming_updates(self) -> dict[object, object]:
        """Hide the waiting indicator while streamed caller output is rendering."""
        return {
            self.container: gr.update(visible=True),
            self.brief_group: gr.update(visible=False),
            self.call_group: gr.update(visible=True),
            self.call.status_markdown: gr.update(visible=False, value=""),
            self.call.input_group: gr.update(visible=True),
            self.call.input_box: gr.update(
                interactive=False,
                value="",
                autofocus=False,
            ),
            self.call.send: gr.update(interactive=False),
            self.call.completion_markdown: gr.update(value=""),
            self.call.restart_button: gr.update(interactive=False),
            self.call.continue_button: gr.update(
                value=self.call.continue_button.value,
                interactive=False,
            ),
        }

    def call_error_updates(self) -> dict[object, object]:
        """Keep the call visible but require a restart when the session fails."""
        return {
            self.container: gr.update(visible=True),
            self.brief_group: gr.update(visible=False),
            self.call_group: gr.update(visible=True),
            self.call.status_markdown: gr.update(visible=False, value=""),
            self.call.input_group: gr.update(visible=False),
            self.call.input_box: gr.update(
                interactive=False, value="", autofocus=False
            ),
            self.call.send: gr.update(interactive=False),
            self.call.completion_markdown: gr.update(value=""),
            self.call.restart_button: gr.update(interactive=True),
            self.call.continue_button: gr.update(
                value=self.call.continue_button.value,
                interactive=False,
            ),
        }

    def complete_updates(
        self,
        *,
        completion_markdown: str | None = None,
        continue_label: str | None = None,
        restart_label: str | None = None,
    ) -> dict[object, object]:
        """Show completion controls while keeping the chat transcript visible."""
        resolved_completion_markdown = (
            completion_markdown or self.call.default_completion_markdown
        )
        resolved_continue_label = continue_label or self.call.continue_button.value
        resolved_restart_label = restart_label or self.call.restart_button.value
        return {
            self.container: gr.update(visible=True),
            self.brief_group: gr.update(visible=False),
            self.call_group: gr.update(visible=True),
            self.call.status_markdown: gr.update(
                visible=False,
                value=self.call.status_markdown.value,
            ),
            self.call.input_group: gr.update(visible=False),
            self.call.input_box: gr.update(
                interactive=False, value="", autofocus=False
            ),
            self.call.send: gr.update(interactive=False),
            self.call.completion_markdown: gr.update(value=resolved_completion_markdown),
            self.call.restart_button: gr.update(
                value=resolved_restart_label,
                interactive=True,
            ),
            self.call.continue_button: gr.update(
                value=resolved_continue_label,
                interactive=True,
            ),
        }


@dataclass(slots=True)
class SurveyStepView:
    """Reusable survey pane."""

    container: gr.Column
    intro_markdown: gr.Markdown
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
        visible: bool | Literal["hidden"] = True,
    ) -> SurveyStepView:
        """Render the survey pane."""
        initially_visible = visible is True
        with gr.Column() as container:
            intro_view = gr.Markdown(intro_markdown, visible=initially_visible)
            survey_radios = [
                gr.Radio(
                    choices=survey.get_choices(),
                    label=question.text,
                    type="value",
                    interactive=True,
                    visible=initially_visible,
                )
                for question in survey.questions
            ]

            survey_feedback = gr.Textbox(
                label=feedback_label,
                placeholder=feedback_placeholder,
                lines=2,
                max_lines=5,
                interactive=True,
                visible=initially_visible,
            )
            survey_submit = gr.Button(
                submit_label,
                variant="primary",
                interactive=False,
                visible=initially_visible,
            )
            with gr.Column(visible=initially_visible) as completion_group:
                survey_thanks = gr.Markdown(thanks_markdown, visible=False)
                restart_button = (
                    gr.Button(restart_label, variant="primary", visible=False)
                    if restart_label is not None
                    else None
                )
        return cls(
            container=container,
            intro_markdown=intro_view,
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
            self.container,
            self.intro_markdown,
            *self.survey_radios,
            self.survey_feedback,
            self.survey_submit,
            self.completion_group,
            self.survey_thanks,
        ]
        if self.restart_button is not None:
            outputs.append(self.restart_button)
        return outputs

    def phase_updates(self, *, visible: bool) -> dict[object, object]:
        """Show or hide the survey widgets together."""
        updates: dict[object, object] = {
            self.intro_markdown: gr.update(visible=visible),
            self.survey_feedback: gr.update(visible=visible),
            self.survey_submit: gr.update(visible=visible),
            self.completion_group: gr.update(visible=visible),
        }
        for radio in self.survey_radios:
            updates[radio] = gr.update(visible=visible)
        if not visible:
            updates[self.survey_thanks] = gr.update(
                value=self.survey_thanks.value,
                visible=False,
            )
        if self.restart_button is not None and not visible:
            updates[self.restart_button] = gr.update(
                value=self.restart_button.value,
                interactive=False,
                visible=False,
            )
        return updates

    def reset_updates(self, *, visible: bool = True) -> dict[object, object]:
        """Reset the survey to its initial interactive state."""
        updates: dict[object, object] = {
            self.intro_markdown: gr.update(visible=visible),
            self.survey_feedback: gr.update(value="", interactive=True, visible=visible),
            self.survey_submit: gr.update(visible=visible, interactive=False),
            self.completion_group: gr.update(visible=visible),
            self.survey_thanks: gr.update(
                value=self.survey_thanks.value,
                visible=False,
            ),
        }
        for radio in self.survey_radios:
            updates[radio] = gr.update(interactive=True, visible=visible)
        if self.restart_button is not None:
            updates[self.restart_button] = gr.update(
                value=self.restart_button.value,
                interactive=True,
                visible=False,
            )
        return updates
