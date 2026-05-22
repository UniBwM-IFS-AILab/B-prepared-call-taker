"""Optional microphone + streaming ASR helpers for Gradio chat UIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import gradio as gr
import numpy as np

if TYPE_CHECKING:
    from transformers.pipelines.base import Pipeline


@dataclass
class AsrPane:
    """Rendered ASR controls that can be attached to a chat input."""

    audio_stream: gr.Audio
    audio_state: gr.State
    transcript: gr.Textbox
    apply_button: gr.Button

    @classmethod
    def render(cls) -> AsrPane:
        """Render microphone controls for live speech-to-text."""
        with gr.Accordion("Microphone Input", open=False):
            audio_stream = gr.Audio(
                label="Speak",
                sources=["microphone"],
                streaming=True,
                type="numpy",
            )
            transcript = gr.Textbox(
                label="Live transcript",
                lines=2,
                max_lines=4,
                interactive=False,
            )
            apply_button = gr.Button("Use Transcript", variant="secondary")
        return cls(
            audio_stream=audio_stream,
            audio_state=gr.State(value=None),
            transcript=transcript,
            apply_button=apply_button,
        )


class StreamingAsr:
    """Small streaming ASR runtime based on Transformers pipeline."""

    def __init__(self, *, model_name: str) -> None:
        """Create ASR runtime with one lazy-loaded model id."""
        self._model_name = model_name
        self._pipeline: Pipeline | None = None

    def _ensure_pipeline(self) -> Pipeline:
        if self._pipeline is None:
            from transformers import pipeline

            self._pipeline = pipeline(
                "automatic-speech-recognition",
                model=self._model_name,
            )
        return self._pipeline

    def transcribe_chunk(
        self,
        stream: np.ndarray | None,
        new_chunk: tuple[int, np.ndarray] | None,
    ) -> tuple[np.ndarray | None, str]:
        """Append one microphone chunk and transcribe the accumulated stream."""
        if new_chunk is None:
            return stream, ""

        sample_rate, audio = new_chunk
        if audio is None or len(audio) == 0:
            return stream, ""

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        normalized = audio.astype(np.float32)
        max_abs = float(np.max(np.abs(normalized)))
        if max_abs > 0:
            normalized = normalized / max_abs

        full_stream = (
            np.concatenate([stream, normalized])
            if stream is not None
            else normalized
        )

        model = self._ensure_pipeline()
        transcription = model({"sampling_rate": sample_rate, "raw": full_stream})[
            "text"
        ]
        return full_stream, transcription.strip()


def apply_asr_transcript(transcript: str) -> tuple[dict[str, object], dict[str, object]]:
    """Apply transcript text to chat input and keep send disabled for empty text."""
    cleaned = transcript.strip()
    return (
        gr.update(value=cleaned, interactive=True, autofocus=True),
        gr.update(interactive=bool(cleaned)),
    )


def wire_asr_to_chat_input(
    *,
    pane: AsrPane,
    input_box: gr.Textbox,
    send_button: gr.Button,
    runtime: StreamingAsr,
    concurrency_id: str,
) -> None:
    """Attach ASR events to a standard chat input/send pair."""
    pane.audio_stream.start_recording(
        lambda: (None, ""),
        outputs=[pane.audio_state, pane.transcript],
        queue=False,
    )
    pane.audio_stream.stream(
        runtime.transcribe_chunk,
        inputs=[pane.audio_state, pane.audio_stream],
        outputs=[pane.audio_state, pane.transcript],
        show_progress="hidden",
        concurrency_id=concurrency_id,
        concurrency_limit=1,
    )
    pane.audio_stream.stop_recording(
        apply_asr_transcript,
        inputs=[pane.transcript],
        outputs=[input_box, send_button],
        queue=False,
    )
    pane.apply_button.click(
        apply_asr_transcript,
        inputs=[pane.transcript],
        outputs=[input_box, send_button],
        queue=False,
    )
