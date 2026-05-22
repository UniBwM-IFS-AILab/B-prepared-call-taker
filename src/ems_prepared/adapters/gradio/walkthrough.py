"""Shared step-navigation helpers for Gradio UIs."""

from __future__ import annotations

from dataclasses import dataclass

import gradio as gr


@dataclass(slots=True)
class WalkthroughController:
    """Manage server-driven selection for one walkthrough."""

    walkthrough: gr.Walkthrough

    def outputs(self) -> list[object]:
        """Return the walkthrough output required for selection updates."""
        return [self.walkthrough]

    def walkthrough_update(self, *, selected_step_id: int) -> tuple[object]:
        """Return walkthrough selection update."""
        return (gr.update(selected=selected_step_id),)



def ordered_component_updates(
    outputs: Sequence[object],
    updates: dict[object, object],
) -> tuple[object, ...]:
    """Convert component-keyed updates into an output-aligned tuple."""
    unknown_outputs = [component for component in updates if component not in outputs]
    if unknown_outputs:
        raise ValueError(
            f"Returned components not present in outputs: {unknown_outputs!r}"
        )
    return tuple(updates.get(component, gr.skip()) for component in outputs)
