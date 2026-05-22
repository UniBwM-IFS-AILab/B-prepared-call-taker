"""Gradio app entrypoint and launch helpers for the Emergency Call Simulator."""

from __future__ import annotations

import logging
import os
import time
from argparse import ArgumentParser, Namespace

import gradio as gr
import requests

from ems_prepared.adapters.gradio.args import GradioAppArgs
from ems_prepared.adapters.gradio.args import (
    from_namespace as gradio_args_from_namespace,
)
from ems_prepared.adapters.gradio.args import (
    register_arguments as register_gradio_arguments,
)
from ems_prepared.adapters.gradio.session_demo import build_session_demo
from ems_prepared.model.contracts import FrontendPlugin, SessionManager

logger = logging.getLogger(__name__)

GRADIO_CSS = """
    .icon-button-wrapper.top-panel { display: none !important; }

    .guided-outer-walkthrough {
        margin-bottom: 0.75rem;
    }

    .guided-phase > .guidance {
        margin-bottom: 0.75rem;
    }

    .guided-call-status {
        margin-bottom: 0.5rem;
    }

    .guided-continue-button[disabled],
    .guided-continue-button:disabled {
        display: none !important;
    }

    #session_walkthrough [role="tab"] {
        pointer-events: none !important;
        cursor: default !important;
    }

    #session_walkthrough [role="tabpanel"] {
        display: none !important;
    }

    #session_info_display {
        position: fixed !important;
        bottom: 8px !important;
        right: 8px !important;
        left: auto !important;
        background: transparent !important;
        padding: 6px 10px !important;
        border: none !important;
        z-index: 1000 !important;
        pointer-events: none;
        color: var(--text-color-primary) !important;
        opacity: 0.95 !important;
        text-align: right !important;
        max-width: 40vw;
    }

    #session_info_display * {
        user-select: text;
    }
"""


def build_demo(session_manager: SessionManager, args: GradioAppArgs) -> gr.Blocks:
    """Build the selected Gradio demo for the configured UI mode."""
    return build_session_demo(session_manager=session_manager, args=args)


def _notify_share_url(share_url: str) -> None:
    """Post the new share URL to Slack via incoming webhook."""
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook_url:
        logger.warning("SLACK_WEBHOOK_URL not set; skipping Slack notification")
        return

    payload = {"text": f"New *Emergency Call Simulator* URL:\n{share_url}"}

    try:
        resp = requests.post(webhook_url, json=payload, timeout=5)
        resp.raise_for_status()
        logger.info("Posted share URL to Slack")
    except requests.RequestException:
        logger.exception("Failed posting share URL to Slack")


def _launch_gradio_app(
    demo: gr.Blocks,
    args: GradioAppArgs,
) -> tuple[object, str, str | None]:
    """Launch Gradio with queueing and return launch metadata."""
    verbose_logging_enabled = args.is_verbose_logging_enabled()
    return demo.queue(default_concurrency_limit=16).launch(
        pwa=True,
        share=True,
        css=GRADIO_CSS,
        footer_links=["settings"],
        debug=verbose_logging_enabled,
        inbrowser=args.debug,
        show_error=True,
        prevent_thread_lock=True,
    )


def _log_launch_info(local_url: str, share_url: str | None) -> None:
    """Log launch URLs and send share URL notification when available."""
    logger.info("Gradio local URL: %s", local_url)
    if share_url:
        logger.info("Gradio share URL: %s", share_url)
        _notify_share_url(share_url)
    else:
        logger.info("Gradio share URL unavailable")


def _is_gradio_server_running(demo: gr.Blocks) -> bool:
    """Return whether the launched Gradio server thread is still alive."""
    server = getattr(demo, "server", None)
    thread = getattr(server, "thread", None)
    if thread is not None:
        return bool(thread.is_alive())
    return bool(getattr(demo, "is_running", False))


def run_gradio_app(session_manager: SessionManager, args: GradioAppArgs) -> int:
    """Build and launch the Gradio app with injected dependencies."""
    demo = build_demo(session_manager=session_manager, args=args)
    _app, local_url, share_url = _launch_gradio_app(demo=demo, args=args)
    _ = _app
    _log_launch_info(local_url, share_url)
    if args.exit_on_launch:
        return 0

    interrupted = False
    try:
        while _is_gradio_server_running(demo):
            time.sleep(1)
    except KeyboardInterrupt:
        interrupted = True
        logger.info("Keyboard interrupt received; shutting down Gradio.")
    finally:
        if _is_gradio_server_running(demo):
            demo.close()
    return 130 if interrupted else 0


class GradioFrontend(FrontendPlugin):
    """Frontend plugin that launches the Gradio UI."""

    def register_arguments(self, subparser: ArgumentParser, /) -> None:
        """Register Gradio frontend specific arguments."""
        register_gradio_arguments(subparser)

    def run(
        self,
        session_manager: SessionManager,
        parsed_args: Namespace,
        /,
    ) -> int | None:
        """Run Gradio with parser-composed launcher arguments."""
        return run_gradio_app(
            session_manager=session_manager,
            args=gradio_args_from_namespace(parsed_args),
        )


FRONTEND_PLUGIN = GradioFrontend()
