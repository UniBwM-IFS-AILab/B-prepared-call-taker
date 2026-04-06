"""Unified model factory using PydanticAI's native provider:model format.

This module provides a single `build_models()` function that creates Model instances
from provider:model strings. Standard providers (github:, ollama:, google-gla:, anthropic:,
openai:, etc.) are handled via PydanticAI's infer_model(). Custom providers (ollama-local:,
openwebui:) are handled separately.

Environment variables:
    - GITHUB_API_KEY: For GitHub Models
    - GOOGLE_API_KEY: For Google Gemini
    - ANTHROPIC_API_KEY: For Anthropic Claude
    - OPENAI_API_KEY: For OpenAI
    - OLLAMA_BASE_URL: For Ollama (cloud or local)
    - OLLAMA_API_KEY: For Ollama (optional, for cloud)
    - OLLAMA_LOCAL_BASE_URL: For local Ollama (defaults to http://localhost:11434/v1)
    - OPENWEBUI_URI: Base URI for OpenWebUI
    - OPENWEBUI_API_KEY: API key for OpenWebUI
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections import deque
from datetime import datetime
from email.utils import parsedate_to_datetime
from time import monotonic
from typing import Any

import httpx
from httpx._client import (
    AsyncBaseTransport,
    AsyncClient,
    AsyncHTTPTransport,
    Request,
    Response,
)
from httpx._exceptions import HTTPStatusError
from pydantic_ai.agent import Agent
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.models import Model, infer_model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.retries import RetryConfig
from pydantic_ai.settings import ModelSettings
from tenacity import retry
from tenacity.retry import retry_if_exception
from tenacity.wait import wait_exponential

# Custom provider prefixes that need special handling
_OLLAMA_LOCAL_PREFIX = "ollama-local:"
_OPENWEBUI_PREFIX = "openwebui:"

logger = logging.getLogger(__name__)
_FALLBACK_WAIT = wait_exponential(multiplier=1, max=60)
_MAX_TRANSIENT_ATTEMPTS = 3
_DEFAULT_REQUEST_TIMEOUT = httpx.Timeout(connect=30.0, read=300.0, write=30.0, pool=30.0)


class RequestRateLimiter:
    """Coordinate a shared requests-per-minute budget across model clients."""

    def __init__(self, requests_per_minute: int | None = None):
        if requests_per_minute is not None and requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        self._requests_per_minute = requests_per_minute
        self._request_timestamps: deque[float] = deque()
        self._rate_limit_lock = asyncio.Lock()

    @property
    def requests_per_minute(self) -> int | None:
        return self._requests_per_minute

    def maybe_reduce_limit(self, requests_per_minute: int | None) -> int | None:
        if requests_per_minute is None:
            return self._requests_per_minute
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        if (
            self._requests_per_minute is None
            or requests_per_minute < self._requests_per_minute
        ):
            self._requests_per_minute = requests_per_minute
        return self._requests_per_minute

    def _prune_request_timestamps(self, now: float) -> None:
        while self._request_timestamps and now - self._request_timestamps[0] >= 60:
            self._request_timestamps.popleft()

    def seconds_until_available_slot(self, now: float) -> float:
        if self._requests_per_minute is None:
            return 0.0
        self._prune_request_timestamps(now)
        if len(self._request_timestamps) < self._requests_per_minute:
            return 0.0
        required_expired_index = (
            len(self._request_timestamps) - self._requests_per_minute
        )
        return max(self._request_timestamps[required_expired_index] + 60 - now, 0.0)

    async def wait_for_available_slot(self) -> None:
        if self._requests_per_minute is None:
            self._request_timestamps.append(monotonic())
            return
        async with self._rate_limit_lock:
            while True:
                now = monotonic()
                sleep_for = self.seconds_until_available_slot(now)
                if sleep_for <= 0:
                    self._request_timestamps.append(now)
                    return
                timestamp = datetime.now().isoformat(timespec="seconds")
                print(
                    f"[{timestamp}] [provider_rate_limit] waiting {sleep_for:.0f}s to stay within active {self._requests_per_minute} RPM",
                    flush=True,
                )
                await asyncio.sleep(sleep_for)


def _extract_per_minute_quota_limit(body: str) -> int | None:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return None
    error = payload.get("error", {})
    message = str(error.get("message", "")).lower()
    for detail in error.get("details", []):
        if detail.get("@type") != "type.googleapis.com/google.rpc.QuotaFailure":
            continue
        for violation in detail.get("violations", []):
            quota_value = violation.get("quotaValue")
            quota_id = str(violation.get("quotaId", "")).lower()
            if quota_value is None:
                continue
            if "perminute" in quota_id.replace("_", "") or "per minute" in message:
                try:
                    return int(quota_value)
                except (TypeError, ValueError):
                    return None
    return None


def _parse_retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(float(value), 0.0)
    except ValueError:
        pass
    try:
        retry_at = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return None
    return max((retry_at - datetime.now(retry_at.tzinfo)).total_seconds(), 0.0)


def _get_google_error_payload(body: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return None
    error = payload.get("error")
    if isinstance(error, dict):
        return error
    return None


def _extract_retry_delay_seconds(body: str) -> float | None:
    error = _get_google_error_payload(body)
    if error is None:
        return None
    for detail in error.get("details", []):
        if detail.get("@type") != "type.googleapis.com/google.rpc.RetryInfo":
            continue
        retry_delay = detail.get("retryDelay")
        if not isinstance(retry_delay, str) or not retry_delay.endswith("s"):
            continue
        try:
            return max(float(retry_delay[:-1]), 0.0)
        except ValueError:
            return None
    return None


def _has_google_rate_limit_details(body: str) -> bool:
    error = _get_google_error_payload(body)
    if error is None:
        return False
    for detail in error.get("details", []):
        detail_type = detail.get("@type")
        if detail_type in (
            "type.googleapis.com/google.rpc.QuotaFailure",
            "type.googleapis.com/google.rpc.RetryInfo",
        ):
            return True
    return False


def _response_indicates_rate_limit(status_code: int, body: str) -> bool:
    if status_code == 429:
        return True
    error = _get_google_error_payload(body)
    if error is None:
        return False
    status = str(error.get("status", "")).upper()
    message = str(error.get("message", "")).lower()
    if status == "RESOURCE_EXHAUSTED":
        return True
    if _has_google_rate_limit_details(body):
        return "quota" in message or "too many requests" in message or status in {
            "RESOURCE_EXHAUSTED",
            "RATE_LIMIT_EXCEEDED",
        }
    return False


def _is_rate_limit_http_error(exc: HTTPStatusError) -> bool:
    body = getattr(exc, "response_body", "")
    return _response_indicates_rate_limit(exc.response.status_code, body)


def _retry_sleep_seconds(retry_state) -> float:
    exc = retry_state.outcome.exception()
    if isinstance(exc, HTTPStatusError):
        retry_after = getattr(exc, "retry_after_seconds", None)
        if retry_after is not None:
            return retry_after
        headers = getattr(exc, "response_headers", dict(exc.response.headers))
        parsed = _parse_retry_after_seconds(
            headers.get("retry-after") or headers.get("Retry-After")
        )
        if parsed is not None:
            return parsed
    return _FALLBACK_WAIT(retry_state)


def _should_retry_provider_exception(exc: BaseException) -> bool:
    if isinstance(exc, HTTPStatusError):
        return _is_rate_limit_http_error(exc)
    return isinstance(exc, (httpx.TransportError, ConnectionError))


def _stop_retrying_provider_exception(retry_state) -> bool:
    exc = retry_state.outcome.exception()
    if isinstance(exc, HTTPStatusError) and _is_rate_limit_http_error(exc):
        return False
    return retry_state.attempt_number >= _MAX_TRANSIENT_ATTEMPTS


def _log_retry(retry_state):
    """Log retry attempts for HTTP requests."""
    exc = retry_state.outcome.exception()
    attempt = retry_state.attempt_number
    sleep = retry_state.next_action.sleep if retry_state.next_action else None
    logger.info(f"HTTP request failed (attempt {attempt}), retrying after error: {exc}")
    timestamp = datetime.now().isoformat(timespec="seconds")
    if isinstance(exc, HTTPStatusError):
        headers = getattr(exc, "response_headers", dict(exc.response.headers))
        body = getattr(exc, "response_body", "<unavailable>")
        print(
            f"[{timestamp}] [provider_retry] attempt={attempt} status={exc.response.status_code} method={exc.request.method} url={exc.request.url}",
            flush=True,
        )
        print(f"[{timestamp}] [provider_retry] headers={headers}", flush=True)
        print(f"[{timestamp}] [provider_retry] body={body}", flush=True)
        learned_limit = getattr(exc, "requests_per_minute", None)
        if learned_limit is not None:
            print(
                f"[{timestamp}] [provider_retry] learned requests_per_minute={learned_limit}",
                flush=True,
            )
    else:
        print(
            f"[{timestamp}] [provider_retry] attempt={attempt} error={type(exc).__name__}: {exc}",
            flush=True,
        )
    if sleep is not None:
        print(f"[{timestamp}] [provider_retry] sleeping {sleep:.0f}s", flush=True)


class _RetryingLoggingTransport(AsyncBaseTransport):
    """Async transport with retry logging and captured response details."""

    def __init__(
        self,
        config: RetryConfig,
        rate_limiter: RequestRateLimiter | None = None,
        wrapped: AsyncBaseTransport | None = None,
    ):
        self.config = config
        self.wrapped = wrapped or AsyncHTTPTransport()
        self.rate_limiter = rate_limiter or RequestRateLimiter()

    def _capture_rate_limit_from_error(self, exc: HTTPStatusError) -> None:
        if not _is_rate_limit_http_error(exc):
            return
        response_body = getattr(exc, "response_body", "")
        retry_delay = _extract_retry_delay_seconds(response_body)
        if retry_delay is not None:
            exc.retry_after_seconds = retry_delay
        rpm_limit = _extract_per_minute_quota_limit(response_body)
        active_limit = self.rate_limiter.maybe_reduce_limit(rpm_limit)
        if active_limit is not None:
            exc.requests_per_minute = active_limit
            exc.retry_after_seconds = max(
                getattr(exc, "retry_after_seconds", 0.0),
                self.rate_limiter.seconds_until_available_slot(monotonic()),
                1.0,
            )

    async def handle_async_request(self, request: Request) -> Response:
        @retry(**self.config)
        async def handle(req: Request) -> Response:
            await self.rate_limiter.wait_for_available_slot()
            response = await self.wrapped.handle_async_request(req)
            response.request = req
            if response.is_error:
                body_bytes = await response.aread()
                try:
                    response.raise_for_status()
                except HTTPStatusError as exc:
                    exc.response_headers = dict(response.headers)
                    exc.response_body = body_bytes.decode("utf-8", errors="replace")
                    self._capture_rate_limit_from_error(exc)
                    await response.aclose()
                    raise
            return response

        return await handle(request)

    async def __aenter__(self) -> _RetryingLoggingTransport:
        await self.wrapped.__aenter__()
        return self

    async def __aexit__(self, exc_type=None, exc_value=None, traceback=None) -> None:
        await self.wrapped.__aexit__(exc_type, exc_value, traceback)

    async def aclose(self) -> None:
        await self.wrapped.aclose()


def build_ollama_local_model(
    model_name: str,
    settings: ModelSettings | None = None,
    http_client: AsyncClient | None = None,
) -> Model:
    """Build a model using a local Ollama instance.

    Uses OLLAMA_LOCAL_BASE_URL env var, defaulting to http://localhost:11434/v1
    """
    base_url = os.getenv("OLLAMA_LOCAL_BASE_URL", "http://localhost:11434/v1")
    provider = OllamaProvider(base_url=base_url, http_client=http_client)
    return OpenAIChatModel(model_name=model_name, provider=provider, settings=settings)


def build_openwebui_model(
    model_name: str,
    settings: ModelSettings | None = None,
    http_client: AsyncClient | None = None,
) -> Model:
    """Build a model using OpenWebUI as the provider.

    Requires OPENWEBUI_URI and OPENWEBUI_API_KEY env vars.
    """
    openwebui_uri = os.environ["OPENWEBUI_URI"]
    openwebui_key = os.environ["OPENWEBUI_API_KEY"]
    provider = OpenAIProvider(
        base_url=f"{openwebui_uri}/api", api_key=openwebui_key, http_client=http_client
    )
    return OpenAIChatModel(model_name=model_name, provider=provider, settings=settings)


def build_models(
    *names: str,
    http_client: AsyncClient | None = None,
) -> tuple[Model, ...]:
    """Create multiple Model instances from provider:model name strings.

    Args:
        *names: Model names in provider:model format (e.g., "github:gpt-4.1-mini",
            "ollama:llama3", "google-gla:gemini-2.5-flash", "anthropic:claude-sonnet-4-5").
            Special prefixes "ollama-local:" and "openwebui:" use custom providers.
        http_client: Optional AsyncClient with retry logic to use for providers that support it.

    Returns:
        Tuple of Model instances, suitable for unpacking into FallbackModel(*build_models(...))

    Examples:
        >>> from pydantic_ai.models.fallback import FallbackModel
        >>> model = FallbackModel(*build_models(
        ...     "github:gpt-4.1-mini",
        ...     "google-gla:gemini-2.5-flash",
        ... ))

    """
    models: list[Model] = []
    for name in names:
        if name.startswith(_OLLAMA_LOCAL_PREFIX):
            model_name = name[len(_OLLAMA_LOCAL_PREFIX) :]
            models.append(
                build_ollama_local_model(
                    model_name,
                    http_client=http_client,  # settings
                )
            )
        elif name.startswith(_OPENWEBUI_PREFIX):
            model_name = name[len(_OPENWEBUI_PREFIX) :]
            models.append(
                build_openwebui_model(
                    model_name,
                    http_client=http_client,  # settings
                )
            )
        else:
            if http_client is not None:
                # Build manually to support http_client
                if name.startswith("google-gla:"):
                    from pydantic_ai.models.google import GoogleModel
                    from pydantic_ai.providers.google import GoogleProvider

                    model_name = name[len("google-gla:") :]
                    provider = GoogleProvider(http_client=http_client)
                    model = GoogleModel(model_name, provider=provider)
                elif name.startswith("anthropic:"):
                    from pydantic_ai.models.anthropic import AnthropicModel
                    from pydantic_ai.providers.anthropic import AnthropicProvider

                    model_name = name[len("anthropic:") :]
                    provider = AnthropicProvider(http_client=http_client)
                    model = AnthropicModel(model_name, provider=provider)
                elif name.startswith("openai:"):
                    model_name = name[len("openai:") :]
                    provider = OpenAIProvider(http_client=http_client)
                    model = OpenAIChatModel(model_name, provider=provider)
                elif name.startswith("github:"):
                    model_name = name[len("github:") :]
                    provider = OpenAIProvider(
                        base_url="https://models.inference.ai.azure.com/",
                        http_client=http_client,
                    )
                    model = OpenAIChatModel(model_name, provider=provider)
                elif name.startswith("groq:"):
                    from pydantic_ai.models.groq import GroqModel
                    from pydantic_ai.providers.groq import GroqProvider

                    model_name = name[len("groq:") :]
                    provider = GroqProvider(http_client=http_client)
                    model = GroqModel(model_name, provider=provider)
                elif name.startswith("cerebras:"):
                    from pydantic_ai.providers.cerebras import CerebrasProvider

                    model_name = name[len("cerebras:") :]
                    provider = CerebrasProvider(http_client=http_client)
                    model = OpenAIChatModel(model_name, provider=provider)
                elif name.startswith("mistral:"):
                    from pydantic_ai.models.mistral import MistralModel
                    from pydantic_ai.providers.mistral import MistralProvider

                    model_name = name[len("mistral:") :]
                    provider = MistralProvider(http_client=http_client)
                    model = MistralModel(model_name, provider=provider)
                elif name.startswith("openrouter:"):
                    from pydantic_ai.providers.openrouter import OpenRouterProvider

                    model_name = name[len("openrouter:") :]
                    provider = OpenRouterProvider(http_client=http_client)
                    model = OpenAIChatModel(model_name, provider=provider)
                else:
                    # For other providers, use infer_model (no custom http_client)
                    model = infer_model(name)
            else:
                model = infer_model(name)
            models.append(model)

    return tuple(models)


def get_default_settings(overrides: dict[str, object] | None = None) -> ModelSettings:
    """Get default ModelSettings with optional overrides.

    Args:
        overrides: Dictionary of settings to override the defaults.

    Returns:
        ModelSettings instance with applied overrides.

    """
    default_settings = ModelSettings(
        temperature=0,
    )
    if overrides:
        for key, value in overrides.items():
            setattr(default_settings, key, value)
    return default_settings


def create_retrying_client(
    *,
    requests_per_minute: int | None = None,
    rate_limiter: RequestRateLimiter | None = None,
    timeout: httpx.Timeout | None = None,
):
    """Create a client with smart retry handling for multiple error types."""
    if rate_limiter is None:
        rate_limiter = RequestRateLimiter(requests_per_minute=requests_per_minute)
    else:
        rate_limiter.maybe_reduce_limit(requests_per_minute)
    transport = _RetryingLoggingTransport(
        config=RetryConfig(
            # Retry rate limits indefinitely and retry generic transport failures a few times.
            retry=retry_if_exception(_should_retry_provider_exception),
            # Respect provider quota signals first, then Retry-After, then exponential backoff.
            wait=_retry_sleep_seconds,
            # Let 429s wait as long as needed, but bound transport-error retries.
            stop=_stop_retrying_provider_exception,
            # Re-raise the last exception if all retries fail
            reraise=True,
            # Log retry attempts
            after=_log_retry,
        ),
        rate_limiter=rate_limiter,
    )
    return AsyncClient(
        transport=transport,
        timeout=timeout or _DEFAULT_REQUEST_TIMEOUT,
    )


def get_default_models(
    overrides: list[str] | None = None,
    extras: list[str] | None = None,
    http_client: AsyncClient | None = None,
) -> tuple[Model, ...]:
    """Get default models with optional overrides and extras.

    Args:
        overrides: List of provider:model strings to use instead of defaults.
        extras: List of additional provider:model strings to include.
        http_client: Optional AsyncClient with retry logic.

    Returns:
        Tuple of Model instances.

    """
    if overrides is not None:
        model_names = overrides
    else:
        model_names = [
            "google-gla:gemini-3.1-flash-lite-preview",  # https://aistudio.google.com/api-keys
            # "google-gla:gemini-2.5-pro",
            # "google-gla:gemini-3-flash-preview",
            # "google-gla:gemini-3-pro-preview",
            # "github:gpt-5",
            # "openai:gpt-5",
            # "github:gpt-5-mini",
            # "openai:gpt-5-mini",
            # "groq:llama-3.3-70b-versatile",
            # "cerebras:gpt-oss-120b",
            # "ollama-local:deepseek-r1:8b",
            # "ollama-local:deepseek-r1:14b",
            # "ollama-local:qwen3",
            # "ollama-local:phi4-reasoning:14b",
            # "ollama-local:phi4:14b",
            # "groq:gpt-oss-20b",
            # "openrouter:openai/gpt-oss-20b:free",
        ]
    if extras:
        model_names.extend(extras)
    return build_models(*model_names, http_client=http_client)


def build_fallback_agent(
    model_overrides: list[str] | None = None,
    model_extras: list[str] | None = None,
    setting_overrides: dict[str, object] | None = None,
    requests_per_minute: int | None = None,
    rate_limiter: RequestRateLimiter | None = None,
    request_timeout: httpx.Timeout | None = None,
    **kwargs: Any,
) -> Agent[Any, Any]:
    """Build a FallbackModel agent with default models and settings.

    Args:
        **kwargs: Additional keyword arguments to pass to the Agent constructor.

    Returns:
        An Agent instance using a FallbackModel with default models.

    """
    from pydantic_ai.models.fallback import FallbackModel

    http_client = create_retrying_client(
        requests_per_minute=requests_per_minute,
        rate_limiter=rate_limiter,
        timeout=request_timeout,
    )
    default_models = get_default_models(
        overrides=model_overrides, extras=model_extras, http_client=http_client
    )
    fallback_model = FallbackModel(
        *default_models,
        fallback_on=(ModelAPIError, HTTPStatusError, httpx.TransportError, ConnectionError),
    )

    settings: ModelSettings = get_default_settings(overrides=setting_overrides)
    return Agent(
        model=fallback_model,
        model_settings=settings,
        **kwargs,
    )
