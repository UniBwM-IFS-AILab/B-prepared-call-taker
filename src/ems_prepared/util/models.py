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

import logging
import os

from httpx import AsyncClient, HTTPStatusError
from loguru import logger
from pydantic_ai.agent import Agent
from pydantic_ai.models import Model, infer_model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.retries import AsyncTenacityTransport, RetryConfig, wait_retry_after
from pydantic_ai.settings import ModelSettings
from tenacity import retry_if_exception_type, stop_after_attempt, wait_exponential

# Custom provider prefixes that need special handling
_OLLAMA_LOCAL_PREFIX = "ollama-local:"
_OPENWEBUI_PREFIX = "openwebui:"


def _build_ollama_local_model(
    model_name: str, settings: ModelSettings | None = None
) -> Model:
    """Build a model using a local Ollama instance.

    Uses OLLAMA_LOCAL_BASE_URL env var, defaulting to http://localhost:11434/v1
    """
    base_url = os.getenv("OLLAMA_LOCAL_BASE_URL", "http://localhost:11434/v1")
    provider = OllamaProvider(base_url=base_url)
    return OpenAIChatModel(model_name=model_name, provider=provider, settings=settings)


def _build_openwebui_model(
    model_name: str, settings: ModelSettings | None = None
) -> Model:
    """Build a model using OpenWebUI as the provider.

    Requires OPENWEBUI_URI and OPENWEBUI_API_KEY env vars.
    """
    openwebui_uri = os.environ["OPENWEBUI_URI"]
    openwebui_key = os.environ["OPENWEBUI_API_KEY"]
    provider = OpenAIProvider(base_url=f"{openwebui_uri}/api", api_key=openwebui_key)
    return OpenAIChatModel(model_name=model_name, provider=provider, settings=settings)


def create_retrying_client():
    """Create a client with smart retry handling for multiple error types."""

    def should_retry_status(response):
        """Raise exceptions for retryable HTTP status codes."""

        if response.status_code in (429, 502, 503, 504):
            logger.warning(
                f"Request failed with status {response.status_code}, retrying..."
            )

            response.raise_for_status()  # This will raise HTTPStatusError

    transport = AsyncTenacityTransport(
        config=RetryConfig(
            # Retry on HTTP errors and connection issues
            retry=retry_if_exception_type((HTTPStatusError, ConnectionError)),
            # Smart waiting: respects Retry-After headers, falls back to exponential backoff
            wait=wait_retry_after(
                fallback_strategy=wait_exponential(multiplier=1, max=60), max_wait=300
            ),
            # Stop after 5 attempts
            stop=stop_after_attempt(5),
            # Re-raise the last exception if all retries fail
            reraise=True,
        ),
        validate_response=should_retry_status,
    )
    return AsyncClient(transport=transport)


def build_models(
    *names: str,
) -> tuple[Model, ...]:
    """Create multiple Model instances from provider:model name strings.

    Args:
        *names: Model names in provider:model format (e.g., "github:gpt-4.1-mini",
            "ollama:llama3", "google-gla:gemini-2.5-flash", "anthropic:claude-sonnet-4-5").
            Special prefixes "ollama-local:" and "openwebui:" use custom providers.
        settings: Optional ModelSettings to apply to all models. Defaults to temperature=0.

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
                _build_ollama_local_model(
                    model_name,  # settings
                )
            )
        elif name.startswith(_OPENWEBUI_PREFIX):
            model_name = name[len(_OPENWEBUI_PREFIX) :]
            models.append(
                _build_openwebui_model(
                    model_name,  # settings
                )
            )
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
    default_settings = ModelSettings(temperature=0)
    if overrides:
        for key, value in overrides.items():
            setattr(default_settings, key, value)
    return default_settings


def get_default_models(
    overrides: list[str] | None = None, extras: list[str] | None = None
) -> tuple[Model, ...]:
    """Get default models with optional overrides and extras.

    Args:
        overrides: List of provider:model strings to use instead of defaults.
        extras: List of additional provider:model strings to include.

    Returns:
        Tuple of Model instances.
    """
    if overrides is not None:
        model_names = overrides
    else:
        model_names = [
            # TODO: implement ModelRetry from PydanticAI, gemini causes this to crash due to http errors: 429, 503, ...
            "github:gpt-5-mini",
            "openai:gpt-5-mini",
            # "groq:llama-3.3-70b-versatile",
            # "cerebras:gpt-oss-120b",
            # "google-gla:gemini-2.5-flash",
            # "google-gla:gemini-2.5-pro",
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
    return build_models(*model_names)


def build_fallback_agent(
    model_overrides=None,
    model_extras=None,
    setting_overrides=None,
    **kwargs,
) -> Agent:
    """Build a FallbackModel agent with default models and settings.

    Args:
        **kwargs: Additional keyword arguments to pass to the Agent constructor.

    Returns:
        An Agent instance using a FallbackModel with default models.
    """
    from pydantic_ai.models.fallback import FallbackModel

    default_models = get_default_models(overrides=model_overrides, extras=model_extras)
    fallback_model = FallbackModel(*default_models)

    settings: ModelSettings = get_default_settings(overrides=setting_overrides)
    return Agent(model=fallback_model, model_settings=settings, **kwargs)
