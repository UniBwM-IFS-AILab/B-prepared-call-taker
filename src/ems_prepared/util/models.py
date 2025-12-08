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

import os
from typing import TYPE_CHECKING

from pydantic_ai.models import Model, infer_model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

if TYPE_CHECKING:
    pass


# Custom provider prefixes that need special handling
_OLLAMA_LOCAL_PREFIX = "ollama-local:"
_OPENWEBUI_PREFIX = "openwebui:"


def _build_ollama_local_model(model_name: str, settings: ModelSettings | None) -> Model:
    """Build a model using a local Ollama instance.

    Uses OLLAMA_LOCAL_BASE_URL env var, defaulting to http://localhost:11434/v1
    """
    base_url = os.getenv("OLLAMA_LOCAL_BASE_URL", "http://localhost:11434/v1")
    provider = OllamaProvider(base_url=base_url)
    return OpenAIChatModel(model_name=model_name, provider=provider, settings=settings)


def _build_openwebui_model(model_name: str, settings: ModelSettings | None) -> Model:
    """Build a model using OpenWebUI as the provider.

    Requires OPENWEBUI_URI and OPENWEBUI_API_KEY env vars.
    """
    openwebui_uri = os.environ["OPENWEBUI_URI"]
    openwebui_key = os.environ["OPENWEBUI_API_KEY"]
    provider = OpenAIProvider(base_url=f"{openwebui_uri}/api", api_key=openwebui_key)
    return OpenAIChatModel(model_name=model_name, provider=provider, settings=settings)


def build_models(
    *names: str,
    settings: ModelSettings | None = None,
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
    if settings is None:
        settings = ModelSettings(temperature=0)

    models: list[Model] = []
    for name in names:
        if name.startswith(_OLLAMA_LOCAL_PREFIX):
            model_name = name[len(_OLLAMA_LOCAL_PREFIX) :]
            models.append(_build_ollama_local_model(model_name, settings))
        elif name.startswith(_OPENWEBUI_PREFIX):
            model_name = name[len(_OPENWEBUI_PREFIX) :]
            models.append(_build_openwebui_model(model_name, settings))
        else:
            # Use PydanticAI's native infer_model for standard providers
            # This handles github:, ollama:, google-gla:, anthropic:, openai:, etc.
            model = infer_model(name)
            # Note: infer_model doesn't accept settings directly, they're applied at agent level
            models.append(model)

    return tuple(models)
