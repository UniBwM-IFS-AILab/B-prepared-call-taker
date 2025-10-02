from __future__ import annotations

from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.providers.openai import OpenAIProvider

from ems_prepared.models.env import get_env_var


def build_openai_client() -> AsyncOpenAI:
    openai_key: str = get_env_var("OPENAI_API_KEY")
    return AsyncOpenAI(max_retries=3, api_key=openai_key)


def default_openai_settings() -> OpenAIResponsesModelSettings:
    # TODO: share settings across models if needed, values can be overridden per-model
    return OpenAIResponsesModelSettings(
        # openai_reasoning_summary="detailed",
        # openai_temperature=0,  # type: ignore
        temperature=0,
    )


def build_gpt4o_model() -> OpenAIResponsesModel:
    client = build_openai_client()
    settings = default_openai_settings()
    return OpenAIResponsesModel(
        "gpt-4o",
        provider=OpenAIProvider(openai_client=client),
        settings=settings,
    )
