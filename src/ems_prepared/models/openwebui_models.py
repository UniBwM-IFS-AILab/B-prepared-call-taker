from __future__ import annotations

from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider

from ems_prepared.models.env import get_env_var


def build_openwebui_client() -> AsyncOpenAI:
    openwebui_uri: str = get_env_var("OPENWEBUI_URI")
    openwebui_key: str = get_env_var("OPENWEBUI_API_KEY")
    return AsyncOpenAI(base_url=f"{openwebui_uri}/api", api_key=openwebui_key)


def build_llama3_model() -> OpenAIResponsesModel:
    client = build_openwebui_client()
    return OpenAIResponsesModel(
        model_name="llama3.3:latest",
        provider=OpenAIProvider(openai_client=client),
    )
