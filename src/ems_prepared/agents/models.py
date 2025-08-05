"""Module for loading LLM models from various providers."""

import os
from dataclasses import asdict, dataclass, field, fields

from openai import AsyncOpenAI
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.openai import OpenAIProvider

from ems_prepared.state_model.emergency_call_state import EmergencyCall


@dataclass
class system_prompt:
    """Represents the different parts typically found in a system prompt."""

    role: str | None = field(default_factory=str)
    task: str | None = field(default_factory=str)
    rules: str | None = field(default_factory=str)
    decisions: str | None = field(default_factory=str)

    @property
    def full_prompt(self) -> str:
        """Concatenation of the parts of the prompt."""
        return "\n".join(
            value for _, value in asdict(self).items() if type(value is str)
        )


def get_env_var(key: str) -> str:
    """Retrieve a Variable from the environment and returns it if it exists."""
    value: str | None = os.getenv(key)

    if not value:
        raise ValueError(f"{key} environment variable is not set.")

    return value


google_key: str = get_env_var("GOOGLE_API_KEY")

provider: GoogleProvider = GoogleProvider(api_key=google_key)

gemini_settings = GoogleModelSettings(
    temperature=1.0,
    extra_body={
        "response_mime_type": "application/json",
        "response_schema": EmergencyCall,
    },
)


# https://ai.google.dev/gemini-api/docs/rate-limits

gemini_flash_model: GoogleModel = GoogleModel(
    model_name="gemini-2.5-flash", provider=provider, settings=gemini_settings
)

gemini_pro_model: GoogleModel = GoogleModel(
    model_name="gemini-2.5-pro", provider=provider, settings=gemini_settings
)


openwebui_uri: str = get_env_var("OPENWEBUI_URI")

openwebui_key: str = get_env_var("OPENWEBUI_API_KEY")

openwebui_client = AsyncOpenAI(
    base_url=f"{openwebui_uri}/api",
    api_key=openwebui_key,
)

llama3_model: OpenAIResponsesModel = OpenAIResponsesModel(
    model_name="llama3.3:latest",
    provider=OpenAIProvider(openai_client=openwebui_client),
)

# Teuken and llama4 models do not seem to work


openai_key: str = get_env_var("OPENAI_API_KEY")

openai_client = AsyncOpenAI(max_retries=3, api_key=openai_key)

# TODO: we can define a single settings class apparently for shared settings, e.g. temperature. IIUC, the values are overridable by using the correct prefix, e.g. "openai_"
openai_settings = OpenAIResponsesModelSettings(
    # openai_reasoning_summary="detailed",
    # openai_temperature=0,  # type: ignore
    temperature=0,
)
gpt4o_model: OpenAIResponsesModel = OpenAIResponsesModel(
    "gpt-4o",
    provider=OpenAIProvider(openai_client=openai_client),
    settings=openai_settings,
)
