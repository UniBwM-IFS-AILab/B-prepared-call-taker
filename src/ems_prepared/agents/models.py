import os

from openai import AsyncOpenAI
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.openai import OpenAIProvider


def get_env_var(key: str) -> str:
    value: str | None = os.getenv(key)
    if not value:
        raise ValueError(f"{key} environment variable is not set.")
    return value


google_key: str = get_env_var("GOOGLE_API_KEY")
provider: GoogleProvider = GoogleProvider(api_key=google_key)

# https://ai.google.dev/gemini-api/docs/rate-limits
gemini_flash_model: GoogleModel = GoogleModel(
    model_name="gemini-2.5-flash", provider=provider
)
gemini_pro_model: GoogleModel = GoogleModel(
    model_name="gemini-2.5-pro", provider=provider
)


openwebui_uri: str = get_env_var("OPENWEBUI_URI")
openwebui_key: str = get_env_var("OPENWEBUI_API_KEY")
openwebui_client = AsyncOpenAI(
    base_url=f"{openwebui_uri}/api",
    api_key=openwebui_key,
)
llama3_model: OpenAIModel = OpenAIModel(
    model_name="llama3.3:latest",
    provider=OpenAIProvider(openai_client=openwebui_client),
)
# Teuken and llama4 models do not seem to work
