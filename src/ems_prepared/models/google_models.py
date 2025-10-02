from __future__ import annotations

from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.providers.google import GoogleProvider

from ems_prepared.models.env import get_env_var
from ems_prepared.dialogue_state.emergency_call_state import EmergencyCall


def build_google_provider() -> GoogleProvider:
    google_key: str = get_env_var("GOOGLE_API_KEY")
    return GoogleProvider(api_key=google_key)


def build_gemini_settings() -> GoogleModelSettings:
    return GoogleModelSettings(
        temperature=0,
        extra_body={
            "response_mime_type": "application/json",
            "response_schema": EmergencyCall,
        },
    )


def build_gemini_flash_model(provider: GoogleProvider | None = None) -> GoogleModel:
    provider = provider or build_google_provider()
    settings = build_gemini_settings()
    return GoogleModel(model_name="gemini-2.5-flash", provider=provider, settings=settings)


def build_gemini_pro_model(provider: GoogleProvider | None = None) -> GoogleModel:
    provider = provider or build_google_provider()
    settings = build_gemini_settings()
    return GoogleModel(model_name="gemini-2.5-pro", provider=provider, settings=settings)
