"""Tests for the additive `Settings` aggregate facade."""

from __future__ import annotations

from uuid import UUID

from ems_prepared.model.context import InputMode, Locale, Settings


def test_settings_exposes_aggregate_bundles(monkeypatch, tmp_path) -> None:
    """`Settings` should expose grouped run_info/transport/storage/telemetry data."""
    monkeypatch.chdir(tmp_path)

    settings = Settings(
        name="unit_test",
        user_id=UUID(int=1),
        session_id=UUID(int=2),
        experiment_name="exp",
        scenario_name="scenario.md",
        policy_name="graph",
        locale=Locale.DE,
        call_origin=InputMode.TEST,
    )

    assert settings.run_info.user_id == settings.user_id
    assert settings.run_info.file_name == settings.file_name
    assert settings.locale is Locale.DE
    assert settings.call_origin is InputMode.TEST
    assert settings.transport.request_input is None
    assert settings.storage.save_path.exists()
    assert settings.telemetry.logger.name.endswith(settings.session_id.hex)
    assert settings.telemetry.messages_logger is not None
    assert settings.telemetry.state_logger is not None


def test_settings_model_dump_excludes_runtime_transport(monkeypatch, tmp_path) -> None:
    """Runtime-only transport fields should stay out of serialized deps data."""
    monkeypatch.chdir(tmp_path)

    settings = Settings(name="serialize_me")
    payload = settings.model_dump()

    assert "emit" not in payload
    assert "request_input" not in payload
