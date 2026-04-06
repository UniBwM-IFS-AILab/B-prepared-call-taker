from __future__ import annotations

import asyncio
import json

import httpx
from httpx import HTTPStatusError, Request, Response

from ems_prepared.util.models import (
    RequestRateLimiter,
    _extract_per_minute_quota_limit,
    _extract_retry_delay_seconds,
    _parse_retry_after_seconds,
    _retry_sleep_seconds,
    _should_retry_provider_exception,
    _stop_retrying_provider_exception,
    create_retrying_client,
)


def test_extract_per_minute_quota_limit_from_google_error_metadata() -> None:
    body = """
    {
      "error": {
        "code": 429,
        "message": "Quota exceeded for quota metric 'GenerateContent requests' and limit 'GenerateContent requests per minute' of service 'generativelanguage.googleapis.com'.",
        "status": "RESOURCE_EXHAUSTED",
        "details": [
          {
            "@type": "type.googleapis.com/google.rpc.QuotaFailure",
            "violations": [
              {
                "quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
                "quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier",
                "quotaDimensions": {
                  "location": "global",
                  "model": "gemini-3.1-flash-lite"
                },
                "quotaValue": "15"
              }
            ]
          }
        ]
      }
    }
    """
    assert _extract_per_minute_quota_limit(body) == 15


def test_extract_retry_delay_seconds_from_retry_info() -> None:
    body = """
    {
      "error": {
        "details": [
          {
            "@type": "type.googleapis.com/google.rpc.RetryInfo",
            "retryDelay": "8s"
          }
        ]
      }
    }
    """
    assert _extract_retry_delay_seconds(body) == 8.0


def test_retry_sleep_seconds_prefers_exception_retry_after() -> None:
    request = Request("POST", "https://example.com")
    response = Response(429, request=request)
    exc = HTTPStatusError("rate limited", request=request, response=response)
    exc.retry_after_seconds = 42.0

    class _Outcome:
        def exception(self):
            return exc

    class _RetryState:
        outcome = _Outcome()

    assert _retry_sleep_seconds(_RetryState()) == 42.0


def test_parse_retry_after_seconds_supports_numeric_header() -> None:
    assert _parse_retry_after_seconds("17") == 17.0


def test_should_retry_provider_exception_for_read_timeout() -> None:
    assert _should_retry_provider_exception(httpx.ReadTimeout("timed out")) is True


def test_should_retry_provider_exception_does_not_retry_gateway_timeout() -> None:
    request = Request("POST", "https://example.com")
    response = Response(504, request=request)
    exc = HTTPStatusError("gateway timeout", request=request, response=response)

    assert _should_retry_provider_exception(exc) is False


def test_should_retry_provider_exception_for_resource_exhausted_payload() -> None:
    request = Request("POST", "https://example.com")
    response = Response(403, request=request)
    exc = HTTPStatusError("quota exhausted", request=request, response=response)
    exc.response_body = json.dumps(
        {
            "error": {
                "status": "RESOURCE_EXHAUSTED",
                "message": "Quota exceeded for this service.",
                "details": [
                    {"@type": "type.googleapis.com/google.rpc.QuotaFailure"}
                ],
            }
        }
    )

    assert _should_retry_provider_exception(exc) is True


def test_request_rate_limiter_waits_for_configured_cap(monkeypatch) -> None:
    limiter = RequestRateLimiter(requests_per_minute=1)
    times = iter([0.0, 0.0, 60.0])
    sleeps: list[float] = []

    monkeypatch.setattr("ems_prepared.util.models.monotonic", lambda: next(times))

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("ems_prepared.util.models.asyncio.sleep", fake_sleep)

    async def run() -> None:
        await limiter.wait_for_available_slot()
        await limiter.wait_for_available_slot()

    asyncio.run(run())

    assert sleeps == [60.0]


def test_request_rate_limiter_only_learns_stricter_limit() -> None:
    limiter = RequestRateLimiter(requests_per_minute=10)

    assert limiter.maybe_reduce_limit(12) == 10
    assert limiter.maybe_reduce_limit(8) == 8


def test_create_retrying_client_uses_long_timeout() -> None:
    client = create_retrying_client()
    try:
        assert client.timeout.connect == 30.0
        assert client.timeout.read == 300.0
        assert client.timeout.write == 30.0
        assert client.timeout.pool == 30.0
    finally:
        asyncio.run(client.aclose())


def test_stop_retrying_provider_exception_never_stops_rate_limits() -> None:
    request = Request("POST", "https://example.com")
    response = Response(429, request=request)
    exc = HTTPStatusError("rate limited", request=request, response=response)

    class _Outcome:
        def exception(self):
            return exc

    class _RetryState:
        attempt_number = 99
        outcome = _Outcome()

    assert _stop_retrying_provider_exception(_RetryState()) is False


def test_stop_retrying_provider_exception_stops_after_transient_attempt_budget() -> None:
    exc = httpx.ReadTimeout("timed out")

    class _Outcome:
        def exception(self):
            return exc

    class _RetryState:
        attempt_number = 3
        outcome = _Outcome()

    assert _stop_retrying_provider_exception(_RetryState()) is True
