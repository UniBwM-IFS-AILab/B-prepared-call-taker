from __future__ import annotations

from httpx import HTTPStatusError, Request, Response

from ems_prepared.util.models import (
    _extract_per_minute_quota_limit,
    _extract_retry_delay_seconds,
    _parse_retry_after_seconds,
    _retry_sleep_seconds,
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
