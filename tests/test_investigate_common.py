# Copyright SUSE LLC
# ruff: file-ignore[magic-value-comparison, any-type, too-many-arguments, assert, boolean-type-hint-positional-argument]
"""Unit tests for investigate_common module."""

from __future__ import annotations

import datetime
import logging
import subprocess
import sys
from email.utils import format_datetime
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, Mock

import httpx
import pytest

import investigate_common

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def test_constants() -> None:
    assert pytest.approx(90.0) == investigate_common.DEFAULT_HTTP_TIMEOUT
    assert pytest.approx(120.0) == investigate_common.DEFAULT_INVESTIGATION_TIMEOUT


@pytest.mark.parametrize(
    ("verbose", "quiet", "expected_level"),
    [
        (0, False, logging.WARNING),
        (1, False, logging.INFO),
        (2, False, logging.DEBUG),
        (3, False, logging.DEBUG),
        (0, True, logging.ERROR),
        (2, True, logging.ERROR),
    ],
)
def test_setup_logging(mocker: MockerFixture, verbose: int, quiet: bool, expected_level: int) -> None:
    mock_config = mocker.patch("investigate_common.logging.basicConfig")
    investigate_common.setup_logging(verbose, quiet)
    mock_config.assert_called_once_with(
        level=expected_level, format="%(levelname)s: %(message)s", stream=sys.stderr, force=True
    )


def test_parse_job_url_valid() -> None:
    assert investigate_common.parse_job_url("https://openqa.opensuse.org/tests/123") == (
        123,
        "https",
        "openqa.opensuse.org",
    )
    assert investigate_common.parse_job_url("http://openqa.example.com/t456") == (
        456,
        "http",
        "openqa.example.com",
    )


def test_parse_job_url_invalid() -> None:
    with pytest.raises(ValueError, match="Invalid job URL"):
        investigate_common.parse_job_url("invalid_url")
    with pytest.raises(ValueError, match="Could not extract job ID from URL path"):
        investigate_common.parse_job_url("https://openqa.opensuse.org/")
    with pytest.raises(ValueError, match="Could not extract numeric job ID"):
        investigate_common.parse_job_url("https://openqa.opensuse.org/tests/abc")


def test_parse_job_url_and_base() -> None:
    base, job_id = investigate_common.parse_job_url_and_base("https://openqa.opensuse.org/tests/1234")
    assert base == "https://openqa.opensuse.org"
    assert job_id == "1234"

    base, job_id = investigate_common.parse_job_url_and_base("http://test.openqa/t5678")
    assert base == "http://test.openqa"
    assert job_id == "5678"

    base, job_id = investigate_common.parse_job_url_and_base("9999")
    assert base == "https://openqa.opensuse.org"
    assert job_id == "9999"

    base, job_id = investigate_common.parse_job_url_and_base("t8888")
    assert base == "https://openqa.opensuse.org"
    assert job_id == "8888"

    # Fallback path when URL parse raises ValueError
    base, job_id = investigate_common.parse_job_url_and_base("https://openqa.opensuse.org/nonnumeric")
    assert base == "https://openqa.opensuse.org"
    assert job_id == "nonnumeric"


def test_transient_http_error() -> None:
    resp = httpx.Response(503)
    err = investigate_common.TransientHTTPError(resp)
    assert str(err) == "HTTP 503"
    assert err.response is resp


def test_retry_transport_success(mocker: MockerFixture) -> None:
    mock_handle = mocker.patch("investigate_common.httpx.HTTPTransport.handle_request")
    resp200 = MagicMock(spec=httpx.Response, status_code=200)
    mock_handle.return_value = resp200

    transport = investigate_common.RetryTransport(retries=2)
    req = httpx.Request("GET", "https://example.com")
    res = transport.handle_request(req)
    assert res == resp200
    mock_handle.assert_called_once_with(req)


def test_retry_transport_retry_after(mocker: MockerFixture) -> None:
    mock_handle = mocker.patch("investigate_common.httpx.HTTPTransport.handle_request")
    mock_sleep = mocker.patch("time.sleep")

    resp429 = MagicMock(spec=httpx.Response, status_code=429, headers={"Retry-After": "1.5"})
    resp200 = MagicMock(spec=httpx.Response, status_code=200)
    mock_handle.side_effect = [resp429, resp200]

    transport = investigate_common.RetryTransport(retries=2)
    req = httpx.Request("GET", "https://example.com")
    res = transport.handle_request(req)
    assert res == resp200
    mock_sleep.assert_called_once_with(1.5)
    resp429.close.assert_called_once()


def test_retry_transport_retry_after_date(mocker: MockerFixture) -> None:
    mock_handle = mocker.patch("investigate_common.httpx.HTTPTransport.handle_request")
    mock_sleep = mocker.patch("time.sleep")

    future = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=10)
    date_str = format_datetime(future)

    resp503 = MagicMock(spec=httpx.Response, status_code=503, headers={"Retry-After": date_str})
    resp200 = MagicMock(spec=httpx.Response, status_code=200)
    mock_handle.side_effect = [resp503, resp200]

    transport = investigate_common.RetryTransport(retries=2)
    req = httpx.Request("GET", "https://example.com")
    res = transport.handle_request(req)
    assert res == resp200
    assert mock_sleep.call_count == 1
    # Sleep delay should be approx 10 seconds
    assert 8.0 <= mock_sleep.call_args[0][0] <= 12.0


def test_retry_transport_exhausted(mocker: MockerFixture) -> None:
    mock_handle = mocker.patch("investigate_common.httpx.HTTPTransport.handle_request")
    mocker.patch("time.sleep")

    resp500 = MagicMock(spec=httpx.Response, status_code=500, headers={})
    mock_handle.return_value = resp500

    transport = investigate_common.RetryTransport(retries=1)
    req = httpx.Request("GET", "https://example.com")
    res = transport.handle_request(req)
    assert res == resp500


def test_retry_transport_branches(mocker: MockerFixture, caplog: pytest.LogCaptureFixture) -> None:
    transport = investigate_common.RetryTransport(retries=1)

    # _wait_strategy with None outcome
    assert transport._wait_strategy(Mock(outcome=None)) == 0.0

    # _wait_strategy with non-TransientHTTPError
    mock_state_other = Mock(outcome=Mock(exception=Mock(return_value=ValueError("other"))))
    assert transport._wait_strategy(mock_state_other) == 0.0

    # _wait_strategy with unparseable date Retry-After
    resp_bad_date = httpx.Response(429, headers={"Retry-After": "invalid-date"})
    mock_state_bad_date = Mock(
        attempt_number=1,
        outcome=Mock(exception=Mock(return_value=investigate_common.TransientHTTPError(resp_bad_date))),
    )
    with caplog.at_level(logging.DEBUG):
        delay = transport._wait_strategy(mock_state_bad_date)
    assert delay == 1.0
    assert "Failed to parse Retry-After header as date" in caplog.text

    # _before_sleep with None outcome or next_action
    transport._before_sleep(Mock(outcome=None, next_action=None))

    # _before_sleep with non-TransientHTTPError
    transport._before_sleep(mock_state_other)


def test_fetch_json_success() -> None:
    mock_client = MagicMock(spec=httpx.Client)
    mock_resp = Mock()
    mock_resp.json.return_value = {"key": "value"}
    mock_client.get.return_value = mock_resp

    data = investigate_common.fetch_json(mock_client, "https://example.com/api", params={"p": 1}, timeout=30.0)
    assert data == {"key": "value"}
    mock_client.get.assert_called_once_with("https://example.com/api", params={"p": 1}, timeout=30.0)


def test_fetch_json_failure() -> None:
    mock_client = MagicMock(spec=httpx.Client)
    mock_client.get.side_effect = httpx.HTTPError("Boom")

    assert investigate_common.fetch_json(mock_client, "https://example.com/api") == {}
    assert investigate_common.fetch_json(mock_client, "https://example.com/api", default=[]) == []


def test_fetch_text_success() -> None:
    mock_client = MagicMock(spec=httpx.Client)
    mock_resp = MagicMock()
    mock_resp.iter_lines.return_value = [f"line {i}" for i in range(10)]
    mock_client.stream.return_value.__enter__.return_value = mock_resp

    text = investigate_common.fetch_text(mock_client, "https://example.com/log.txt", max_lines=5, timeout=10.0)
    assert text == "line 5\nline 6\nline 7\nline 8\nline 9"
    mock_client.stream.assert_called_once_with("GET", "https://example.com/log.txt", timeout=10.0)


def test_fetch_text_failure() -> None:
    mock_client = MagicMock(spec=httpx.Client)
    mock_client.stream.side_effect = httpx.HTTPError("Boom")

    assert not investigate_common.fetch_text(mock_client, "https://example.com/log.txt")


def test_post_comment_success(mocker: MockerFixture) -> None:
    mock_run = mocker.patch("investigate_common.subprocess.run")
    investigate_common.post_comment("https://openqa.opensuse.org", 123, "comment text", user_agent="test-agent")
    mock_run.assert_called_once_with(
        [
            "openqa-cli",
            "api",
            "--header",
            "User-Agent: test-agent",
            "--host",
            "https://openqa.opensuse.org",
            "-X",
            "POST",
            "jobs/123/comments",
            "text=comment text",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_post_comment_failure(mocker: MockerFixture, caplog: pytest.LogCaptureFixture) -> None:
    mocker.patch(
        "investigate_common.subprocess.run",
        side_effect=subprocess.CalledProcessError(1, ["openqa-cli"], stderr="Auth failed"),
    )
    with caplog.at_level(logging.ERROR):
        investigate_common.post_comment("https://openqa.opensuse.org", 123, "comment text")
    assert "Failed to post comment: Auth failed" in caplog.text
