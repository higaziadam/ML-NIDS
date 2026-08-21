"""Regression tests for Docker API smoke-test readiness handling."""

from scripts import docker_api_e2e


def test_wait_for_health_retries_a_transient_connection_reset(monkeypatch) -> None:
    """A container startup reset must not fail the whole integration test."""
    responses = iter([ConnectionResetError("container is still starting"), {"status": "ok"}])

    def fake_request_json(*_args, **_kwargs):
        result = next(responses)
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(docker_api_e2e, "_request_json", fake_request_json)
    monkeypatch.setattr(docker_api_e2e.time, "sleep", lambda _seconds: None)

    assert docker_api_e2e._wait_for_health("http://127.0.0.1:8000", deadline_seconds=1) == {
        "status": "ok"
    }
