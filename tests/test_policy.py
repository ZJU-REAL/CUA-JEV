import httpx
import pytest

from cua_jev.errors import PolicyError
from cua_jev.models import ActionCandidate, Channel, Observation
from cua_jev.policy import JevPolicy, RulePolicy


def candidates():
    return [
        ActionCandidate("gui", Channel.GUI, "uia.invoke", "Click a button"),
        ActionCandidate("mcp", Channel.MCP, "mcp.filesystem.read", "Read with MCP"),
    ]


def test_rule_policy_prefers_structured_read_only_channel():
    obs = Observation("read", "read file", {})
    assert RulePolicy().choose(obs, candidates()).candidate_id == "mcp"


def test_jev_policy_validates_and_returns_choice():
    def handler(request):
        assert "secret" not in request.content.decode()
        return httpx.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "answers": {
                    "action": {
                        "choice": "mcp",
                        "probabilities": {"gui": 0.1, "mcp": 0.9},
                        "confidence": 0.8,
                    }
                },
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    policy = JevPolicy(api_key="secret", client=client, retries=0)
    decision = policy.choose(Observation("read", "read file", {}), candidates())
    assert decision.candidate_id == "mcp"
    assert decision.probabilities["mcp"] == 0.9


def test_jev_policy_rejects_inconsistent_choice():
    def handler(_request):
        return httpx.Response(
            200,
            json={
                "answers": {
                    "action": {
                        "choice": "gui",
                        "probabilities": {"gui": 0.1, "mcp": 0.9},
                        "confidence": 0.8,
                    }
                }
            },
        )

    policy = JevPolicy(
        api_key="secret", client=httpx.Client(transport=httpx.MockTransport(handler)), retries=0
    )
    with pytest.raises(PolicyError, match="invalid Jev"):
        policy.choose(Observation("read", "read file", {}), candidates())


def test_jev_policy_retries_transient_connection_failures():
    attempts = 0
    sleeps = []

    def handler(request):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise httpx.ConnectError("temporary outage", request=request)
        return httpx.Response(
            200,
            json={
                "answers": {
                    "action": {
                        "choice": "mcp",
                        "probabilities": {"gui": 0.1, "mcp": 0.9},
                        "confidence": 0.8,
                    }
                }
            },
        )

    policy = JevPolicy(
        api_key="secret",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        retries=4,
        sleep=sleeps.append,
    )
    decision = policy.choose(Observation("read", "read file", {}), candidates())

    assert decision.candidate_id == "mcp"
    assert attempts == 3
    assert sleeps == [0.5, 1.0]


def test_jev_policy_reports_actionable_connection_error():
    def handler(request):
        raise httpx.ConnectError("secret network details", request=request)

    policy = JevPolicy(
        api_key="secret",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        retries=1,
        sleep=lambda _: None,
    )
    with pytest.raises(PolicyError, match="temporarily unreachable after 2 attempts; retry") as error:
        policy.choose(Observation("read", "read file", {}), candidates())
    assert "secret network details" not in str(error.value)


def test_jev_policy_can_transparently_fall_back_on_transport_failure():
    def handler(request):
        raise httpx.ConnectError("temporary outage", request=request)

    policy = JevPolicy(
        api_key="secret",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        retries=0,
        fallback_on_transport=True,
    )
    decision = policy.choose(Observation("read", "read file", {}), candidates())

    assert decision.candidate_id == "mcp"
    assert decision.model == "jev-unavailable/rule-fallback"
    assert policy.last_exchange["fallback"]["policy"] == "rule-baseline"


def test_jev_policy_never_falls_back_on_authentication_failure():
    def handler(_request):
        return httpx.Response(401, json={"detail": "unauthorized"})

    policy = JevPolicy(
        api_key="secret",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        retries=2,
        fallback_on_transport=True,
    )
    with pytest.raises(PolicyError, match="authentication failed"):
        policy.choose(Observation("read", "read file", {}), candidates())
    assert "fallback" not in policy.last_exchange


@pytest.mark.parametrize("failure,code", [
    (httpx.ConnectTimeout, "connect_timeout"),
    (httpx.ReadTimeout, "timeout"),
    (httpx.ConnectError, "connection_error"),
])
def test_transport_failure_has_typed_sanitized_diagnostics_and_resets(failure, code):
    failing = True

    def handler(request):
        if failing:
            raise failure("secret endpoint and key", request=request)
        return httpx.Response(200, json={"answers": {"action": {
            "choice": "mcp", "probabilities": {"gui": 0.1, "mcp": 0.9}, "confidence": 0.8,
        }}})

    policy = JevPolicy(api_key="secret", retries=1, sleep=lambda _: None,
                       client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(PolicyError):
        policy.choose(Observation("test", "test", {}), candidates())
    assert policy.last_failure == {
        "code": code, "attempts": 2, "retryable": True, "http_status": None,
    }
    assert "secret" not in str(policy.last_failure)
    failing = False
    policy.choose(Observation("test", "test", {}), candidates())
    assert policy.last_failure == {}


@pytest.mark.parametrize("body", [[], {"answers": {"action": {"choice": "invented"}}}])
def test_invalid_response_records_failure_without_raw_provider_data(body):
    policy = JevPolicy(api_key="secret", retries=0, client=httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body)),
    ))
    with pytest.raises(PolicyError):
        policy.choose(Observation("test", "test", {}), candidates())
    assert policy.last_failure == {
        "code": "invalid_response", "attempts": 1, "retryable": False, "http_status": None,
    }


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422, 451])
def test_permanent_http_failure_stops_without_retry_or_fallback(status):
    requests, sleeps = [], []

    def handler(request):
        requests.append(request)
        return httpx.Response(status, text="private provider details and credential")

    policy = JevPolicy(
        api_key="secret", retries=4, fallback_on_transport=True, sleep=sleeps.append,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(PolicyError) as error:
        policy.choose(Observation("test", "test", {}), candidates())
    assert len(requests) == 1
    assert sleeps == []
    assert policy.last_failure["attempts"] == 1
    assert policy.last_failure["retryable"] is False
    assert "fallback" not in policy.last_exchange
    assert "private provider" not in str(error.value)
    assert "private provider" not in str(policy.last_exchange)
    assert policy.last_exchange["retry_history"][0]["http_status"] == status


def test_recovered_service_call_keeps_sanitized_retry_diagnostics():
    requests, sleeps = [], []

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(503, text="private provider error")
        if len(requests) == 2:
            raise httpx.ReadTimeout("private provider timeout", request=request)
        return httpx.Response(200, json={"answers": {"action": {
            "choice": "mcp", "probabilities": {"gui": 0.1, "mcp": 0.9}, "confidence": 0.8,
        }}})

    policy = JevPolicy(
        api_key="secret", retries=4, sleep=sleeps.append,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    policy.choose(Observation("test", "test", {}), candidates())
    history = policy.last_exchange["retry_history"]
    assert [(item["attempt"], item["code"], item["http_status"]) for item in history] == [
        (1, "http_status", 503), (2, "timeout", None),
    ]
    assert all(item["elapsed_ms"] >= 0 for item in history)
    assert sleeps == [0.5, 1.0]
    assert policy.last_exchange["attempts"] == 3
    assert policy.last_failure == {}
    assert "private provider" not in str(history)
    policy.choose(Observation("test", "test", {}), candidates())
    assert policy.last_exchange["retry_history"] == []
