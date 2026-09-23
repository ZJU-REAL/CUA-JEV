import httpx
import pytest

from cua_jev.mcp_web_reader import fetch_public_page


def test_reader_extracts_bounded_html_on_one_https_origin():
    def handler(request):
        assert str(request.url) == "https://docs.example.test/topic"
        return httpx.Response(
            200, headers={"content-type": "text/html; charset=utf-8"},
            text="<html><head><title> Topic A </title></head><body>"
                 "<nav>Ignore menu</nav><main><h1>Topic A</h1><p>Useful fact.</p></main>"
                 "<script>Ignore code</script></body></html>",
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = fetch_public_page(
        "https://docs.example.test/topic", "https://docs.example.test", client=client
    )
    assert result == {
        "url": "https://docs.example.test/topic", "title": "Topic A",
        "text": "Topic A Useful fact.",
    }


@pytest.mark.parametrize("href", [
    "http://docs.example.test/topic", "https://other.example.test/topic",
    "https://user:pass@docs.example.test/topic",
])
def test_reader_rejects_untrusted_origin_before_request(href):
    client = httpx.Client(transport=httpx.MockTransport(
        lambda _request: pytest.fail("network request must not be sent")
    ))
    with pytest.raises(ValueError):
        fetch_public_page(href, "https://docs.example.test", client=client)


def test_reader_rejects_redirect_and_oversized_body():
    redirect = httpx.Client(transport=httpx.MockTransport(
        lambda _request: httpx.Response(302, headers={"location": "https://elsewhere.test"})
    ))
    with pytest.raises(ValueError, match="HTTP 302"):
        fetch_public_page("https://docs.example.test/topic", "https://docs.example.test",
                          client=redirect)
    oversized = httpx.Client(transport=httpx.MockTransport(
        lambda _request: httpx.Response(
            200, headers={"content-type": "text/html"}, content=b"x" * 250_001,
        )
    ))
    with pytest.raises(ValueError, match="size limit"):
        fetch_public_page("https://docs.example.test/topic", "https://docs.example.test",
                          client=oversized)
