"""Chromium backend that admits scoped redirects without replaying requests.

Playwright routes only the first URL of a redirect chain. A separate CDP Fetch
request-stage guard checks every document hop; the ordinary context route still
rejects a popup's first request before its page/frame exists.
"""

from .agent_browser import BrowserAgentBackend


class RedirectingBrowserBackend(BrowserAgentBackend):
    def __init__(self, page, **kwargs):
        self._cdp = None
        super().__init__(page, **kwargs)
        try:
            self._cdp = page.context.new_cdp_session(page)
            self._main_frame_id = self._cdp.send("Page.getFrameTree")["frameTree"]["frame"]["id"]
            self._cdp.on("Fetch.requestPaused", self._document_request)
            self._cdp.send("Fetch.enable", {"patterns": [
                {"resourceType": "Document", "requestStage": "Request"},
            ]})
        except Exception:
            self.close()
            raise RuntimeError("scoped redirect interception requires Chromium CDP") from None

    def _route_request(self, route):
        request = route.request
        if request.is_navigation_request():
            try:
                owned = request.frame == self.page.main_frame
            except Exception:
                owned = False
            if not owned or not self._allowed(request.url):
                self._blocked += 1
                route.abort("blockedbyclient")
                return
        # No route.fetch: no duplicate POST, request probing, or URL rewriting.
        route.continue_()

    def _document_request(self, event):
        allowed = (event.get("frameId") == self._main_frame_id
                   and self._allowed(event["request"]["url"]))
        if allowed:
            self._cdp.send("Fetch.continueRequest", {"requestId": event["requestId"]})
        else:
            self._blocked += 1
            self._cdp.send("Fetch.failRequest", {"requestId": event["requestId"],
                                                "errorReason": "BlockedByClient"})

    def close(self):
        try:
            if self._cdp is not None:
                session, self._cdp = self._cdp, None
                try:
                    session.send("Fetch.disable")
                finally:
                    session.detach()
        finally:
            super().close()
