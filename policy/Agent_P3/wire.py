"""Wire adapters that are transport-only extensions to inspect-robots-agent."""

from __future__ import annotations

import json
import urllib.parse

import httpx


class AzureChatTransport(httpx.BaseTransport):
    """Adapt upstream ChatClient requests to Azure's deployment URL/auth.

    The upstream client still owns request construction, retries, parsing,
    usage accounting, and replay-grade wire capture.
    """

    def __init__(
        self,
        *,
        base_url: str,
        deployment: str,
        api_version: str,
        inner: httpx.BaseTransport | None = None,
    ) -> None:
        self._base = urllib.parse.urlsplit(base_url.rstrip("/"))
        self._deployment = deployment
        self._api_version = api_version
        self._inner = inner or httpx.HTTPTransport()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        path = (
            self._base.path.rstrip("/")
            + "/openai/deployments/"
            + urllib.parse.quote(self._deployment, safe="")
            + "/chat/completions"
        )
        query = urllib.parse.urlencode({"api-version": self._api_version})
        url = httpx.URL(
            urllib.parse.urlunsplit(
                (self._base.scheme, self._base.netloc, path, query, "")
            )
        )
        headers = dict(request.headers)
        authorization = headers.pop("authorization", "")
        if authorization.lower().startswith("bearer "):
            headers["api-key"] = authorization[7:]
        body = json.loads(request.content.decode("utf-8"))
        body["model"] = self._deployment
        forwarded = httpx.Request(
            request.method,
            url,
            headers=headers,
            content=json.dumps(body).encode("utf-8"),
            extensions=request.extensions,
        )
        return self._inner.handle_request(forwarded)

    def close(self) -> None:
        self._inner.close()
