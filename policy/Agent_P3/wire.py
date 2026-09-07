"""Provider resolution and the HTTP wires P3 speaks.

Raw HTTP against each vendor's own endpoint rather than their SDKs. P3 exists
to measure a model as its vendor exposes it, so the native surface is the point
rather than an implementation detail: Anthropic's `/messages` carries thinking
blocks and `cache_control`, and only the compat layer would hide them. Going
through `urllib` also keeps this adapter free of a dependency the eval
environments do not already have, and makes every request injectable in tests.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

Transport = Callable[[str, Mapping[str, str], bytes, float], tuple[int, bytes]]

# prefix -> (base_url, key env var, native wire)
DIRECT_PROVIDERS: dict[str, tuple[str, str, str]] = {
    "anthropic": ("https://api.anthropic.com/v1", "ANTHROPIC_API_KEY", "messages"),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY", "chat"),
    "google": (
        "https://generativelanguage.googleapis.com/v1beta/openai",
        "GEMINI_API_KEY",
        "chat",
    ),
    "x-ai": ("https://api.x.ai/v1", "XAI_API_KEY", "chat"),
    "deepseek": ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY", "chat"),
    "qwen": (
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "DASHSCOPE_API_KEY",
        "chat",
    ),
}

OPENROUTER = ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "chat")


class ConfigError(RuntimeError):
    """Raised for a misconfiguration the operator has to fix, not a retry."""


@dataclass(frozen=True)
class Provider:
    model: str
    base_url: str
    api_key: str
    wire: str


@dataclass
class Reply:
    """One assistant turn, reduced to what the policy needs."""

    text: str | None = None
    tool_name: str | None = None
    tool_arguments: str | None = None
    usage: dict[str, int] = field(default_factory=dict)


def resolve_provider(
    model: str | None = None,
    *,
    base_url: str | None = None,
    api_key_env: str | None = None,
    wire: str | None = None,
    env: Mapping[str, str] | None = None,
) -> Provider:
    """First match wins: explicit base_url, then the prefix table, then OpenRouter."""
    env = os.environ if env is None else env
    model = model or env.get("P3_MODEL") or ""
    if not model:
        raise ConfigError(
            "No model selected. fix: set P3_MODEL, e.g. "
            "P3_MODEL=anthropic/claude-sonnet-4-20250514"
        )

    if base_url or env.get("P3_BASE_URL"):
        resolved_url = base_url or env["P3_BASE_URL"]
        key_env = api_key_env or env.get("P3_API_KEY_ENV") or "P3_API_KEY"
        return Provider(
            model=model.split("/", 1)[-1] if "/" in model else model,
            base_url=resolved_url.rstrip("/"),
            api_key=env.get(key_env, ""),
            wire=wire or env.get("P3_WIRE") or "chat",
        )

    prefix, _, tail = model.partition("/")
    entry = DIRECT_PROVIDERS.get(prefix)
    if entry is not None and tail:
        url, key_env, native = entry
        key = env.get(api_key_env or key_env, "")
        if key:
            return Provider(
                model=tail,
                base_url=url,
                api_key=key,
                wire=wire or env.get("P3_WIRE") or native,
            )

    router_key = env.get(OPENROUTER[1], "")
    if router_key:
        return Provider(
            model=model,
            base_url=OPENROUTER[0],
            api_key=router_key,
            wire=wire or env.get("P3_WIRE") or OPENROUTER[2],
        )

    wanted = entry[1] if entry else "a provider"
    raise ConfigError(
        f"No API key for {model!r}. fix: export {wanted} for the direct "
        f"endpoint, or OPENROUTER_API_KEY to route it, or set P3_BASE_URL for "
        f"a local server."
    )


def _urllib_post(
    url: str, headers: Mapping[str, str], body: bytes, timeout: float
) -> tuple[int, bytes]:
    request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status), response.read()
    except urllib.error.HTTPError as error:
        return int(error.code), error.read()


class _Client:
    """Shared retry policy. Subclasses own the payload and the parsing."""

    #: 4xx other than these is our request's fault, so retrying cannot help.
    RETRY_STATUS = frozenset({408, 409, 429})

    def __init__(
        self,
        provider: Provider,
        *,
        timeout_s: float = 120.0,
        max_retries: int = 3,
        backoff_s: float = 1.0,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.provider = provider
        self.timeout_s = timeout_s
        self.max_retries = max(1, int(max_retries))
        self.backoff_s = backoff_s
        self._post = transport or _urllib_post
        self._sleep = sleep

    def _endpoint(self) -> str:
        raise NotImplementedError

    def _headers(self) -> dict[str, str]:
        raise NotImplementedError

    def _body(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        system: str | None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def _parse(self, payload: Mapping[str, Any]) -> Reply:
        raise NotImplementedError

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] = (),
        *,
        system: str | None = None,
    ) -> Reply:
        url = f"{self.provider.base_url}{self._endpoint()}"
        body = json.dumps(self._body(messages, tools, system)).encode("utf-8")
        headers = self._headers()
        last = ""
        for attempt in range(self.max_retries):
            try:
                status, raw = self._post(url, headers, body, self.timeout_s)
            except Exception as error:  # noqa: BLE001 - any transport failure retries
                last = f"{type(error).__name__}: {error}"
            else:
                if status == 200:
                    return self._parse(json.loads(raw.decode("utf-8")))
                last = f"HTTP {status}: {raw.decode('utf-8', 'replace')[:400]}"
                if 400 <= status < 500 and status not in self.RETRY_STATUS:
                    raise RuntimeError(f"LLM request rejected — {last}")
            if attempt + 1 < self.max_retries:
                self._sleep(self.backoff_s * 2**attempt)
        raise RuntimeError(
            f"LLM request failed after {self.max_retries} attempts — {last}"
        )


class ChatClient(_Client):
    """OpenAI-compatible `/chat/completions`."""

    def _endpoint(self) -> str:
        return "/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.provider.api_key:
            headers["Authorization"] = f"Bearer {self.provider.api_key}"
        return headers

    def _body(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        system: str | None,
    ) -> dict[str, Any]:
        turns = list(messages)
        if system is not None:
            turns = [{"role": "system", "content": system}, *turns]
        body: dict[str, Any] = {"model": self.provider.model, "messages": turns}
        if tools:
            body["tools"] = list(tools)
            body["tool_choice"] = "required"
        return body

    def _parse(self, payload: Mapping[str, Any]) -> Reply:
        message = payload["choices"][0]["message"]
        calls = message.get("tool_calls") or []
        usage_in = payload.get("usage") or {}
        cached = (usage_in.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        usage = {
            "input_tokens": int(usage_in.get("prompt_tokens", 0)),
            "output_tokens": int(usage_in.get("completion_tokens", 0)),
            "cache_read_input_tokens": int(cached),
        }
        if not calls:
            return Reply(text=message.get("content"), usage=usage)
        call = calls[0]["function"]
        return Reply(
            text=message.get("content"),
            tool_name=str(call["name"]),
            tool_arguments=str(call["arguments"]),
            usage=usage,
        )


class MessagesClient(_Client):
    """Anthropic's native `/messages`, with prompt caching on the static head.

    The system prompt and the action tool schema are identical on every request
    of an episode while the observation suffix changes, so one ephemeral
    breakpoint after the system block turns the fixed part into a cache read.
    """

    def __init__(self, *args: Any, max_output_tokens: int = 4096, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.max_output_tokens = max_output_tokens

    def _endpoint(self) -> str:
        return "/messages"

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "x-api-key": self.provider.api_key,
            "anthropic-version": "2023-06-01",
        }

    def _body(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        system: str | None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.provider.model,
            "max_tokens": self.max_output_tokens,
            "messages": [_to_anthropic_turn(turn) for turn in messages],
        }
        if system is not None:
            body["system"] = [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        if tools:
            body["tools"] = [_to_anthropic_tool(tool) for tool in tools]
            body["tool_choice"] = {"type": "any"}
        return body

    def _parse(self, payload: Mapping[str, Any]) -> Reply:
        usage_in = payload.get("usage") or {}
        usage = {
            "input_tokens": int(usage_in.get("input_tokens", 0)),
            "output_tokens": int(usage_in.get("output_tokens", 0)),
            "cache_read_input_tokens": int(
                usage_in.get("cache_read_input_tokens", 0)
            ),
            "cache_creation_input_tokens": int(
                usage_in.get("cache_creation_input_tokens", 0)
            ),
        }
        text: str | None = None
        name: str | None = None
        arguments: str | None = None
        for block in payload.get("content") or []:
            if block.get("type") == "text" and text is None:
                text = block.get("text")
            elif block.get("type") == "tool_use" and name is None:
                name = str(block.get("name"))
                arguments = json.dumps(block.get("input") or {})
        return Reply(text=text, tool_name=name, tool_arguments=arguments, usage=usage)


def _to_anthropic_tool(tool: Mapping[str, Any]) -> dict[str, Any]:
    function = tool.get("function") or tool
    return {
        "name": function["name"],
        "description": function.get("description", ""),
        "input_schema": function.get("parameters") or {"type": "object"},
    }


def _to_anthropic_turn(turn: Mapping[str, Any]) -> dict[str, Any]:
    """Translate one OpenAI-format turn into Messages content blocks."""
    content = turn.get("content")
    if isinstance(content, str):
        return {"role": turn["role"], "content": content}
    blocks: list[dict[str, Any]] = []
    for part in content or []:
        if part.get("type") == "text":
            blocks.append({"type": "text", "text": part["text"]})
        elif part.get("type") == "image_url":
            url = part["image_url"]["url"]
            header, _, data = url.partition(",")
            media_type = header.split(";")[0].removeprefix("data:") or "image/png"
            blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": data,
                    },
                }
            )
    return {"role": turn["role"], "content": blocks}


def create_client(provider: Provider, **kwargs: Any) -> _Client:
    if provider.wire == "messages":
        return MessagesClient(provider, **kwargs)
    if provider.wire == "chat":
        return ChatClient(provider, **kwargs)
    raise ConfigError(
        f"Unknown wire {provider.wire!r}. fix: set P3_WIRE to 'chat' or 'messages'."
    )
