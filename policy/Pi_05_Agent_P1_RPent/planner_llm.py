"""Planner LLM backends: local/remote Qwen and Azure OpenAI GPT."""

from __future__ import annotations

import os
import random
import time
from typing import Any
from uuid import uuid4

from .qwen_client import QwenClient, parse_chat_message

DEFAULT_GPT_ENDPOINT = "https://aidp.bytedance.net/api/modelhub/online/v2/crawl"
DEFAULT_GPT_API_VERSION = "2024-03-01-preview"
DEFAULT_GPT_MODEL = "gpt-5.5-2026-04-24"
QWEN_ONLY_CHAT_KEYS = ("enable_thinking",)

# AIDP answers a busy vendor pool with HTTP 429 that can persist for minutes, so
# the ladder is capped per attempt and bounded in total rather than short.
DEFAULT_GPT_MAX_RETRIES = 12
DEFAULT_GPT_RETRY_CAP_S = 120.0
DEFAULT_GPT_RETRY_BUDGET_S = 1800.0
DEFAULT_GPT_RETRY_JITTER = 0.1
TRANSIENT_ERROR_HINTS = (
    "apiconnectionerror",
    "apitimeouterror",
    "connectionerror",
    "internalservererror",
    "timeout",
)


def _status_code(error: Exception) -> int | None:
    status = getattr(error, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def _retry_after_seconds(error: Exception) -> float | None:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    try:
        raw = headers.get("retry-after") or headers.get("Retry-After")
    except AttributeError:
        return None
    if raw is None:
        return None
    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        return None
    return seconds if seconds >= 0 else None


def _is_retryable(error: Exception) -> bool:
    status = _status_code(error)
    if status is not None:
        return status == 429 or 500 <= status < 600
    # Transport failures carry no status; the request never reached the model.
    if isinstance(error, (TimeoutError, ConnectionError)):
        return True
    name = type(error).__name__.lower()
    return any(hint in name for hint in TRANSIENT_ERROR_HINTS)


def qwen_api_key() -> str:
    for key in ("DASHSCOPE_API_KEY", "QWEN_API_KEY"):
        value = os.environ.get(key, "").strip()
        if value:
            return value
    return ""


def gpt_api_key() -> str:
    for key in ("RPENT_GPT_API_KEY", "AZURE_OPENAI_API_KEY", "OPENAI_API_KEY"):
        value = os.environ.get(key, "").strip()
        if value:
            return value
    return ""


def planner_backend() -> str:
    explicit = os.environ.get("RPENT_LLM_BACKEND", "").strip().lower()
    if explicit in {"azure", "azure_openai", "gpt", "openai"}:
        return "azure_openai"
    if explicit in {"qwen", "dashscope"}:
        return "qwen"
    if gpt_api_key() and not qwen_api_key():
        return "azure_openai"
    return "qwen"


def remote_planner_configured() -> bool:
    backend = planner_backend()
    if backend == "azure_openai":
        return bool(gpt_api_key())
    return bool(qwen_api_key())


class AzureOpenAIPlannerClient:
    """ByteDance AIDP AzureOpenAI wrapper for planner and vision calls."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        endpoint: str | None = None,
        api_version: str | None = None,
        model: str | None = None,
    ) -> None:
        self.api_key = (api_key if api_key is not None else gpt_api_key()).strip()
        self.endpoint = (
            endpoint
            or os.environ.get("RPENT_GPT_ENDPOINT")
            or os.environ.get("AZURE_OPENAI_ENDPOINT")
            or DEFAULT_GPT_ENDPOINT
        ).rstrip("/")
        self.api_version = (
            api_version
            or os.environ.get("RPENT_GPT_API_VERSION")
            or os.environ.get("OPENAI_API_VERSION")
            or DEFAULT_GPT_API_VERSION
        )
        self.model = (
            model
            or os.environ.get("RPENT_GPT_MODEL")
            or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
            or DEFAULT_GPT_MODEL
        )
        self.logid = os.environ.get("RPENT_GPT_LOGID", "").strip() or uuid4().hex
        self.max_retries = max(
            0,
            int(
                os.environ.get("RPENT_GPT_MAX_RETRIES", str(DEFAULT_GPT_MAX_RETRIES))
            ),
        )
        self.retry_cap_s = max(
            0.0,
            float(
                os.environ.get("RPENT_GPT_RETRY_CAP_S", str(DEFAULT_GPT_RETRY_CAP_S))
            ),
        )
        self.retry_budget_s = max(
            0.0,
            float(
                os.environ.get(
                    "RPENT_GPT_RETRY_BUDGET_S", str(DEFAULT_GPT_RETRY_BUDGET_S)
                )
            ),
        )
        self.retry_jitter = max(
            0.0,
            float(
                os.environ.get("RPENT_GPT_RETRY_JITTER", str(DEFAULT_GPT_RETRY_JITTER))
            ),
        )

    def _retry_delay(self, attempt: int, error: Exception) -> float:
        """Vendor-requested delay when offered, else capped exponential backoff."""
        retry_after = _retry_after_seconds(error)
        if retry_after is not None:
            return retry_after
        delay = min(self.retry_cap_s, 2.0**attempt)
        if self.retry_jitter:
            delay += delay * random.uniform(0.0, self.retry_jitter)
        return delay

    def available(self) -> bool:
        return bool(self.api_key)

    def _client(self):
        try:
            from openai import AzureOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Azure OpenAI backend requires the openai package. "
                "Install it in the Pi_05 uv env or set RPENT_LLM_BACKEND=qwen."
            ) from exc
        return AzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.endpoint,
            api_version=self.api_version,
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = "auto",
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError(
                "GPT API key missing. Set RPENT_GPT_API_KEY or AZURE_OPENAI_API_KEY."
            )
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice
        max_tokens = os.environ.get("RPENT_GPT_MAX_TOKENS", "").strip()
        if max_tokens:
            payload["max_tokens"] = int(max_tokens)
        temperature = os.environ.get("RPENT_GPT_TEMPERATURE", "").strip()
        if temperature:
            payload["temperature"] = float(temperature)
        payload.update(extra or {})
        for key in QWEN_ONLY_CHAT_KEYS:
            payload.pop(key, None)
        headers = dict(payload.pop("extra_headers", {}) or {})
        headers.setdefault("X-TT-LOGID", self.logid)
        payload["extra_headers"] = headers
        last_error: Exception | None = None
        waited_s = 0.0
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client().chat.completions.create(**payload)
                if hasattr(response, "model_dump"):
                    return response.model_dump()
                return dict(response)
            except Exception as exc:
                last_error = exc
                if not _is_retryable(exc) or attempt >= self.max_retries:
                    raise
                wait_s = self._retry_delay(attempt, exc)
                if waited_s + wait_s > self.retry_budget_s:
                    print(
                        f"[P1-RPent] GPT retry budget {self.retry_budget_s:.0f}s "
                        f"exhausted after {waited_s:.0f}s; giving up",
                        flush=True,
                    )
                    raise
                waited_s += wait_s
                status = _status_code(exc)
                reason = f"HTTP {status}" if status else type(exc).__name__
                print(
                    f"[P1-RPent] GPT retry {attempt + 1}/{self.max_retries} "
                    f"after {reason}; sleeping {wait_s:.0f}s "
                    f"(waited {waited_s:.0f}s/{self.retry_budget_s:.0f}s)",
                    flush=True,
                )
                time.sleep(wait_s)
        raise RuntimeError(f"GPT request failed: {last_error}") from last_error

    def message_text_and_tools(
        self, result: dict[str, Any]
    ) -> tuple[str, list[dict[str, Any]]]:
        return parse_chat_message(result)


def create_planner_llm() -> QwenClient | AzureOpenAIPlannerClient:
    if planner_backend() == "azure_openai":
        return AzureOpenAIPlannerClient()
    return QwenClient()
