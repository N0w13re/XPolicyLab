import sys
import types

import pytest

from XPolicyLab.policy.Pi_05_Agent_P1_RPent.qwen_client import QwenClient


def _clear_llm_env(monkeypatch):
    for name in (
        "RPENT_LLM_BACKEND",
        "RPENT_GPT_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "OPENAI_API_KEY",
        "DASHSCOPE_API_KEY",
        "QWEN_API_KEY",
        "RPENT_GPT_ENDPOINT",
        "AZURE_OPENAI_ENDPOINT",
        "RPENT_GPT_API_VERSION",
        "OPENAI_API_VERSION",
        "RPENT_GPT_MODEL",
        "RPENT_GPT_LOGID",
        "RPENT_GPT_MAX_TOKENS",
        "RPENT_GPT_TEMPERATURE",
        "QWEN_MODEL",
        "QWEN_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)


def test_factory_defaults_to_qwen_when_no_gpt_key(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("QWEN_API_KEY", "qwen-test")
    from XPolicyLab.policy.Pi_05_Agent_P1_RPent.planner_llm import (
        create_planner_llm,
    )

    client = create_planner_llm()
    assert isinstance(client, QwenClient)
    assert client.available()


def test_factory_selects_azure_when_backend_is_gpt(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("RPENT_LLM_BACKEND", "gpt")
    monkeypatch.setenv("RPENT_GPT_API_KEY", "gpt-test")
    from XPolicyLab.policy.Pi_05_Agent_P1_RPent.planner_llm import (
        AzureOpenAIPlannerClient,
        create_planner_llm,
    )

    client = create_planner_llm()
    assert isinstance(client, AzureOpenAIPlannerClient)
    assert client.available()


def test_factory_auto_selects_azure_when_only_gpt_key_is_set(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("RPENT_GPT_API_KEY", "gpt-test")
    from XPolicyLab.policy.Pi_05_Agent_P1_RPent.planner_llm import (
        AzureOpenAIPlannerClient,
        create_planner_llm,
    )

    client = create_planner_llm()
    assert isinstance(client, AzureOpenAIPlannerClient)


def _install_fake_openai(monkeypatch, captured):
    fake_openai = types.ModuleType("openai")

    class FakeResponse:
        def model_dump(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"label": "ok"}',
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "function": {
                                        "name": "ground",
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        }
                    }
                ]
            }

    class FakeCompletions:
        def create(self, **kwargs):
            captured["create"] = kwargs
            return FakeResponse()

    class FakeChat:
        completions = FakeCompletions()

    class FakeAzureOpenAI:
        def __init__(self, **kwargs):
            captured["init"] = kwargs
            self.chat = FakeChat()

    fake_openai.AzureOpenAI = FakeAzureOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    return fake_openai


def test_azure_client_uses_bytedance_azure_openai_contract(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("RPENT_GPT_API_KEY", "gpt-test-key")
    monkeypatch.setenv(
        "RPENT_GPT_ENDPOINT",
        "https://aidp.bytedance.net/api/modelhub/online/v2/crawl",
    )
    monkeypatch.setenv("RPENT_GPT_API_VERSION", "2024-03-01-preview")
    monkeypatch.setenv("RPENT_GPT_MODEL", "gpt-5.5-2026-04-24")
    monkeypatch.setenv("RPENT_GPT_LOGID", "log-123")
    monkeypatch.setenv("RPENT_GPT_MAX_TOKENS", "500")
    captured = {}
    _install_fake_openai(monkeypatch, captured)

    from XPolicyLab.policy.Pi_05_Agent_P1_RPent.planner_llm import (
        AzureOpenAIPlannerClient,
    )

    client = AzureOpenAIPlannerClient()
    result = client.chat(
        [{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "ground"}}],
        tool_choice="auto",
        extra={"enable_thinking": False},
    )
    text, tool_calls = client.message_text_and_tools(result)

    assert captured["init"]["api_key"] == "gpt-test-key"
    assert (
        captured["init"]["azure_endpoint"]
        == "https://aidp.bytedance.net/api/modelhub/online/v2/crawl"
    )
    assert captured["init"]["api_version"] == "2024-03-01-preview"
    create = captured["create"]
    assert create["model"] == "gpt-5.5-2026-04-24"
    assert create["stream"] is False
    assert create["max_tokens"] == 500
    assert "enable_thinking" not in create
    assert create["extra_headers"]["X-TT-LOGID"] == "log-123"
    assert create["tools"][0]["function"]["name"] == "ground"
    assert text == '{"label": "ok"}'
    assert tool_calls[0]["function"]["name"] == "ground"


def test_azure_client_omits_qwen_thinking_and_default_temperature(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("RPENT_GPT_API_KEY", "gpt-test-key")
    captured = {}
    _install_fake_openai(monkeypatch, captured)
    from XPolicyLab.policy.Pi_05_Agent_P1_RPent.planner_llm import (
        AzureOpenAIPlannerClient,
    )

    AzureOpenAIPlannerClient().chat([{"role": "user", "content": "hi"}])
    create = captured["create"]
    assert "enable_thinking" not in create
    assert "temperature" not in create


def test_deploy_uses_factory_client(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("RPENT_LLM_BACKEND", "gpt")
    monkeypatch.setenv("RPENT_GPT_API_KEY", "gpt-test")
    captured = {}

    class FakePlanner:
        def __init__(self, primitives, qwen):
            captured["client"] = qwen

        def run(self):
            captured["ran"] = True

    monkeypatch.setattr(
        "XPolicyLab.policy.Pi_05_Agent_P1_RPent.deploy.RpentPlanner",
        FakePlanner,
    )
    monkeypatch.setattr(
        "XPolicyLab.policy.Pi_05_Agent_P1_RPent.deploy.RpentPrimitives",
        lambda *args, **kwargs: types.SimpleNamespace(trace=[], finished=True),
    )
    monkeypatch.setattr(
        "XPolicyLab.policy.Pi_05_Agent_P1_RPent.deploy.enable_camera_calibration",
        lambda env: None,
    )

    from XPolicyLab.policy.Pi_05_Agent_P1_RPent.deploy import eval_one_episode
    from XPolicyLab.policy.Pi_05_Agent_P1_RPent.planner_llm import (
        AzureOpenAIPlannerClient,
    )

    class Env:
        def is_episode_end(self):
            return True

        def get_running_env_idx_list(self):
            return [0]

        success = [False]
        task_name = "general_pickup"
        seed = 0

    class Model:
        def call(self, *, func_name, **kwargs):
            return None

    eval_one_episode(Env(), Model())
    assert captured["ran"] is True
    assert isinstance(captured["client"], AzureOpenAIPlannerClient)
