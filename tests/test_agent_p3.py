import json

import numpy as np
import pytest
from XPolicyLab.policy.Agent_P3.deploy import (
    _mark_incomplete_episode_failed,
    eval_one_episode,
)
from XPolicyLab.policy.Agent_P3.policy import ActionDecodeError, LlmPolicy
from XPolicyLab.policy.Agent_P3.types import (
    EE_CHANNELS,
    JOINT_CHANNELS,
    ActionChunk,
    ActionSpace,
    Observation,
    Policy,
)
from XPolicyLab.policy.Agent_P3.wire import (
    ChatClient,
    ConfigError,
    MessagesClient,
    Provider,
    resolve_provider,
)


def _observation(step: int = 0) -> Observation:
    return Observation(
        images={"head": np.zeros((4, 4, 3), dtype=np.uint8)},
        state={
            "left_arm_joint_state": np.zeros(6, dtype=np.float32),
            "left_ee_joint_state": np.ones(1, dtype=np.float32),
            "right_arm_joint_state": np.zeros(6, dtype=np.float32),
            "right_ee_joint_state": np.ones(1, dtype=np.float32),
        },
        instruction="pick up the block",
        step=step,
    )


class _ScriptedClient:
    """Answers with a queued reply and records what it was sent."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def complete(self, messages, tools=(), *, system=None):
        self.requests.append({"messages": list(messages), "tools": list(tools), "system": system})
        return self.replies.pop(0)


def _reply(actions=None, *, text=None, name="act", usage=None):
    from XPolicyLab.policy.Agent_P3.wire import Reply

    arguments = None if actions is None else json.dumps({"actions": actions})
    return Reply(
        text=text,
        tool_name=None if actions is None and name == "act" else name,
        tool_arguments=arguments,
        usage=usage or {},
    )


# --- action space -----------------------------------------------------------


def test_the_tool_schema_width_matches_the_websocket_action_dict():
    space = ActionSpace(JOINT_CHANNELS)

    assert space.width == 14
    schema = space.tool_schema()["function"]["parameters"]["properties"]["actions"]
    assert schema["items"]["minItems"] == 14
    assert schema["items"]["maxItems"] == 14


def test_an_action_decodes_into_the_channels_the_environment_accepts():
    space = ActionSpace(JOINT_CHANNELS)

    action = space.decode(list(range(14)))

    assert set(action.data) == {name for name, _ in JOINT_CHANNELS}
    assert action.data["left_arm_joint_state"].tolist() == [0, 1, 2, 3, 4, 5]
    assert action.data["left_ee_joint_state"].tolist() == [6]
    assert action.data["right_ee_joint_state"].tolist() == [13]


def test_the_ee_action_space_is_wider_than_the_joint_one():
    assert ActionSpace(EE_CHANNELS).width == 16


def test_a_wrong_length_action_is_rejected_with_the_expected_order():
    space = ActionSpace(JOINT_CHANNELS)

    with pytest.raises(ValueError) as failure:
        space.decode([0.0, 1.0])

    message = str(failure.value)
    assert "expected 14" in message
    assert "left_arm_joint_state[0]" in message


def test_a_non_finite_action_is_rejected():
    with pytest.raises(ValueError, match="non-finite"):
        ActionSpace(JOINT_CHANNELS).decode([float("nan")] * 14)


def test_state_encodes_back_in_the_order_the_model_answers_in():
    space = ActionSpace(JOINT_CHANNELS)
    state = {
        "left_arm_joint_state": np.arange(6, dtype=np.float32),
        "left_ee_joint_state": np.array([0.5], dtype=np.float32),
        "right_arm_joint_state": np.zeros(6, dtype=np.float32),
        "right_ee_joint_state": np.array([1.0], dtype=np.float32),
    }

    assert space.encode(state) == [0, 1, 2, 3, 4, 5, 0.5, 0, 0, 0, 0, 0, 0, 1.0]


def test_an_empty_chunk_cannot_be_constructed():
    with pytest.raises(ValueError, match="at least one action"):
        ActionChunk(actions=[])


# --- provider resolution ----------------------------------------------------


def test_a_known_prefix_resolves_to_its_native_wire():
    provider = resolve_provider(
        "anthropic/claude-sonnet-4", env={"ANTHROPIC_API_KEY": "k"}
    )

    assert provider.base_url == "https://api.anthropic.com/v1"
    assert provider.model == "claude-sonnet-4"
    assert provider.wire == "messages"


def test_a_missing_direct_key_falls_back_to_the_router_with_the_full_name():
    provider = resolve_provider(
        "anthropic/claude-sonnet-4", env={"OPENROUTER_API_KEY": "k"}
    )

    assert provider.base_url == "https://openrouter.ai/api/v1"
    assert provider.model == "anthropic/claude-sonnet-4"
    assert provider.wire == "chat"


def test_an_explicit_base_url_wins_over_the_prefix_table():
    provider = resolve_provider(
        "anthropic/claude-sonnet-4",
        env={
            "ANTHROPIC_API_KEY": "k",
            "P3_BASE_URL": "http://localhost:8000/v1",
        },
    )

    assert provider.base_url == "http://localhost:8000/v1"
    assert provider.wire == "chat"


def test_no_key_anywhere_is_a_config_error_that_names_the_fix():
    with pytest.raises(ConfigError) as failure:
        resolve_provider("anthropic/claude-sonnet-4", env={})

    message = str(failure.value)
    assert "ANTHROPIC_API_KEY" in message
    assert "OPENROUTER_API_KEY" in message


def test_no_model_at_all_is_a_config_error():
    with pytest.raises(ConfigError, match="P3_MODEL"):
        resolve_provider(env={})


# --- wires ------------------------------------------------------------------


def _capture_transport(status=200, payload=None):
    seen = {}

    def transport(url, headers, body, timeout):
        seen["url"] = url
        seen["headers"] = dict(headers)
        seen["body"] = json.loads(body.decode("utf-8"))
        seen["timeout"] = timeout
        return status, json.dumps(payload or {}).encode("utf-8")

    return transport, seen


def test_the_chat_wire_requires_a_tool_call_and_reads_usage():
    payload = {
        "choices": [
            {
                "message": {
                    "content": "moving",
                    "tool_calls": [
                        {"function": {"name": "act", "arguments": '{"actions": []}'}}
                    ],
                }
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "prompt_tokens_details": {"cached_tokens": 80},
        },
    }
    transport, seen = _capture_transport(payload=payload)
    client = ChatClient(
        Provider("m", "http://x/v1", "key", "chat"), transport=transport
    )

    reply = client.complete(
        [{"role": "user", "content": "hi"}],
        [ActionSpace().tool_schema()],
        system="sys",
    )

    assert seen["url"] == "http://x/v1/chat/completions"
    assert seen["headers"]["Authorization"] == "Bearer key"
    assert seen["body"]["messages"][0] == {"role": "system", "content": "sys"}
    assert seen["body"]["tool_choice"] == "required"
    assert reply.tool_name == "act"
    assert reply.usage["cache_read_input_tokens"] == 80


def test_the_messages_wire_caches_the_system_block_and_translates_images():
    payload = {
        "content": [
            {"type": "text", "text": "thinking"},
            {"type": "tool_use", "name": "act", "input": {"actions": [[0.0]]}},
        ],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 4,
            "cache_read_input_tokens": 7,
        },
    }
    transport, seen = _capture_transport(payload=payload)
    client = MessagesClient(
        Provider("m", "http://x/v1", "key", "messages"), transport=transport
    )

    reply = client.complete(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AAA"},
                    },
                ],
            }
        ],
        [ActionSpace().tool_schema()],
        system="sys",
    )

    assert seen["url"] == "http://x/v1/messages"
    assert seen["headers"]["x-api-key"] == "key"
    assert seen["body"]["system"][0]["cache_control"] == {"type": "ephemeral"}
    image = seen["body"]["messages"][0]["content"][1]
    assert image["source"] == {
        "type": "base64",
        "media_type": "image/png",
        "data": "AAA",
    }
    assert seen["body"]["tools"][0]["input_schema"]["type"] == "object"
    assert reply.tool_name == "act"
    assert reply.usage["cache_read_input_tokens"] == 7


def test_a_rate_limit_is_retried_with_exponential_backoff():
    attempts = []
    slept = []

    def transport(url, headers, body, timeout):
        attempts.append(1)
        if len(attempts) < 3:
            return 429, b"slow down"
        return 200, json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()

    client = ChatClient(
        Provider("m", "http://x/v1", "k", "chat"),
        transport=transport,
        sleep=slept.append,
    )

    reply = client.complete([{"role": "user", "content": "hi"}])

    assert len(attempts) == 3
    assert slept == [1.0, 2.0]
    assert reply.text == "ok"


def test_a_bad_request_fails_immediately_rather_than_retrying():
    attempts = []

    def transport(url, headers, body, timeout):
        attempts.append(1)
        return 400, b"bad model"

    client = ChatClient(
        Provider("m", "http://x/v1", "k", "chat"), transport=transport, sleep=lambda _: None
    )

    with pytest.raises(RuntimeError, match="rejected"):
        client.complete([{"role": "user", "content": "hi"}])
    assert len(attempts) == 1


# --- policy -----------------------------------------------------------------


def test_the_llm_policy_satisfies_the_same_contract_a_vla_fills():
    assert isinstance(LlmPolicy(client=_ScriptedClient([])), Policy)


def test_a_tool_call_becomes_the_action_chunk_the_harness_runs():
    client = _ScriptedClient(
        [_reply([[0.1] * 14], text="approaching", usage={"input_tokens": 5})]
    )
    policy = LlmPolicy(client=client, action_space=ActionSpace(max_chunk=4))

    chunk = policy.act(_observation())

    assert len(chunk) == 1
    assert chunk.actions[0].data["left_arm_joint_state"].tolist() == pytest.approx(
        [0.1] * 6
    )
    assert chunk.reasoning == "approaching"
    assert chunk.usage["input_tokens"] == 5
    assert policy.calls == 1


def test_the_request_shows_the_state_in_the_order_the_answer_uses():
    client = _ScriptedClient([_reply([[0.0] * 14])])
    policy = LlmPolicy(client=client)

    policy.act(_observation())

    text = client.requests[0]["messages"][0]["content"][0]["text"]
    assert "left_arm_joint_state[0]" in text
    assert "right_ee_joint_state" in text
    assert "pick up the block" in text
    labels = ActionSpace().flat_labels()
    assert text.index("left_arm_joint_state[0]") < text.index("right_ee_joint_state")
    assert len(labels) == 14


def test_every_request_carries_the_current_camera_images():
    client = _ScriptedClient([_reply([[0.0] * 14])])
    policy = LlmPolicy(client=client)

    policy.act(_observation())

    parts = client.requests[0]["messages"][0]["content"]
    images = [part for part in parts if part.get("type") == "image_url"]
    assert len(images) == 1
    assert images[0]["image_url"]["url"].startswith("data:image/png;base64,")


def test_a_malformed_action_is_sent_back_for_repair_with_the_reason():
    client = _ScriptedClient([_reply([[0.0] * 3]), _reply([[0.0] * 14])])
    policy = LlmPolicy(client=client)

    chunk = policy.act(_observation())

    assert len(chunk) == 1
    assert policy.calls == 2
    repair = client.requests[1]["messages"][-1]["content"]
    assert "not executable" in repair
    assert "expected 14" in repair


def test_repairs_are_bounded_and_the_failure_names_the_last_reason():
    client = _ScriptedClient([_reply([[0.0] * 3]) for _ in range(3)])
    policy = LlmPolicy(client=client, max_repairs=2)

    with pytest.raises(ActionDecodeError, match="3 attempts"):
        policy.act(_observation())
    assert policy.calls == 3


def test_prose_without_a_tool_call_is_rejected():
    client = _ScriptedClient([_reply(None, text="I will move the arm")] * 2)
    policy = LlmPolicy(client=client, max_repairs=1)

    with pytest.raises(ActionDecodeError, match="No `act` call"):
        policy.act(_observation())


def test_a_chunk_longer_than_the_condition_allows_is_rejected():
    client = _ScriptedClient([_reply([[0.0] * 14] * 5), _reply([[0.0] * 14])])
    policy = LlmPolicy(client=client, action_space=ActionSpace(max_chunk=2))

    policy.act(_observation())

    assert "exceeds the limit of 2" in client.requests[1]["messages"][-1]["content"]


def test_history_keeps_intent_but_never_replays_stale_images():
    client = _ScriptedClient([_reply([[0.0] * 14], text=f"turn {i}") for i in range(3)])
    policy = LlmPolicy(client=client, history_turns=1)

    for step in range(3):
        policy.act(_observation(step))

    carried = client.requests[2]["messages"][:-1]
    assert carried, "the window should carry the previous turn"
    for turn in carried:
        assert isinstance(turn["content"], str)
    assert len(carried) == 2


def test_reset_clears_the_window_between_episodes():
    client = _ScriptedClient([_reply([[0.0] * 14]) for _ in range(2)])
    policy = LlmPolicy(client=client, history_turns=2)

    policy.act(_observation(0))
    policy.reset()
    policy.act(_observation(1))

    assert len(client.requests[1]["messages"]) == 1


def test_usage_accumulates_across_the_episode():
    client = _ScriptedClient(
        [
            _reply([[0.0] * 14], usage={"input_tokens": 10, "output_tokens": 2}),
            _reply([[0.0] * 14], usage={"input_tokens": 7, "output_tokens": 3}),
        ]
    )
    policy = LlmPolicy(client=client)

    policy.act(_observation(0))
    policy.act(_observation(1))

    assert policy.usage_totals == {"input_tokens": 17, "output_tokens": 5}


# --- harness integration ----------------------------------------------------


class _FakeEnv:
    """Enough of the RoboDojo env for the P3 loop."""

    step_lim = 20
    task_name = "unit"
    seed = 0
    instruction = "pick up the block"

    def __init__(self, ends_after=2):
        self.take_action_cnt = 0
        self.ends_after = ends_after
        self.actions = []
        self.success = [True]

    def get_obs(self):
        return {
            "observation": {"head": {"rgb": np.zeros((4, 4, 3), dtype=np.uint8)}},
            "state": {
                "left_arm_joint_state": np.zeros(6, dtype=np.float32),
                "left_ee_joint_state": np.ones(1, dtype=np.float32),
                "right_arm_joint_state": np.zeros(6, dtype=np.float32),
                "right_ee_joint_state": np.ones(1, dtype=np.float32),
            },
        }

    def take_action(self, action):
        self.actions.append(action)
        self.take_action_cnt += 1

    def is_episode_end(self):
        return self.take_action_cnt >= self.ends_after

    def get_running_env_idx_list(self):
        return [0]


def test_the_loop_sends_the_models_numbers_to_the_environment_unchanged(monkeypatch):
    env = _FakeEnv(ends_after=2)
    client = _ScriptedClient([_reply([[0.25] * 14]) for _ in range(4)])
    monkeypatch.setattr(
        "XPolicyLab.policy.Agent_P3.deploy.LlmPolicy",
        lambda: LlmPolicy(client=client),
    )

    eval_one_episode(env, model_client=None)

    assert len(env.actions) == 2
    assert env.actions[0]["left_arm_joint_state"].tolist() == pytest.approx([0.25] * 6)


def test_the_observation_carries_the_remaining_step_budget(monkeypatch):
    env = _FakeEnv(ends_after=1)
    client = _ScriptedClient([_reply([[0.0] * 14]) for _ in range(2)])
    monkeypatch.setattr(
        "XPolicyLab.policy.Agent_P3.deploy.LlmPolicy",
        lambda: LlmPolicy(client=client),
    )

    eval_one_episode(env, model_client=None)

    assert "20 steps remain" in client.requests[0]["messages"][0]["content"][0]["text"]


def test_stopping_before_the_official_end_is_recorded_as_a_failure():
    env = _FakeEnv(ends_after=99)

    _mark_incomplete_episode_failed(env)

    assert env.success == [False]


def test_p3_has_no_vla_to_call():
    from XPolicyLab.policy.Agent_P3.model import Model

    with pytest.raises(RuntimeError, match="no VLA"):
        Model().get_action()


def test_images_still_encode_where_pillow_is_missing():
    """The eval environments do not all ship Pillow, so the fallback must work."""
    from XPolicyLab.policy.Agent_P3.policy import _encode_png

    pixels = np.arange(8 * 6 * 3, dtype=np.uint8).reshape(8, 6, 3)

    raw = _encode_png(pixels)

    assert raw.startswith(b"\x89PNG\r\n\x1a\n")
    pillow = pytest.importorskip("PIL.Image")
    import io

    decoded = np.asarray(pillow.open(io.BytesIO(raw)).convert("RGB"))
    assert np.array_equal(decoded, pixels)
