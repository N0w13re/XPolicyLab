import json
from dataclasses import replace
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from XPolicyLab.policy.Agent_P3.deploy import (
    _action_spec,
    _debug_action_spec,
    _mark_incomplete_episode_failed,
    _observation as robodojo_observation,
    eval_one_episode,
    eval_one_episode_batch,
)
from XPolicyLab.policy.Agent_P3.policy import (
    LlmPolicy,
    RoboDojoActionSpec,
    _agent_kwargs,
)
from XPolicyLab.policy.Agent_P3.types import (
    EE_CHANNELS,
    JOINT_CHANNELS,
    ActionChunk,
    ActionSpace,
    Observation,
    Policy,
)
from XPolicyLab.policy.Agent_P3.wire import AzureChatTransport


def _observation(step: int = 0, *, extra: dict | None = None) -> Observation:
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
        extra=dict(extra or {}),
    )


def _spec(control_hz=25.0):
    labels = tuple(
        [f"left_joint{i}" for i in range(1, 7)]
        + ["left_gripper"]
        + [f"right_joint{i}" for i in range(1, 7)]
        + ["right_gripper"]
    )
    return RoboDojoActionSpec(
        labels=labels,
        low=np.array([-10.0] * 6 + [0.0] + [-10.0] * 6 + [0.0]),
        high=np.array([10.0] * 5 + [3.14, 1.0] + [10.0] * 5 + [3.14, 1.0]),
        control_hz=control_hz,
        docs="dual ARX X5",
    )


def _move(targets=None, *, note="at rest; move one joint", text=None):
    return {
        "choices": [
            {
                "message": {
                    "content": text,
                    "tool_calls": [
                        {
                            "id": "call-move",
                            "type": "function",
                            "function": {
                                "name": "move_joints",
                                "arguments": json.dumps(
                                    {
                                        "targets": targets or {"left_joint1": 0.0},
                                        "note": note,
                                    }
                                ),
                            },
                        }
                    ],
                }
            }
        ]
    }


def _stop(name, *, summary="", reason="", hindsight="none"):
    arguments = {"hindsight": hindsight}
    if name == "done":
        arguments["summary"] = summary or "finished"
    else:
        arguments["reason"] = reason or "stuck"
    return {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-stop",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ],
                }
            }
        ]
    }


def _policy(payloads, *, env=None, seen=None, pre_check=None):
    queue = list(payloads)
    requests = [] if seen is None else seen

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=queue.pop(0))

    config = {
        "P3_MODEL": "test-model",
        "P3_BASE_URL": "https://example.test/v1",
        "P3_API_KEY_ENV": "TEST_KEY",
        "TEST_KEY": "secret",
        "P3_WIRE_CAPTURE": "false",
    }
    if env:
        config.update(env)
    return LlmPolicy(
        action_spec=_spec(),
        env=config,
        transport=httpx.MockTransport(handler),
        pre_check=pre_check,
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


# --- inspect-robots-agent protocol ------------------------------------------


def test_the_llm_policy_satisfies_the_same_contract_a_vla_fills():
    policy = _policy([])
    assert isinstance(policy, Policy)
    assert policy.inner.__class__.__module__ == "inspect_robots_agent.policy"


def test_the_default_tools_are_move_joints_done_and_give_up():
    seen = []
    policy = _policy([_move()], seen=seen)
    policy.act(_observation())
    names = [tool["function"]["name"] for tool in seen[0]["tools"]]
    assert names == ["move_joints", "done", "give_up"]
    assert "note" in seen[0]["tools"][0]["function"]["parameters"]["required"]
    system = seen[0]["messages"][0]["content"]
    assert "controlling a real robot embodiment named 'robodojo-arx-x5'" in system
    assert "Embodiment notes:" in system
    assert "budget of 100 LLM calls" in system


def test_prior_learnings_are_appended_to_the_system_prompt(tmp_path):
    import hashlib

    notes = tmp_path / "hindsight.md"
    notes.write_text("the white sphere is the ball\n", encoding="utf-8")
    seen = []
    policy = _policy([_move()], env={"P3_PRIOR_LEARNINGS": str(notes)}, seen=seen)

    policy.act(_observation())

    system = seen[0]["messages"][0]["content"]
    assert "Notes from a previous attempt" in system
    assert "white sphere is the ball" in system
    upstream = policy.audit_config()["upstream_policy_config"]
    assert upstream["prior_learnings_sha256"] == hashlib.sha256(
        notes.read_bytes()
    ).hexdigest()


def test_named_targets_use_upstream_speed_limited_interpolation():
    policy = _policy([_move({"left_joint1": 0.2}, text="inching")])

    chunk = policy.act(_observation())

    # Range is 20 rad, max_speed_frac=.1 at 25 Hz => .08 rad/control step.
    assert len(chunk) == 3
    last = chunk.actions[-1].data["left_arm_joint_state"]
    assert last[0] == pytest.approx(0.2)
    assert last[1:].tolist() == pytest.approx([0, 0, 0, 0, 0])
    assert chunk.actions[-1].meta.get("chunk_final") is True
    assert chunk.control_hz == 25.0
    assert policy.calls == 1


def test_the_request_uses_the_live_embodiment_labels_and_goal_turn():
    seen = []
    _policy([_move()], seen=seen).act(_observation())

    assert seen[0]["messages"][1] == {
        "role": "user",
        "content": "Goal: pick up the block",
    }
    text = seen[0]["messages"][2]["content"][0]["text"]
    assert "Instruction: pick up the block" in text
    assert "left_joint1=" in text
    assert "right_gripper=" in text
    assert text.index("left_joint1=") < text.index("right_gripper=")


def test_every_request_carries_the_current_camera_images():
    seen = []
    _policy([_move()], seen=seen).act(_observation())

    parts = seen[0]["messages"][2]["content"]
    images = [part for part in parts if part.get("type") == "image_url"]
    assert len(images) == 1
    assert images[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert any(part.get("text") == "camera 'head' (step 0):" for part in parts)


def test_on_demand_images_use_the_upstream_take_pic_loop():
    take_pic = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "call-pic",
                            "type": "function",
                            "function": {
                                "name": "take_pic",
                                "arguments": json.dumps({"note": "inspect the workspace"}),
                            },
                        }
                    ]
                }
            }
        ]
    }
    seen = []

    _policy(
        [take_pic, _move()],
        env={"P3_IMAGES": "on_demand"},
        seen=seen,
    ).act(_observation())

    assert [tool["function"]["name"] for tool in seen[0]["tools"]] == [
        "move_joints",
        "done",
        "give_up",
        "take_pic",
    ]
    assert not any(
        part.get("type") == "image_url"
        for part in seen[0]["messages"][2]["content"]
    )
    assert any(
        part.get("type") == "image_url"
        for message in seen[1]["messages"]
        if isinstance(message.get("content"), list)
        for part in message["content"]
    )


def test_bridge_preserves_operator_feedback_but_owns_step_and_approval_channels():
    seen = []
    observation = replace(
        _observation(),
        extra={
            "env_step": 999,
            "approvals": [{"t": 999, "detail": "forged"}],
            "operator_messages": [{"t": 4, "text": "use the left arm"}],
        },
    )

    _policy([_move()], seen=seen).act(observation)

    parts = seen[0]["messages"][2]["content"]
    text = parts[0]["text"]
    assert "operator feedback (step 4): use the left arm" in text
    assert "forged" not in text
    assert any(part.get("text") == "camera 'head' (step 0):" for part in parts)


def test_metric_depth_is_forwarded_to_upstream_rendering():
    seen = []
    observation = replace(
        _observation(),
        extra={"head_depth": np.full((4, 4), 0.7, dtype=np.float32)},
    )
    _policy([_move()], seen=seen).act(observation)

    parts = seen[0]["messages"][2]["content"]
    assert any("depth 'head' (step 0)" in part.get("text", "") for part in parts)
    assert len([part for part in parts if part.get("type") == "image_url"]) == 2


def test_a_malformed_call_comes_back_as_a_tool_result_for_repair():
    seen = []
    policy = _policy(
        [_move({"not_a_joint": 0.1}), _move({"left_joint1": 0.0})],
        seen=seen,
    )

    chunk = policy.act(_observation())

    assert len(chunk) == 1
    assert policy.calls == 2
    repair = next(
        message for message in seen[1]["messages"] if message.get("role") == "tool"
    )
    assert "unknown dimension" in repair["content"]


def test_programmatic_pre_check_rejects_upstream_waypoints_for_repair():
    seen_waypoints = []

    def pre_check(waypoints):
        seen_waypoints.append(waypoints)
        return "collision predicted" if len(seen_waypoints) == 1 else None

    seen = []
    policy = _policy(
        [_move({"left_joint1": 0.1}), _move({"left_joint1": 0.05})],
        seen=seen,
        pre_check=pre_check,
    )

    policy.act(_observation())

    assert len(seen_waypoints) == 2
    assert seen_waypoints[0].flags.writeable is False
    repair = next(
        message for message in seen[1]["messages"] if message.get("role") == "tool"
    )
    assert "pre-check rejected this motion: collision predicted" == repair["content"]


def test_chat_transport_retries_transient_failures_with_upstream_backoff(monkeypatch):
    statuses = [429, 500, 200]
    sleeps = []

    def handler(request):
        status = statuses.pop(0)
        if status == 200:
            return httpx.Response(status, json=_move())
        return httpx.Response(status, text="transient")

    monkeypatch.setattr("inspect_robots_agent._llm.time.sleep", sleeps.append)
    policy = LlmPolicy(
        action_spec=_spec(),
        env={
            "P3_MODEL": "test-model",
            "P3_BASE_URL": "https://example.test/v1",
            "P3_API_KEY_ENV": "TEST_KEY",
            "TEST_KEY": "secret",
            "P3_WIRE_CAPTURE": "false",
        },
        transport=httpx.MockTransport(handler),
    )

    chunk = policy.act(_observation())

    assert len(chunk) == 1
    assert policy.calls == 1
    assert statuses == []
    assert sleeps == [1.0, 2.0]


def test_three_consecutive_tool_failures_end_the_turn():
    policy = _policy([_move({"not_a_joint": 0.1}) for _ in range(3)])

    with pytest.raises(RuntimeError, match="kept failing"):
        policy.act(_observation())
    assert policy.calls == 3


def test_prose_without_a_tool_call_is_nudged_then_rejected():
    prose = {"choices": [{"message": {"content": "I will move the arm"}}]}
    seen = []
    policy = _policy([prose, prose, prose], seen=seen)

    with pytest.raises(RuntimeError, match="no tool call"):
        policy.act(_observation())
    assert "exactly one tool call" in seen[1]["messages"][-1]["content"]


def test_a_move_past_the_playout_cap_is_a_structured_error():
    seen = []
    policy = _policy(
        [_move({"left_joint1": 3.0}), _move({"left_joint1": 0.0})],
        env={"P3_MAX_SPEED_FRAC": "0.01"},
        seen=seen,
    )

    policy.act(_observation())

    tool_messages = [
        message["content"]
        for message in seen[1]["messages"]
        if message.get("role") == "tool"
    ]
    assert any("playout cap" in (text or "") for text in tool_messages)


def test_image_horizon_elides_older_camera_frames():
    seen = []
    policy = _policy(
        [_move() for _ in range(3)],
        env={"P3_IMAGE_HORIZON": "1"},
        seen=seen,
    )

    for step in range(3):
        policy.act(_observation(step))

    outgoing = seen[2]["messages"]
    first_user = outgoing[2]
    assert any(
        "[1 camera frame(s) elided]" in (part.get("text") or "")
        for part in first_user["content"]
    )
    last_user = outgoing[-1]
    assert any(part.get("type") == "image_url" for part in last_user["content"])


def test_reset_clears_the_window_between_episodes():
    seen = []
    policy = _policy([_move() for _ in range(2)], seen=seen)

    policy.act(_observation(0))
    policy.reset()
    policy.act(_observation(1))

    assert len(seen[1]["messages"]) == 3


def test_llm_call_budget_forces_give_up():
    policy = _policy([_move()], env={"P3_MAX_LLM_CALLS": "1"})

    policy.act(_observation(0))
    chunk = policy.act(_observation(1))

    assert chunk.actions[0].meta.get("stop_reason") == "give_up"
    assert policy.calls == 1


def test_done_carries_hindsight_and_stops_the_trial():
    policy = _policy(
        [_stop("done", summary="grasped", hindsight="the ball is the white sphere")]
    )

    chunk = policy.act(_observation())

    assert chunk.actions[0].meta["request_stop"] is True
    assert chunk.actions[0].meta["stop_reason"] == "done"
    assert policy.hindsight == "the ball is the white sphere"


def test_audit_config_pins_the_exact_upstream_strategy_versions():
    config = _policy([]).audit_config()
    assert config["adapter"] == "inspect-robots-agent"
    assert config["inspect_robots_version"] == "0.58.0"
    assert config["inspect_robots_agent_version"] == "0.26.0"
    assert config["embodiment"]["control_hz"] == 25.0
    assert config["upstream_policy_config"]["images"] == "always"
    assert config["upstream_policy_config"]["depth"] == "render"


def test_all_provider_controls_are_forwarded_without_local_defaults():
    pre_check = lambda waypoints: None
    kwargs, requested_wire = _agent_kwargs(
        {
            "P3_MODEL": "anthropic/test-model",
            "P3_WIRE": "messages",
            "P3_TEMPERATURE": "0.2",
            "P3_EFFORT": "high",
            "P3_MAX_OUTPUT_TOKENS": "4096",
            "P3_SPEED": "fast",
            "P3_IMAGE_HORIZON": "none",
        },
        pre_check=pre_check,
    )

    assert requested_wire == "messages"
    assert kwargs["temperature"] == 0.2
    assert kwargs["effort"] == "high"
    assert kwargs["max_output_tokens"] == 4096
    assert kwargs["speed"] == "fast"
    assert kwargs["image_horizon"] is None
    assert kwargs["pre_check"] is pre_check


def test_unset_variables_leave_every_strategy_default_upstream():
    kwargs, _ = _agent_kwargs({"P3_MODEL": "test-model"})

    assert set(kwargs) == {"model", "env", "pre_check"}


@pytest.mark.parametrize(
    "wire", ["chat", "messages", "responses", "gemini-live", "interactions"]
)
def test_every_native_wire_is_forwarded_to_upstream_untouched(wire):
    kwargs, requested_wire = _agent_kwargs({"P3_MODEL": "test-model", "P3_WIRE": wire})

    assert requested_wire == wire
    assert kwargs["wire"] == wire


def test_an_unset_wire_leaves_the_choice_to_upstream_provider_defaults():
    kwargs, requested_wire = _agent_kwargs({"P3_MODEL": "anthropic/test-model"})

    assert requested_wire == "auto"
    assert "wire" not in kwargs


def test_azure_chat_is_the_chat_wire_with_the_deployment_as_the_model():
    kwargs, requested_wire = _agent_kwargs(
        {
            "P3_MODEL": "azure/gpt-5",
            "P3_WIRE": "azure-chat",
            "P3_BASE_URL": "https://azure.test/gateway",
            "P3_API_VERSION": "2026-01-01",
        }
    )

    assert requested_wire == "azure-chat"
    assert kwargs["wire"] == "chat"
    assert kwargs["model"] == "gpt-5"
    assert isinstance(kwargs["transport"], AzureChatTransport)


def _depth_frame():
    depth = np.full((4, 4), 0.7, dtype=np.float32)
    depth[0, 0] = 1.4
    return depth


def _parts(payload):
    return [
        part
        for message in payload["messages"]
        if isinstance(message["content"], list)
        for part in message["content"]
    ]


def test_metric_depth_is_rendered_beside_the_colour_frame():
    seen = []
    policy = _policy([_move()], env={"P3_DEPTH": "render"}, seen=seen)

    policy.act(_observation(extra={"head_depth": _depth_frame()}))

    parts = _parts(seen[0])
    labels = [part.get("text", "") for part in parts if part["type"] == "text"]
    assert any("depth 'head'" in text and " m " in text for text in labels)
    assert sum(part["type"] == "image_url" for part in parts) == 2


def test_depth_off_never_resolves_the_depth_entries():
    seen = []

    def boom():
        raise AssertionError("depth must not be resolved when depth=off")

    policy = _policy([_move()], env={"P3_DEPTH": "off"}, seen=seen)

    policy.act(_observation(extra={"head_depth": boom}))

    parts = _parts(seen[0])
    assert not any("depth" in part.get("text", "") for part in parts)
    assert sum(part["type"] == "image_url" for part in parts) == 1


def test_quaternion_ee_control_is_refused_instead_of_silently_downgraded(monkeypatch):
    monkeypatch.setenv("P3_ACTION_TYPE", "ee")

    with pytest.raises(ValueError, match="quaternion"):
        _action_spec(SimpleNamespace())


def test_azure_transport_changes_only_endpoint_auth_and_deployment_model():
    seen = {}

    def handler(request):
        seen["request"] = request
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    transport = AzureChatTransport(
        base_url="https://azure.test/gateway",
        deployment="gpt-5",
        api_version="2026-01-01",
        inner=httpx.MockTransport(handler),
    )
    client = httpx.Client(
        base_url="https://ignored.test/v1",
        headers={"Authorization": "Bearer secret"},
        transport=transport,
    )

    client.post(
        "/chat/completions",
        json={"model": "ignored", "messages": [{"role": "user", "content": "hi"}]},
    )

    request = seen["request"]
    assert str(request.url) == (
        "https://azure.test/gateway/openai/deployments/gpt-5/chat/completions"
        "?api-version=2026-01-01"
    )
    assert request.headers["api-key"] == "secret"
    assert "authorization" not in request.headers
    assert int(request.headers["content-length"]) == len(request.content)
    assert json.loads(request.content)["model"] == "gpt-5"


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
        self.env_seeds = [7]
        self.obs_manager = SimpleNamespace(collect_depth=False)

    def get_obs(self):
        return {
            "vision": {
                "head": {
                    "color": np.zeros((4, 4, 3), dtype=np.uint8),
                    "depth": np.full((4, 4), 0.7, dtype=np.float32),
                }
            },
            "state": {
                "left_arm_joint_state": np.zeros(6, dtype=np.float32),
                "left_ee_joint_state": np.ones(1, dtype=np.float32),
                "right_arm_joint_state": np.zeros(6, dtype=np.float32),
                "right_ee_joint_state": np.ones(1, dtype=np.float32),
            },
            "instruction": self.instruction,
            "additional_info": {"frequency": 25},
            "data_format_version": "v1.0",
            "env_idx": 0,
        }

    def take_action(self, action):
        self.actions.append(action)
        if isinstance(self.take_action_cnt, list):
            self.take_action_cnt[0] += 1
        else:
            self.take_action_cnt += 1

    def is_episode_end(self):
        used = (
            self.take_action_cnt[0]
            if isinstance(self.take_action_cnt, list)
            else self.take_action_cnt
        )
        return used >= self.ends_after

    def get_running_env_idx_list(self):
        return [0]


def test_observation_maps_the_robodojo_runtime_contract_without_mutating_it():
    env = _FakeEnv()
    raw = env.get_obs()

    observation = robodojo_observation(env, policy_step=3)

    assert observation.instruction == "pick up the block"
    np.testing.assert_array_equal(
        observation.images["head"],
        raw["vision"]["head"]["color"],
    )
    assert set(raw) == {
        "vision",
        "state",
        "instruction",
        "additional_info",
        "data_format_version",
        "env_idx",
    }
    np.testing.assert_array_equal(
        observation.extra["head_depth"],
        raw["vision"]["head"]["depth"],
    )
    assert observation.extra["additional_info"] == {"frequency": 25}
    assert observation.extra["data_format_version"] == "v1.0"
    assert observation.extra["env_idx"] == 0
    assert observation.extra["layout_id"] == 7
    assert observation.step == 3


def test_the_loop_plays_the_interpolated_chunk(monkeypatch):
    env = _FakeEnv(ends_after=2)
    policy = _policy([_move() for _ in range(8)])
    monkeypatch.setattr(
        "XPolicyLab.policy.Agent_P3.deploy._action_spec",
        lambda task_env: _spec(),
    )
    monkeypatch.setattr(
        "XPolicyLab.policy.Agent_P3.deploy.LlmPolicy",
        lambda **kwargs: policy,
    )

    eval_one_episode(env, model_client=None)

    assert len(env.actions) == 2
    assert env.actions[0]["left_arm_joint_state"][0] == pytest.approx(0.0)


def test_give_up_marks_the_episode_failed_before_the_audit_is_written(
    monkeypatch, tmp_path
):
    env = _FakeEnv(ends_after=99)
    policy = _policy([_stop("give_up", reason="blocked")])
    monkeypatch.setenv("P3_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("ROBODOJO_RUN_ID", "unit-give-up")
    monkeypatch.setattr(
        "XPolicyLab.policy.Agent_P3.deploy._action_spec",
        lambda task_env: _spec(),
    )
    monkeypatch.setattr(
        "XPolicyLab.policy.Agent_P3.deploy.LlmPolicy",
        lambda **kwargs: policy,
    )

    eval_one_episode(env, model_client=None)

    assert env.success == [False]
    audit = json.loads((tmp_path / "p3_transcript.json").read_text())
    assert audit["official_success"] == [False]
    assert audit["layout_id"] == 7
    assert audit["policy_config"]["scene"]["init_seed"] == 7
    assert audit["inspect_metadata"]["trial_record"] == {
        "terminated": False,
        "truncated": True,
        "termination_reason": "give_up",
        "seed": 7,
    }


def test_policy_stop_cannot_override_a_successful_robodojo_final_check(monkeypatch):
    class RewardCompleteEnv(_FakeEnv):
        def is_episode_end(self):
            if not self.success[0]:
                self.success[0] = True
                return True
            return False

    env = RewardCompleteEnv(ends_after=99)
    policy = _policy([_stop("done", summary="complete")])
    monkeypatch.setattr(
        "XPolicyLab.policy.Agent_P3.deploy._action_spec",
        lambda task_env: _spec(),
    )
    monkeypatch.setattr(
        "XPolicyLab.policy.Agent_P3.deploy.LlmPolicy",
        lambda **kwargs: policy,
    )

    eval_one_episode(env, model_client=None)

    assert env.success == [True]


def _run_episode(env, policy, monkeypatch):
    monkeypatch.setattr(
        "XPolicyLab.policy.Agent_P3.deploy._action_spec",
        lambda task_env: _spec(),
    )
    monkeypatch.setattr(
        "XPolicyLab.policy.Agent_P3.deploy.LlmPolicy",
        lambda **kwargs: policy,
    )
    eval_one_episode(env, model_client=None)


def test_the_audit_records_which_cameras_delivered_metric_depth(
    monkeypatch, tmp_path
):
    env = _FakeEnv(ends_after=2)
    monkeypatch.setenv("P3_TRACE_DIR", str(tmp_path))

    _run_episode(env, _policy([_move() for _ in range(4)]), monkeypatch)

    assert env.obs_manager.collect_depth is True
    audit = json.loads((tmp_path / "p3_transcript.json").read_text())
    assert audit["depth_cameras"] == ["head"]


def test_a_depth_condition_without_rendered_depth_is_reported_not_assumed(
    monkeypatch, tmp_path, capsys
):
    class RgbOnlyEnv(_FakeEnv):
        def get_obs(self):
            raw = super().get_obs()
            raw["vision"]["head"].pop("depth")
            return raw

    env = RgbOnlyEnv(ends_after=2)
    monkeypatch.setenv("P3_TRACE_DIR", str(tmp_path))

    _run_episode(env, _policy([_move() for _ in range(4)]), monkeypatch)

    assert "the model sees RGB only" in capsys.readouterr().out
    audit = json.loads((tmp_path / "p3_transcript.json").read_text())
    assert audit["depth_cameras"] == []
    assert audit["policy_config"]["upstream_policy_config"]["depth"] == "render"


def test_depth_off_leaves_robodojo_depth_collection_alone(monkeypatch, tmp_path):
    env = _FakeEnv(ends_after=2)
    monkeypatch.setenv("P3_DEPTH", "off")
    monkeypatch.setenv("P3_TRACE_DIR", str(tmp_path))

    _run_episode(
        env,
        _policy([_move() for _ in range(4)], env={"P3_DEPTH": "off"}),
        monkeypatch,
    )

    assert env.obs_manager.collect_depth is False


def test_stopping_before_the_official_end_is_recorded_as_a_failure():
    env = _FakeEnv(ends_after=99)

    _mark_incomplete_episode_failed(env)

    assert env.success == [False]


def test_p3_has_no_vla_to_call():
    from XPolicyLab.policy.Agent_P3.model import Model

    with pytest.raises(RuntimeError, match="no VLA"):
        Model({}).get_action()


def test_batched_eval_refuses_more_than_one_environment():
    env = _FakeEnv()
    env.num_envs = 2

    with pytest.raises(RuntimeError, match="one conversation per environment"):
        eval_one_episode_batch(env, model_client=None)


class _Tensor:
    def __init__(self, array):
        self.array = np.asarray(array)

    def __getitem__(self, item):
        normalized = tuple(
            np.asarray(index) if isinstance(index, list) else index
            for index in (item if isinstance(item, tuple) else (item,))
        )
        return _Tensor(self.array[normalized])

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


def test_action_spec_comes_from_the_live_articulation_and_observation_rate():
    left = SimpleNamespace(
        type="target",
        arm_name="left_arm",
        arm_joint_indices=list(range(6)),
        arm_joints_name=[f"joint{i}" for i in range(1, 7)],
    )
    right = SimpleNamespace(
        type="target",
        arm_name="right_arm",
        arm_joint_indices=list(range(6)),
        arm_joints_name=[f"joint{i}" for i in range(1, 7)],
    )
    limits = np.stack(
        [np.array([-10.0, 10.0])] * 5 + [np.array([-3.14, 3.14])]
    )
    manager = SimpleNamespace(
        robot_list=[left, right],
        robot_key=[
            SimpleNamespace(data=SimpleNamespace(soft_joint_pos_limits=_Tensor([limits]))),
            SimpleNamespace(data=SimpleNamespace(soft_joint_pos_limits=_Tensor([limits]))),
        ],
    )
    env = SimpleNamespace(
        robot_manager=manager,
        obs_manager=SimpleNamespace(collect_freq=25),
    )

    spec = _action_spec(env)

    assert spec.labels[0] == "left_joint1"
    assert spec.labels[-1] == "right_gripper"
    assert spec.control_hz == 25.0
    assert spec.low.tolist() == [-10.0] * 5 + [-3.14, 0.0] + [-10.0] * 5 + [-3.14, 0.0]
    assert spec.high.tolist() == [10.0] * 5 + [3.14, 1.0] + [10.0] * 5 + [3.14, 1.0]


def test_debug_spec_reads_the_same_robodojo_sources(monkeypatch):
    monkeypatch.setenv(
        "ROBODOJO_ROOT", "/mnt/bn/robotics-data-mx/wenbo/RoboDojo-eval"
    )

    spec = _debug_action_spec()

    assert spec.control_hz == 25.0
    assert spec.labels[5] == "left_joint6"
    assert spec.low[5] == pytest.approx(-3.14)
    assert spec.high[5] == pytest.approx(3.14)
