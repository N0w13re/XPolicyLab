import json
from http.server import ThreadingHTTPServer
from pathlib import Path
import socket

import numpy as np
import pytest

from XPolicyLab.policy.Pi_05_Agent_P1_RPent.planner import (
    PLANNER_PROMPT_VERSION,
    SYSTEM_PROMPT,
    TOOLS_SPEC,
    RpentPlanner,
)
from XPolicyLab.policy.Pi_05_Agent_P1_RPent.prompt_versions import (
    RPENT_V0_UPSTREAM_COMMIT,
)
from XPolicyLab.policy.Pi_05_Agent_P1_RPent.deploy import (
    _mark_incomplete_episode_failed,
)
from XPolicyLab.policy.Pi_05_Agent_P1_RPent.geometry import (
    query_world_map,
    sample_world_xyz,
    world_from_depth,
)
from XPolicyLab.policy.Pi_05_Agent_P1_RPent.robot_profile import default_clearance
from XPolicyLab.policy.Pi_05_Agent_P1_RPent.tools import RpentPrimitives
from XPolicyLab.policy.Pi_05_Agent_P1_RPent.trace import EpisodeTrace
from XPolicyLab.policy.Pi_05_Agent_P1_RPent.trace_viewer import (
    HTML,
    IPv6ThreadingHTTPServer,
    build_collection,
    build_manifest,
    server_class_for_host,
)


def _observation(*, left_z=0.9, left_gripper=1.0, right_gripper=1.0):
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    camera = {
        "color": image,
        "depth": np.full((12, 16), 0.5, dtype=np.float32),
        "intrinsic_matrix": np.array(
            [[10.0, 0.0, 8.0], [0.0, 10.0, 6.0], [0.0, 0.0, 1.0]]
        ),
        "extrinsics_matrix": np.eye(4),
    }
    return {
        "instruction": (
            "Put pepper objects into the left basket, car objects into the "
            "middle basket, and chocolate_bar objects into the right basket, "
            "then reset the robot arm."
        ),
        "state": {
            "left_ee_pose": np.array(
                [-0.25, -0.2, left_z, 1.0, 0.0, 0.0, 0.0], dtype=np.float32
            ),
            "right_ee_pose": np.array(
                [0.25, -0.2, 0.9, 1.0, 0.0, 0.0, 0.0], dtype=np.float32
            ),
            "left_ee_joint_state": np.array(
                [left_gripper], dtype=np.float32
            ),
            "right_ee_joint_state": np.array([right_gripper], dtype=np.float32),
        },
        "vision": {
            "cam_head": {**camera, "color": image},
            "cam_left_wrist": {**camera, "color": image + 1},
            "cam_right_wrist": {**camera, "color": image + 2},
        },
    }


class _FakeEnv:
    step_lim = 1100

    def __init__(self, observations):
        self.observations = list(observations)
        self.index = 0
        self.take_action_cnt = [0]
        self.actions = []

    def get_obs(self):
        return self.observations[self.index]

    def take_action(self, action):
        self.actions.append(action)
        self.take_action_cnt[0] += 1
        if self.index + 1 < len(self.observations):
            self.index += 1

    def is_episode_end(self):
        return False


class _FakeWriter:
    def __init__(self, n_frames=0):
        self.n_frames = n_frames


class _FakeRobot:
    arm_name = "left_arm"
    robot_name = "left_robot"
    entity_origin_pose = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]


class _FakePlanner:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def plan_path(self, current, target, real_robot_pose):
        self.calls.append((np.asarray(current), np.asarray(target), real_robot_pose))
        return self.result


class _FakeRobotManager:
    def __init__(self, result):
        self.robot = _FakeRobot()
        self.planner = {self.robot.robot_name: _FakePlanner(result)}

    def get_robot_by_arm_name(self, name):
        assert name == "left_arm"
        return self.robot

    def get_joint(self, robot, env_idx_list):
        assert robot is self.robot
        return {env_idx_list[0]: np.zeros(6, dtype=np.float32)}


class _FakeModelClient:
    def __init__(self, action_horizon=1):
        self.updated = []
        self.action_horizon = action_horizon

    def call(self, *, func_name, **kwargs):
        if func_name == "update_obs":
            self.updated.append(kwargs["obs"])
            return None
        if func_name == "get_action":
            action = {
                "left_arm_joint_state": np.zeros(6, dtype=np.float32),
                "right_arm_joint_state": np.zeros(6, dtype=np.float32),
                "left_ee_joint_state": np.ones(1, dtype=np.float32),
                "right_ee_joint_state": np.ones(1, dtype=np.float32),
            }
            return [
                {
                    key: value.copy()
                    for key, value in action.items()
                }
                for _ in range(self.action_horizon)
            ]
        raise AssertionError(func_name)


class _UnusedQwen:
    pass


class _GroundingQwen:
    def chat(self, messages, **kwargs):
        del messages, kwargs
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"label":"requested target",'
                            '"bbox_2d":[400,300,500,500]}'
                        )
                    }
                }
            ]
        }

    def message_text_and_tools(self, result):
        return result["choices"][0]["message"]["content"], []


class _FinishingQwen:
    def __init__(self):
        self.messages = None

    def chat(self, messages, **kwargs):
        del kwargs
        self.messages = messages
        return {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "finish_call",
                                "type": "function",
                                "function": {
                                    "name": "finish",
                                    "arguments": json.dumps(
                                        {
                                            "status": "test",
                                            "summary": "stop after prompt capture",
                                        }
                                    ),
                                },
                            }
                        ],
                    }
                }
            ]
        }

    def message_text_and_tools(self, result):
        message = result["choices"][0]["message"]
        return message["content"], message["tool_calls"]


class _ObserveThenFinishQwen:
    def __init__(self):
        self.calls = 0

    def chat(self, messages, **kwargs):
        del messages, kwargs
        name = "observe" if self.calls == 0 else "finish"
        arguments = (
            {}
            if name == "observe"
            else {"status": "test", "summary": "frame boundary test complete"}
        )
        self.calls += 1
        return {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": f"{name}_call",
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

    def message_text_and_tools(self, result):
        message = result["choices"][0]["message"]
        return message["content"], message["tool_calls"]


class _FrameRecordingEnv(_FakeEnv):
    def __init__(self, observations):
        super().__init__(observations)
        self.video_writers = {
            0: {
                camera: _FakeWriter()
                for camera in (
                    "cam_head",
                    "cam_left_wrist",
                    "cam_right_wrist",
                )
            }
        }

    def get_obs(self):
        for writer in self.video_writers[0].values():
            writer.n_frames += 1
        return super().get_obs()


def test_pi05_pick_uses_full_episode_instruction_and_post_descent_lift(tmp_path):
    env = _FakeEnv(
        [
            _observation(left_z=0.90, left_gripper=1.0),
            _observation(left_z=0.84, left_gripper=0.1),
            _observation(left_z=0.90, left_gripper=0.1),
        ]
    )
    model = _FakeModelClient()
    primitives = RpentPrimitives(
        env,
        model,
        _UnusedQwen(),
        trace=EpisodeTrace(tmp_path),
    )

    result = primitives.pi05_pick("pick up the pepper", max_chunks=2)

    assert result["candidate_success"]
    assert result["carrying_arm"] == "left"
    assert result["descent_done"]["left"]
    assert result["post_descent_lift_m"]["left"] >= 0.04
    assert [obs["instruction"] for obs in model.updated] == [
        _observation()["instruction"],
        _observation()["instruction"],
    ]


def test_pi05_pick_can_record_every_action_for_diagnostics(tmp_path, monkeypatch):
    env = _FakeEnv([_observation(), _observation()])
    model = _FakeModelClient()
    primitives = RpentPrimitives(
        env,
        model,
        _UnusedQwen(),
        trace=EpisodeTrace(tmp_path),
    )
    observed_indices = []
    original_get_obs = env.get_obs

    def recording_get_obs():
        observed_indices.append(env.index)
        return original_get_obs()

    env.get_obs = recording_get_obs
    monkeypatch.setenv("RPENT_RECORD_EVERY_PI05_ACTION", "1")

    primitives.pi05_pick("pick up the pepper", max_chunks=1)

    assert observed_indices == [0, 0, 1, 1, 1]


def test_pi05_pick_executes_only_configured_action_prefix(tmp_path, monkeypatch):
    env = _FakeEnv([_observation() for _ in range(50)])
    model = _FakeModelClient(action_horizon=50)
    primitives = RpentPrimitives(
        env,
        model,
        _UnusedQwen(),
        trace=EpisodeTrace(tmp_path),
    )
    monkeypatch.setenv("RPENT_PI05_EXECUTION_HORIZON", "20")

    result = primitives.pi05_pick("pick up the pepper", max_chunks=1)

    assert len(env.actions) == 20
    assert result["execution_horizon"] == 20
    assert result["actions_executed"] == 20


def test_move_preserves_current_orientation_by_default(tmp_path):
    start = _observation()
    moved = _observation()
    moved["state"]["right_ee_pose"][3:] = np.array(
        [0.5, 0.5, 0.5, 0.5], dtype=np.float32
    )
    env = _FakeEnv([start, moved])
    primitives = RpentPrimitives(
        env,
        _FakeModelClient(),
        _UnusedQwen(),
        trace=EpisodeTrace(tmp_path),
    )

    primitives.move_to([0.20, -0.20, 0.90], arm="right")

    np.testing.assert_allclose(
        env.actions[0]["right_ee_pose"][3:],
        start["state"]["right_ee_pose"][3:],
        atol=1e-6,
    )


def test_release_only_opens_at_the_current_pose(tmp_path):
    start = _observation(left_z=0.9, left_gripper=0.1)
    opened = _observation(left_z=0.9, left_gripper=1.0)
    env = _FakeEnv([start, opened])
    primitives = RpentPrimitives(
        env,
        _FakeModelClient(),
        _UnusedQwen(),
        trace=EpisodeTrace(tmp_path),
    )
    primitives.last_reached_move["left"] = {
        "target_xyz": [-0.25, -0.2, 0.9],
        "sim_step": 0,
        "carrying": True,
    }
    primitives.ledger.holding_arm = "left"
    primitives.ledger.hold_state = "verified"

    result = primitives.release("left", max_steps=4)

    assert result["opened"]
    assert len(env.actions) == 1
    np.testing.assert_allclose(
        env.actions[0]["left_ee_pose"],
        start["state"]["left_ee_pose"],
    )
    np.testing.assert_allclose(env.actions[0]["left_ee_joint_state"], [1.0])


def test_release_is_rejected_without_a_reached_carry_move(tmp_path):
    env = _FakeEnv([_observation(left_gripper=0.1)])
    primitives = RpentPrimitives(
        env,
        _FakeModelClient(),
        _UnusedQwen(),
        trace=EpisodeTrace(tmp_path),
    )

    result = primitives.release("left", max_steps=4)

    assert result["error"] == "release_requires_reached_carry_move"
    assert not env.actions


_UPSTREAM_RPENT_SECTION_TITLES = (
    "ROLE",
    "READ ORDER",
    "CLEAN-TO-RANDOMIZED TRANSFER",
    "ACCURACY-FIRST LOOP",
    "CONDITIONAL TASK-FAMILY PLAYBOOKS",
    "PERCEPTION",
    "RUNTIME",
    "BUDGET AND SUCCESS",
    "MODE",
)


def _ground_tool_parameters():
    for tool in TOOLS_SPEC:
        if tool["function"]["name"] == "ground":
            return tool["function"]["parameters"]
    raise AssertionError("ground tool not found in TOOLS_SPEC")


def test_default_system_prompt_preserves_upstream_rpent_strategy():
    prompt = SYSTEM_PROMPT

    for title in _UPSTREAM_RPENT_SECTION_TITLES:
        assert title in prompt, f"missing upstream section {title!r}"

    assert "head view as semantic authority" in prompt
    assert "wrist view to refine geometry" in prompt
    assert "sample_world_xyz" in prompt
    assert "query_world_map" in prompt

    perception_start = prompt.index("PERCEPTION")
    wrist_guidance = prompt[perception_start:].split(
        "wrist view to refine geometry", 1
    )[1]
    assert "sample_world_xyz" in wrist_guidance
    assert "query_world_map" in wrist_guidance
    assert "ground" not in wrist_guidance.lower()

    assert "RoboDojo" in prompt
    assert "Pi_05" in prompt or "pi05_act" in prompt
    assert "RoboTwin" not in prompt
    assert "LingBot" not in prompt
    assert "lingbot_act" not in prompt


def test_ground_tool_schema_has_no_clearance_parameter():
    properties = _ground_tool_parameters()["properties"]
    assert "clearance" not in properties


def test_ground_result_has_no_suggested_hover_xyz(tmp_path):
    observation = _observation(right_gripper=0.1)
    observation["vision"]["cam_head"]["intrinsic_matrix"] = np.array(
        [[615.0, 0.0, 8.0], [0.0, 615.0, 6.0], [0.0, 0.0, 1.0]]
    )
    observation["vision"]["cam_head"]["extrinsics_matrix"] = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, -0.20],
            [0.0, 0.0, 1.0, 1.308],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    env = _FakeEnv([observation])
    primitives = RpentPrimitives(
        env,
        _FakeModelClient(),
        _GroundingQwen(),
        trace=EpisodeTrace(tmp_path),
    )

    result = primitives.ground(
        "one requested target",
        camera="head",
        anchor="lower_center",
    )

    assert "suggested_hover_xyz" not in result


def test_move_does_not_use_rectangular_workspace_as_authority(tmp_path):
    env = _FakeEnv([_observation(left_gripper=0.1)])
    primitives = RpentPrimitives(
        env,
        _FakeModelClient(),
        _UnusedQwen(),
        trace=EpisodeTrace(tmp_path),
    )

    result = primitives.move_to([0.30, 0.01, 0.90], arm="left")

    assert result["execution_mode"] == "ee_servo_fallback"
    assert env.actions


def test_ground_returns_one_target_and_geometric_reachability(tmp_path):
    observation = _observation(right_gripper=0.1)
    observation["vision"]["cam_head"]["intrinsic_matrix"] = np.array(
        [[615.0, 0.0, 8.0], [0.0, 615.0, 6.0], [0.0, 0.0, 1.0]]
    )
    observation["vision"]["cam_head"]["extrinsics_matrix"] = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, -0.20],
            [0.0, 0.0, 1.0, 1.308],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    env = _FakeEnv([observation])
    primitives = RpentPrimitives(
        env,
        _FakeModelClient(),
        _GroundingQwen(),
        trace=EpisodeTrace(tmp_path),
    )

    result = primitives.ground(
        "one requested target",
        camera="head",
        anchor="lower_center",
    )

    assert result["query"] == "one requested target"
    assert result["label"] == "requested target"
    assert len(result["anchor_world_xyz"]) == 3
    assert result["world_summary"]["valid_samples"] > 0
    assert result["suggested_hover_xyz"][2] == pytest.approx(1.008)
    assert result["carrying_arm"] is None


def test_default_approach_clearance_is_twenty_centimeters(monkeypatch):
    monkeypatch.delenv("RPENT_APPROACH_CLEARANCE_M", raising=False)

    assert default_clearance() == 0.20


def test_world_from_depth_matches_opengl_camera_geometry():
    depth = np.array([[2.0]], dtype=np.float32)
    intrinsic = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
    extrinsic = np.eye(4)
    extrinsic[:3, 3] = [1.0, 2.0, 3.0]

    world = world_from_depth(depth, intrinsic, extrinsic)

    np.testing.assert_allclose(world[0, 0], [1.0, 2.0, 1.0])


def test_world_map_sampling_ignores_invalid_depth():
    world = np.full((3, 3, 3), np.nan, dtype=np.float32)
    world[1, 1] = [0.1, 0.2, 0.3]
    world[1, 2] = [0.3, 0.4, 0.5]

    sampled = sample_world_xyz(world, [1, 1], radius=1)
    summary = query_world_map(world, [0, 0, 3, 3])

    np.testing.assert_allclose(sampled["xyz"], [0.2, 0.3, 0.4])
    np.testing.assert_allclose(summary["median_xyz"], [0.2, 0.3, 0.4])
    assert summary["valid_samples"] == 2


def test_curobo_plan_failure_executes_no_action(tmp_path):
    env = _FakeEnv([_observation()])
    env.robot_manager = _FakeRobotManager({"status": "Fail"})
    primitives = RpentPrimitives(
        env,
        _FakeModelClient(),
        _UnusedQwen(),
        trace=EpisodeTrace(tmp_path),
    )

    result = primitives.move_to(
        [-0.2, -0.2, 0.9],
        arm="left",
        quat=[1.0, 0.0, 0.0, 0.0],
    )

    assert result["stop_reason"] == "plan_failed"
    assert result["executed_steps"] == 0
    assert not env.actions


def test_curobo_path_executes_joint_waypoints(tmp_path):
    moved = _observation()
    moved["state"]["left_ee_pose"][:3] = [-0.2, -0.2, 0.9]
    env = _FakeEnv([_observation(), moved, moved])
    env.robot_manager = _FakeRobotManager(
        {
            "status": "Success",
            "position": np.array(
                [[0.1] * 6, [0.2] * 6], dtype=np.float32
            ),
            "velocity": np.zeros((2, 6), dtype=np.float32),
        }
    )
    primitives = RpentPrimitives(
        env,
        _FakeModelClient(),
        _UnusedQwen(),
        trace=EpisodeTrace(tmp_path),
    )

    result = primitives.move_to(
        [-0.2, -0.2, 0.9],
        arm="left",
        quat=[1.0, 0.0, 0.0, 0.0],
    )

    assert result["execution_mode"] == "curobo_joint_path"
    assert result["executed_steps"] == 2
    np.testing.assert_allclose(env.actions[-1]["left_arm_joint_state"], [0.2] * 6)


def test_closed_gripper_is_not_verified_hold(tmp_path):
    primitives = RpentPrimitives(
        _FakeEnv([_observation(left_gripper=0.1)]),
        _FakeModelClient(),
        _UnusedQwen(),
        trace=EpisodeTrace(tmp_path),
    )

    snapshot = primitives.snapshot()

    assert snapshot["gripper_state"]["left"] == "closed"
    assert snapshot["manipulation"]["hold_state"] == "empty"
    assert snapshot["carrying_arm"] is None


def test_release_requires_visual_hold_verification(tmp_path):
    primitives = RpentPrimitives(
        _FakeEnv([_observation(left_gripper=0.1)]),
        _FakeModelClient(),
        _UnusedQwen(),
        trace=EpisodeTrace(tmp_path),
    )
    primitives.last_reached_move["left"] = {"carrying": True}

    result = primitives.release("left")

    assert result["error"] == "release_requires_reached_carry_move"


def test_ground_uses_bbox_center_by_default(tmp_path):
    observation = _observation()
    observation["vision"]["cam_head"]["intrinsic_matrix"] = np.eye(3)
    observation["vision"]["cam_head"]["extrinsics_matrix"] = np.eye(4)
    primitives = RpentPrimitives(
        _FakeEnv([observation]),
        _FakeModelClient(),
        _GroundingQwen(),
        trace=EpisodeTrace(tmp_path),
    )
    result = primitives.ground("one requested target")

    assert result["anchor"] == "center"
    assert result["anchor_pixel"] == [7.2, 4.8]
    assert result["env_state_step"] == 0


def test_tool_trace_persists_three_camera_views(tmp_path):
    env = _FakeEnv([_observation()])
    primitives = RpentPrimitives(
        env,
        _FakeModelClient(),
        _UnusedQwen(),
        trace=EpisodeTrace(tmp_path),
    )

    result = primitives.record_tool_result("observe", {}, primitives.observe())

    assert result["trace_step"] == 0
    assert {"head", "left_wrist", "right_wrist"} <= set(result["artifacts"])
    assert {
        "head_depth",
        "head_world_xyz",
        "head_camera",
    } <= set(result["artifacts"])
    for path in result["artifacts"].values():
        assert tmp_path.joinpath("step_000", path.split("/")[-1]).is_file()
    events = [
        json.loads(line)
        for line in (tmp_path / "transcript.jsonl").read_text().splitlines()
    ]
    assert events[0]["type"] == "tool_result"
    assert events[0]["tool"] == "observe"


def test_video_frame_counts_use_live_robodojo_writers(tmp_path):
    env = _FakeEnv([_observation()])
    env.video_writers = {
        0: {
            "cam_head": _FakeWriter(12),
            "cam_left_wrist": _FakeWriter(11),
            "cam_right_wrist": _FakeWriter(10),
        }
    }
    primitives = RpentPrimitives(
        env,
        _FakeModelClient(),
        _UnusedQwen(),
        trace=EpisodeTrace(tmp_path),
    )

    assert primitives.video_frame_counts() == {
        "head": 12,
        "left_wrist": 11,
        "right_wrist": 10,
    }


def test_tool_frame_range_is_written_to_transcript(tmp_path):
    trace = EpisodeTrace(tmp_path)

    trace.record_tool_frame_range(
        step=3,
        turn=4,
        tool="pi05_pick",
        frame_start={"head": 10, "left_wrist": 10},
        frame_end={"head": 31, "left_wrist": 30},
        env_step_start=7,
        env_step_end=27,
    )

    event = json.loads(trace.transcript_path.read_text())
    assert event["type"] == "tool_frame_range"
    assert event["step"] == 3
    assert event["cameras"]["head"] == {"start": 10, "end": 31}
    assert event["cameras"]["left_wrist"] == {"start": 10, "end": 30}


def test_planner_frame_range_includes_tool_and_post_tool_observation(tmp_path):
    env = _FrameRecordingEnv([_observation()])
    qwen = _ObserveThenFinishQwen()
    primitives = RpentPrimitives(
        env,
        _FakeModelClient(),
        qwen,
        trace=EpisodeTrace(tmp_path),
    )

    RpentPlanner(primitives, qwen).run()

    events = [
        json.loads(line)
        for line in (tmp_path / "transcript.jsonl").read_text().splitlines()
    ]
    config = next(event for event in events if event["type"] == "planner_config")
    assert config["prompt_version"] == "v1"
    assert config["system_prompt"] == SYSTEM_PROMPT
    turns = [event for event in events if event["type"] == "planner_turn"]
    assert turns
    assert all(
        event["prompt_version"] == PLANNER_PROMPT_VERSION for event in turns
    )
    ranges = [event for event in events if event["type"] == "tool_frame_range"]
    assert ranges[0]["tool"] == "observe"
    assert ranges[0]["cameras"] == {
        "head": {"start": 2, "end": 4},
        "left_wrist": {"start": 2, "end": 4},
        "right_wrist": {"start": 2, "end": 4},
    }
    assert ranges[1]["tool"] == "finish"
    assert ranges[1]["cameras"]["head"] == {"start": 4, "end": 6}


def test_trace_viewer_manifest_merges_tool_calls_and_extends_video_edges(tmp_path):
    trace_dir = tmp_path / "trace"
    video_dir = tmp_path / "video"
    trace_dir.mkdir()
    video_dir.mkdir()
    for camera in ("head", "left_wrist", "right_wrist"):
        (video_dir / f"episode_0000000_cam_{camera}_fail.mp4").write_bytes(b"mp4")
    events = [
        {
            "type": "planner_turn",
            "turn": 2,
            "text": "continue",
            "tool": "pi05_pick",
            "arguments": {"prompt": "red car"},
        },
        {
            "type": "tool_result",
            "step": 0,
            "tool": "pi05_pick",
            "arguments": {"prompt": "red car"},
            "result": {"carrying_arm": "right"},
            "artifacts": {
                "head_depth_preview": str(
                    trace_dir / "step_000" / "head_depth.png"
                )
            },
        },
        {
            "type": "tool_frame_range",
            "step": 0,
            "turn": 2,
            "tool": "pi05_pick",
            "env_step_start": 5,
            "env_step_end": 25,
            "cameras": {
                camera: {"start": 4, "end": 24}
                for camera in ("head", "left_wrist", "right_wrist")
            },
        },
    ]
    (trace_dir / "transcript.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events)
    )

    manifest = build_manifest(
        trace_dir,
        video_dir,
        probe=lambda path: {
            "fps": 25.0,
            "frame_count": 30,
            "duration": 1.2,
            "width": 640,
            "height": 480,
        },
    )

    assert set(manifest["videos"]) == {"head", "left_wrist", "right_wrist"}
    assert manifest["tools"][0]["arguments"] == {"prompt": "red car"}
    assert manifest["tools"][0]["result"] == {"carrying_arm": "right"}
    assert manifest["tools"][0]["artifacts"]["head_depth_preview"].endswith(
        "head_depth.png"
    )
    assert manifest["tools"][0]["exec_step_count"] == 20
    assert not manifest["tools"][0]["is_zero_step"]
    assert manifest["tools"][0]["cameras"]["head"] == {"start": 0, "end": 30}


def test_trace_viewer_manifest_includes_instruction_and_official_result(tmp_path):
    trace_dir = tmp_path / "trace"
    video_dir = tmp_path / "video"
    trace_dir.mkdir()
    video_dir.mkdir()
    for camera in ("head", "left_wrist", "right_wrist"):
        (video_dir / f"episode_0000000_cam_{camera}_success.mp4").write_bytes(b"mp4")
    events = [
        {
            "type": "tool_result",
            "step": 0,
            "tool": "observe",
            "arguments": {},
            "result": {"instruction": "Pick up the green scissors."},
        },
        {
            "type": "tool_frame_range",
            "step": 0,
            "turn": 0,
            "tool": "observe",
            "env_step_start": 0,
            "env_step_end": 0,
            "cameras": {
                camera: {"start": 0, "end": 1}
                for camera in ("head", "left_wrist", "right_wrist")
            },
        },
    ]
    (trace_dir / "transcript.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events)
    )
    (video_dir / "_result.json").write_text(
        json.dumps(
            {
                "success_rate": 1.0,
                "details": {"0": {"success": True, "score": 1.0}},
            }
        )
    )

    manifest = build_manifest(
        trace_dir,
        video_dir,
        probe=lambda path: {
            "fps": 25.0,
            "frame_count": 1,
            "duration": 0.04,
            "width": 640,
            "height": 480,
        },
    )

    assert manifest["episode"] == {
        "instruction": "Pick up the green scissors.",
        "official_success": True,
        "score": 1.0,
        "layout_id": None,
    }


def test_trace_viewer_collection_separates_tasks_and_episode_results(tmp_path):
    runs = []
    for task, success in (("pickup", True), ("stack", False)):
        trace_root = tmp_path / task / "trace"
        video_dir = tmp_path / task / "video"
        episode_dir = trace_root / "episode_0000000"
        episode_dir.mkdir(parents=True)
        video_dir.mkdir()
        for camera in ("head", "left_wrist", "right_wrist"):
            status = "success" if success else "fail"
            (
                video_dir
                / f"episode_0000000_cam_{camera}_{status}.mp4"
            ).write_bytes(b"mp4")
        events = [
            {
                "type": "tool_result",
                "step": 0,
                "tool": "observe",
                "arguments": {},
                "result": {"instruction": f"Do {task}."},
            },
            {
                "type": "tool_frame_range",
                "step": 0,
                "turn": 0,
                "tool": "observe",
                "env_step_start": 0,
                "env_step_end": 0,
                "cameras": {
                    camera: {"start": 0, "end": 1}
                    for camera in ("head", "left_wrist", "right_wrist")
                },
            },
        ]
        (episode_dir / "transcript.jsonl").write_text(
            "".join(json.dumps(event) + "\n" for event in events)
        )
        (video_dir / "_result.json").write_text(
            json.dumps(
                {
                    "details": {
                        "0": {
                            "layout_id": 0,
                            "success": success,
                            "score": int(success),
                        }
                    }
                }
            )
        )
        runs.append((task, trace_root, video_dir))

    collection, manifests, videos = build_collection(
        runs,
        probe=lambda path: {
            "fps": 25.0,
            "frame_count": 1,
            "duration": 0.04,
            "width": 640,
            "height": 480,
        },
    )

    assert collection["summary"] == {
        "episode_count": 2,
        "finished_count": 2,
        "success_count": 1,
        "failure_count": 1,
    }
    assert [episode["id"] for episode in collection["episodes"]] == [
        "pickup:video:0000000",
        "stack:video:0000000",
    ]
    assert manifests["pickup:video:0000000"]["episode"]["instruction"] == "Do pickup."
    assert manifests["stack:video:0000000"]["episode"]["official_success"] is False
    assert set(videos["pickup:video:0000000"]) == {
        "head",
        "left_wrist",
        "right_wrist",
    }


def test_trace_viewer_collection_uses_result_layout_after_skipped_layout(tmp_path):
    trace_root = tmp_path / "trace"
    video_dir = tmp_path / "video"
    video_dir.mkdir()
    (trace_root / "episodes.json").parent.mkdir(parents=True)
    (trace_root / "episodes.json").write_text(
        json.dumps({"layout_ids": [28, 29, 30]})
    )
    for trace_index, instruction in enumerate(("Pick up tape.", "Pick up tiara.")):
        episode_dir = trace_root / f"episode_{trace_index:07d}"
        episode_dir.mkdir()
        events = [
            {
                "type": "tool_result",
                "step": 0,
                "tool": "observe",
                "arguments": {},
                "result": {"instruction": instruction},
            },
            {
                "type": "tool_frame_range",
                "step": 0,
                "turn": 0,
                "tool": "observe",
                "env_step_start": 0,
                "env_step_end": 0,
                "cameras": {
                    camera: {"start": 0, "end": 1}
                    for camera in ("head", "left_wrist", "right_wrist")
                },
            },
        ]
        (episode_dir / "transcript.jsonl").write_text(
            "".join(json.dumps(event) + "\n" for event in events)
        )
        for camera in ("head", "left_wrist", "right_wrist"):
            (
                video_dir
                / f"episode_{trace_index:07d}_cam_{camera}_fail.mp4"
            ).write_bytes(b"mp4")
    (video_dir / "_result.json").write_text(
        json.dumps(
            {
                "details": {
                    "0": {"layout_id": 28, "success": False, "score": 0},
                    "1": {"layout_id": 30, "success": False, "score": 0},
                }
            }
        )
    )

    collection, manifests, _ = build_collection(
        [("pickup", trace_root, video_dir)],
        probe=lambda path: {
            "fps": 25.0,
            "frame_count": 1,
            "duration": 0.04,
            "width": 640,
            "height": 480,
        },
    )

    assert [episode["layout_id"] for episode in collection["episodes"]] == [28, 30]
    assert manifests["pickup:video:0000030"]["episode"]["instruction"] == (
        "Pick up tiara."
    )
    assert manifests["pickup:video:0000030"]["episode"]["index"] == 1


def test_trace_viewer_manifest_marks_tools_without_environment_actions(tmp_path):
    trace_dir = tmp_path / "trace"
    video_dir = tmp_path / "video"
    trace_dir.mkdir()
    video_dir.mkdir()
    for camera in ("head", "left_wrist", "right_wrist"):
        (video_dir / f"episode_0000000_cam_{camera}_fail.mp4").write_bytes(b"mp4")
    events = [
        {
            "type": "tool_result",
            "step": 0,
            "tool": "ground",
            "arguments": {},
            "result": {},
        },
        {
            "type": "tool_frame_range",
            "step": 0,
            "turn": 0,
            "tool": "ground",
            "env_step_start": 4,
            "env_step_end": 4,
            "cameras": {
                camera: {"start": 1, "end": 3}
                for camera in ("head", "left_wrist", "right_wrist")
            },
        },
    ]
    (trace_dir / "transcript.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events)
    )

    manifest = build_manifest(
        trace_dir,
        video_dir,
        probe=lambda path: {
            "fps": 25.0,
            "frame_count": 4,
            "duration": 0.16,
            "width": 640,
            "height": 480,
        },
    )

    assert manifest["tools"][0]["exec_step_count"] == 0
    assert manifest["tools"][0]["is_zero_step"]


def test_trace_viewer_manifest_converts_ground_bbox_to_video_pixels(tmp_path):
    trace_dir = tmp_path / "trace"
    video_dir = tmp_path / "video"
    trace_dir.mkdir()
    video_dir.mkdir()
    for camera in ("head", "left_wrist", "right_wrist"):
        (video_dir / f"episode_0000000_cam_{camera}_fail.mp4").write_bytes(b"mp4")
    events = [
        {
            "type": "tool_result",
            "step": 0,
            "tool": "ground",
            "arguments": {"query": "green scissors", "camera": "head"},
            "result": {
                "label": "mint green scissors",
                "bbox_2d": [100, 200, 600, 800],
                "anchor": "lower_center",
                "anchor_pixel": [224.0, 336.0],
            },
        },
        {
            "type": "tool_frame_range",
            "step": 0,
            "turn": 0,
            "tool": "ground",
            "env_step_start": 0,
            "env_step_end": 0,
            "cameras": {
                camera: {"start": 1, "end": 3}
                for camera in ("head", "left_wrist", "right_wrist")
            },
        },
    ]
    (trace_dir / "transcript.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events)
    )

    manifest = build_manifest(
        trace_dir,
        video_dir,
        probe=lambda path: {
            "fps": 25.0,
            "frame_count": 4,
            "duration": 0.16,
            "width": 640,
            "height": 480,
        },
    )

    assert manifest["tools"][0]["overlay"] == {
        "camera": "head",
        "bbox_1000": [100.0, 200.0, 600.0, 800.0],
        "bbox_pixel": [64.0, 96.0, 384.0, 384.0],
        "anchor": "lower_center",
        "anchor_pixel": [224.0, 336.0],
        "query": "green scissors",
        "label": "mint green scissors",
    }


def test_trace_viewer_selects_ipv6_server_for_ipv6_bind_address():
    assert server_class_for_host("127.0.0.1") is ThreadingHTTPServer
    assert server_class_for_host("0.0.0.0") is ThreadingHTTPServer
    assert server_class_for_host("::") is IPv6ThreadingHTTPServer
    assert IPv6ThreadingHTTPServer.address_family == socket.AF_INET6


def test_trace_viewer_initial_collection_load_is_safe_without_video_elements():
    assert "if(!primary)return 0" in HTML
    assert "else if(head()&&manifest?.tools?.length)updatePosition()" in HTML


def test_trace_viewer_supports_per_tool_depth_preview():
    assert "RGB / Depth" in HTML
    assert "updateDepthPreviews(tool)" in HTML
    assert "/artifact?path=" in HTML


def test_post_tool_turn_contains_fresh_labeled_images(tmp_path):
    env = _FakeEnv([_observation()])
    primitives = RpentPrimitives(
        env,
        _FakeModelClient(),
        _UnusedQwen(),
        trace=EpisodeTrace(tmp_path),
    )
    planner = RpentPlanner(primitives, _UnusedQwen())

    message = planner._post_tool_turn("move_to", {"reached": True})

    assert message["role"] == "user"
    labels = [
        part["text"]
        for part in message["content"]
        if part["type"] == "text" and part["text"].startswith("[")
    ]
    images = [
        part["image_url"]["url"]
        for part in message["content"]
        if part["type"] == "image_url"
    ]
    assert labels == ["[head camera]", "[left_wrist camera]", "[right_wrist camera]"]
    assert len(images) == 3
    assert all(url.startswith("data:image/jpeg;base64,") for url in images)


def test_old_image_turns_are_compacted_but_latest_is_retained(tmp_path):
    env = _FakeEnv([_observation()])
    primitives = RpentPrimitives(
        env,
        _FakeModelClient(),
        _UnusedQwen(),
        trace=EpisodeTrace(tmp_path),
    )
    planner = RpentPlanner(primitives, _UnusedQwen())
    messages = [
        planner._user_turn("old"),
        {"role": "tool", "content": "{}"},
        planner._user_turn("latest"),
    ]

    planner._compact_old_images(messages)

    assert not any(
        part["type"] == "image_url" for part in messages[0]["content"]
    )
    assert sum(
        part["type"] == "image_url" for part in messages[2]["content"]
    ) == 3


def test_planner_v1_injects_matching_task_recipe_and_records_it(tmp_path):
    env = _FakeEnv([_observation()])
    env.task_name = "classify_objects_by_language"
    qwen = _FinishingQwen()
    primitives = RpentPrimitives(
        env,
        _FakeModelClient(),
        qwen,
        trace=EpisodeTrace(tmp_path),
    )

    RpentPlanner(primitives, qwen).run()

    assert qwen.messages is not None
    prompt = json.dumps(qwen.messages, ensure_ascii=False)
    assert "TASK RECIPE:" in prompt
    assert "Complete all instances of one class" in prompt
    assert "place the object in a grounded free" in prompt
    events = [
        json.loads(line)
        for line in (tmp_path / "transcript.jsonl").read_text().splitlines()
    ]
    config = next(event for event in events if event["type"] == "planner_config")
    assert config["prompt_version"] == "v1"
    assert config["recipe_path"].endswith(
        "recipes/classify_objects_by_language.md"
    )
    assert config["recipe"].startswith("# Classify Objects by Language")


def test_planner_v1_runs_without_recipe_for_unknown_task(tmp_path):
    env = _FakeEnv([_observation()])
    env.task_name = "task_without_recipe"
    qwen = _FinishingQwen()
    primitives = RpentPrimitives(
        env,
        _FakeModelClient(),
        qwen,
        trace=EpisodeTrace(tmp_path),
    )

    RpentPlanner(primitives, qwen).run()

    assert qwen.messages is not None
    prompt = json.dumps(qwen.messages, ensure_ascii=False)
    assert "No task recipe is available" in prompt
    events = [
        json.loads(line)
        for line in (tmp_path / "transcript.jsonl").read_text().splitlines()
    ]
    config = next(event for event in events if event["type"] == "planner_config")
    assert config["recipe_path"] is None
    assert config["recipe"] is None


def test_planner_can_stop_after_first_completed_pi05_call(tmp_path, monkeypatch):
    class Pi05ThenFinishQwen:
        def __init__(self):
            self.calls = 0

        def chat(self, messages, **kwargs):
            del messages, kwargs
            self.calls += 1
            name = "pi05_pick" if self.calls == 1 else "finish"
            arguments = (
                {"prompt": "pick up the requested object", "max_chunks": 1}
                if name == "pi05_pick"
                else {"status": "test", "summary": "unexpected second call"}
            )
            return {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": f"{name}_call",
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

        def message_text_and_tools(self, result):
            message = result["choices"][0]["message"]
            return message["content"], message["tool_calls"]

    monkeypatch.setenv("RPENT_STOP_AFTER_FIRST_PI05", "1")
    env = _FakeEnv([_observation()])
    qwen = Pi05ThenFinishQwen()
    primitives = RpentPrimitives(
        env,
        _FakeModelClient(),
        qwen,
        trace=EpisodeTrace(tmp_path),
    )

    RpentPlanner(primitives, qwen).run()

    assert qwen.calls == 1
    events = [
        json.loads(line)
        for line in (tmp_path / "transcript.jsonl").read_text().splitlines()
    ]
    stop = next(event for event in events if event["type"] == "diagnostic_stop")
    assert stop["reason"] == "first_pi05_act_completed"
    assert stop["actions_executed"] == 1


def test_planner_v0_uses_and_records_upstream_rpent_prompt(tmp_path, monkeypatch):
    monkeypatch.setenv("RPENT_PLANNER_PROMPT_VERSION", "v0")
    env = _FakeEnv([_observation()])
    env.task_name = "classify_objects_by_language"
    env.seed = 3
    env.task_config = "RoboDojo"
    qwen = _FinishingQwen()
    primitives = RpentPrimitives(
        env,
        _FakeModelClient(),
        qwen,
        trace=EpisodeTrace(tmp_path),
    )

    RpentPlanner(primitives, qwen).run()

    assert qwen.messages is not None
    assert "You control one dual-arm RoboTwin" in qwen.messages[0]["content"]
    opening = qwen.messages[1]["content"][0]["text"]
    assert "- task: classify_objects_by_language" in opening
    assert "- seed: 3" in opening
    assert "first unmet recipe phase" in opening
    events = [
        json.loads(line)
        for line in (tmp_path / "transcript.jsonl").read_text().splitlines()
    ]
    config = next(event for event in events if event["type"] == "planner_config")
    assert config["prompt_version"] == "v0"
    assert config["upstream_commit"] == RPENT_V0_UPSTREAM_COMMIT
    turn = next(event for event in events if event["type"] == "planner_turn")
    assert turn["prompt_version"] == "v0"


def test_planner_rejects_unknown_prompt_version(tmp_path, monkeypatch):
    monkeypatch.setenv("RPENT_PLANNER_PROMPT_VERSION", "unknown")
    primitives = RpentPrimitives(
        _FakeEnv([_observation()]),
        _FakeModelClient(),
        _UnusedQwen(),
        trace=EpisodeTrace(tmp_path),
    )

    try:
        RpentPlanner(primitives, _UnusedQwen())
    except ValueError as exc:
        assert "must be one of: v0, v1" in str(exc)
    else:
        raise AssertionError("unknown prompt version was accepted")


def test_incomplete_planner_exit_is_never_counted_as_success():
    class Env:
        success = [True]
        end_flag = [False]

        def get_running_env_idx_list(self):
            return [0]

        def is_episode_end(self):
            if not self.success[0]:
                self.end_flag[0] = True
            return self.end_flag[0]

    env = Env()

    _mark_incomplete_episode_failed(env)

    assert env.success == [False]
    assert env.end_flag == [True]
