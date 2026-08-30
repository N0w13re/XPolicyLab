import numpy as np
import pytest

from XPolicyLab.policy.Pi_05_Agent_P1.deploy import (
    _build_hover_action,
    _delivery_state,
    _make_deliver_action,
    _make_reset_action,
    _track_gripper_transition,
    arm_can_reach,
    closed_arm,
    decide_reprime,
    deliver_target,
    relay_target,
    select_pick_arm,
)
from XPolicyLab.policy.Pi_05_Agent_P1.instruction import (
    destination_side,
    parse_basket_assignment,
)
from XPolicyLab.policy.Pi_05_Agent_P1.table_mapper import (
    TableMapConfig,
    TablePlaneMapper,
    bbox_to_basket_anchor,
    bbox_to_table_anchor,
)


OFFICIAL_INSTRUCTION = (
    "Put pepper objects into the left basket, car objects into the middle "
    "basket, and chocolate_bar objects into the right basket, then reset the "
    "robot arm."
)


def _head_camera_extrinsic() -> np.ndarray:
    angle = np.deg2rad(30.0)
    rotation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angle), -np.sin(angle)],
            [0.0, np.sin(angle), np.cos(angle)],
        ]
    )
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = rotation
    extrinsic[:3, 3] = [0.0, -0.41, 1.308]
    return extrinsic


def _project_world_to_pixel(point_world: np.ndarray, intrinsic: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    point_camera_gl = extrinsic[:3, :3].T @ (point_world - extrinsic[:3, 3])
    point_camera_cv = np.array(
        [point_camera_gl[0], -point_camera_gl[1], -point_camera_gl[2]]
    )
    pixel_h = intrinsic @ (point_camera_cv / point_camera_cv[2])
    return pixel_h[:2]


def test_pixel_ray_recovers_table_point() -> None:
    intrinsic = np.array([[374.56, 0.0, 320.0], [0.0, 374.56, 240.0], [0.0, 0.0, 1.0]])
    extrinsic = _head_camera_extrinsic()
    expected = np.array([0.21, -0.12, 0.765])
    pixel = _project_world_to_pixel(expected, intrinsic, extrinsic)

    mapper = TablePlaneMapper(TableMapConfig())
    actual = mapper.pixel_to_table(pixel, intrinsic, extrinsic)

    np.testing.assert_allclose(actual, expected, atol=1e-8)


def test_bbox_anchor_uses_lower_center() -> None:
    anchor = bbox_to_table_anchor([250, 100, 450, 500], (480, 640, 3))
    np.testing.assert_allclose(anchor, [224.0, 211.2])


def _reprime_state() -> tuple[dict[int, bool], dict[int, int], set[int], dict[int, int]]:
    return {}, {}, set(), {}


def test_release_reprime_bypasses_min_gap_after_real_grasp() -> None:
    was_closed, closed_step, started, last = _reprime_state()
    decide_reprime(
        reprime_enabled=True,
        grippers_open=False,
        step_count=40,
        env_idx=0,
        mode="release",
        gripper_was_closed=was_closed,
        gripper_closed_step=closed_step,
        episode_started=started,
        last_reprime_step=last,
        min_gap=80,
        grasp_min_steps=15,
    )
    assert decide_reprime(
        reprime_enabled=True,
        grippers_open=True,
        step_count=60,
        env_idx=0,
        mode="release",
        gripper_was_closed=was_closed,
        gripper_closed_step=closed_step,
        episode_started={0},
        last_reprime_step={0: 0},
        min_gap=80,
        grasp_min_steps=15,
    )


def test_failed_grasp_release_is_throttled_by_min_gap() -> None:
    was_closed, closed_step, started, last = _reprime_state()
    decide_reprime(
        reprime_enabled=True,
        grippers_open=False,
        step_count=10,
        env_idx=0,
        mode="release",
        gripper_was_closed=was_closed,
        gripper_closed_step=closed_step,
        episode_started={0},
        last_reprime_step={0: 0},
        min_gap=80,
        grasp_min_steps=15,
    )
    assert not decide_reprime(
        reprime_enabled=True,
        grippers_open=True,
        step_count=12,
        env_idx=0,
        mode="release",
        gripper_was_closed=was_closed,
        gripper_closed_step=closed_step,
        episode_started={0},
        last_reprime_step={0: 0},
        min_gap=80,
        grasp_min_steps=15,
    )


def test_hover_restores_episode_neutral_orientation() -> None:
    observation = {
        "state": {
            "left_ee_pose": np.array([-0.2, -0.1, 0.9, 0.0, 1.0, 0.0, 0.0]),
            "right_ee_pose": np.array([0.2, -0.1, 0.9, 0.0, 0.0, 1.0, 0.0]),
            "left_ee_joint_state": np.array([1.0]),
            "right_ee_joint_state": np.array([1.0]),
        }
    }
    neutral = {
        "left": np.array([1.0, 0.0, 0.0, 0.0]),
        "right": np.array([1.0, 0.0, 0.0, 0.0]),
    }
    action, arm = _build_hover_action(
        observation,
        np.array([-0.3, -0.2, 0.865]),
        neutral_quaternions=neutral,
    )
    assert arm == "left"
    np.testing.assert_allclose(action["left_ee_pose"][3:], neutral["left"])
    np.testing.assert_allclose(action["right_ee_pose"][3:], neutral["right"])


def test_within_chunk_close_open_cycle_is_observed() -> None:
    class FakeEnv:
        take_action_cnt = [20]

    task_env = FakeEnv()
    closed = {
        "state": {
            "left_ee_joint_state": np.array([0.0]),
            "right_ee_joint_state": np.array([1.0]),
        }
    }
    opened = {
        "state": {
            "left_ee_joint_state": np.array([1.0]),
            "right_ee_joint_state": np.array([1.0]),
        }
    }

    assert not _track_gripper_transition(closed, 0, task_env)
    task_env.take_action_cnt[0] = 35
    assert _track_gripper_transition(opened, 0, task_env)


def test_official_instruction_binds_every_category_to_a_basket() -> None:
    assignment = parse_basket_assignment(OFFICIAL_INSTRUCTION)
    assert assignment == {
        "pepper": "left",
        "car": "middle",
        "chocolate_bar": "right",
    }


def test_destination_side_tolerates_locator_label_spacing() -> None:
    assert destination_side(OFFICIAL_INSTRUCTION, "chocolate bar") == "right"
    assert destination_side(OFFICIAL_INSTRUCTION, "Pepper") == "left"
    assert destination_side(OFFICIAL_INSTRUCTION, "watch") is None
    assert destination_side(OFFICIAL_INSTRUCTION, None) is None


def test_basket_anchor_uses_front_rim() -> None:
    anchor = bbox_to_basket_anchor([200, 100, 400, 500], (480, 640, 3))
    np.testing.assert_allclose(anchor, [192.0, 240.0])


def test_delivery_hover_keeps_the_carrying_arm_and_grip() -> None:
    observation = {
        "state": {
            "left_ee_pose": np.array([-0.2, -0.1, 0.9, 1.0, 0.0, 0.0, 0.0]),
            "right_ee_pose": np.array([0.2, -0.1, 0.9, 1.0, 0.0, 0.0, 0.0]),
            "left_ee_joint_state": np.array([0.1]),
            "right_ee_joint_state": np.array([1.0]),
        }
    }
    assert closed_arm(observation) == "left"

    basket_xyz = np.array([0.28, 0.05, 0.945])
    action, arm = _build_hover_action(observation, basket_xyz, arm="left")

    # Without the explicit arm the positive x would have chosen the right arm,
    # which is not the one holding the object.
    assert arm == "left"
    np.testing.assert_allclose(action["left_ee_pose"][:3], basket_xyz)
    np.testing.assert_allclose(action["left_ee_joint_state"], [0.1])


def test_pick_arm_follows_the_object_side() -> None:
    # Pi_05 grasps with the arm on the object's side, so P1 must agree with it;
    # picking by destination basket put two arms on one object.
    assert select_pick_arm(-0.05) == "left"
    assert select_pick_arm(0.05) == "right"
    assert select_pick_arm(-0.40) == "left"
    assert select_pick_arm(0.40) == "right"


def test_cross_body_reach_is_recognised() -> None:
    assert arm_can_reach("left", -0.29, 0.15)
    assert not arm_can_reach("right", -0.29, 0.15)
    assert arm_can_reach("right", 0.29, 0.15)
    assert not arm_can_reach("left", 0.29, 0.15)
    # Both arms can serve the middle basket.
    assert arm_can_reach("left", 0.0, 0.15)
    assert arm_can_reach("right", 0.0, 0.15)


def test_delivery_depth_step_is_capped() -> None:
    # Commanding the basket depth outright is unreachable for a fixed wrist
    # orientation, so only a bounded step toward it is asked for.
    carrying = np.array([0.10, -0.19, 0.95], dtype=np.float32)
    basket = np.array([-0.29, 0.01, 0.895], dtype=np.float32)

    target = deliver_target(
        carrying, basket, "basket", min_z=0.895, z_margin=0.06, depth_step=0.08
    )

    np.testing.assert_allclose(target[0], -0.29, atol=1e-6)
    np.testing.assert_allclose(target[1], -0.11, atol=1e-6)
    assert 0.895 <= target[2] <= 0.955


def test_delivery_depth_step_does_not_overshoot_a_near_basket() -> None:
    carrying = np.array([0.10, -0.02, 0.95], dtype=np.float32)
    basket = np.array([-0.29, 0.01, 0.895], dtype=np.float32)

    target = deliver_target(
        carrying, basket, "basket", min_z=0.895, z_margin=0.06, depth_step=0.08
    )

    # The gap is smaller than the cap, so the basket depth is reached exactly.
    np.testing.assert_allclose(target[1], 0.01, atol=1e-6)


def test_basket_pose_mode_still_commands_the_mapped_point(monkeypatch) -> None:
    monkeypatch.setenv("P1_IDENTIFY_HELD", "0")
    monkeypatch.setenv("P1_DELIVER_MODE", "basket_pose")

    task_env = _FakeTaskEnv()
    observation = _carrying_observation(-0.10)
    basket = np.array([-0.29, 0.01, 0.895])
    _delivery_state(task_env)[0][0] = "pepper"
    _delivery_state(task_env)[2][0] = {
        "left": basket,
        "middle": np.array([0.0, 0.01, 0.895]),
        "right": np.array([0.29, 0.01, 0.895]),
    }

    _, _, target = _make_deliver_action(
        task_env, 0, observation, TablePlaneMapper(TableMapConfig())
    )

    np.testing.assert_allclose(target, basket)


def test_lateral_delivery_keeps_reachable_depth() -> None:
    carrying = np.array([0.10, -0.19, 0.90], dtype=np.float32)
    basket = np.array([-0.29, 0.017, 0.895], dtype=np.float32)

    target = deliver_target(carrying, basket, "lateral", min_z=float(basket[2]))

    # Only the axis that tells the three baskets apart is corrected.
    assert target[0] == np.float32(-0.29)
    assert target[1] == carrying[1]
    # The arm is never asked to drop below the release clearance.
    assert target[2] >= basket[2]


def test_lateral_delivery_lifts_to_the_release_clearance() -> None:
    carrying = np.array([0.10, -0.19, 0.80], dtype=np.float32)
    basket = np.array([0.29, 0.017, 0.895], dtype=np.float32)

    target = deliver_target(carrying, basket, "lateral", min_z=float(basket[2]))

    np.testing.assert_allclose(target[2], 0.895, atol=1e-6)


def test_basket_pose_delivery_mode_uses_the_mapped_basket() -> None:
    carrying = np.array([0.10, -0.19, 0.90], dtype=np.float32)
    basket = np.array([-0.29, 0.017, 0.895], dtype=np.float32)

    target = deliver_target(carrying, basket, "basket_pose", min_z=float(basket[2]))

    np.testing.assert_allclose(target, basket)


def test_delivery_release_height_is_clamped() -> None:
    basket = np.array([-0.29, 0.017, 0.895], dtype=np.float32)

    # Pi_05 often carries far above the table; releasing from there would bounce
    # the object out of the basket.
    high_carry = np.array([0.10, -0.19, 1.056], dtype=np.float32)
    target = deliver_target(high_carry, basket, "lateral", min_z=0.895, z_margin=0.06)
    np.testing.assert_allclose(target[2], 0.955, atol=1e-6)

    # A carry that is already inside the band is left alone.
    low_carry = np.array([0.10, -0.19, 0.92], dtype=np.float32)
    target = deliver_target(low_carry, basket, "lateral", min_z=0.895, z_margin=0.06)
    np.testing.assert_allclose(target[2], 0.92, atol=1e-6)


class _FakeTaskEnv:
    """Minimum surface `_make_deliver_action` touches."""

    step_lim = 1100

    def __init__(self) -> None:
        self.take_action_cnt = [0]
        self.scene = None
        self.scene_manager = None


def _carrying_observation(gripper_x: float) -> dict:
    return {
        "instruction": OFFICIAL_INSTRUCTION,
        "state": {
            "left_ee_pose": np.array([gripper_x, -0.19, 1.05, 1.0, 0.0, 0.0, 0.0]),
            "right_ee_pose": np.array([0.30, -0.19, 1.05, 1.0, 0.0, 0.0, 0.0]),
            "left_ee_joint_state": np.array([0.1]),
            "right_ee_joint_state": np.array([1.0]),
        },
        "vision": {},
    }


def test_make_deliver_action_drives_the_carrying_arm_to_the_named_basket(monkeypatch) -> None:
    monkeypatch.setenv("P1_IDENTIFY_HELD", "0")
    monkeypatch.delenv("P1_DELIVER_MODE", raising=False)

    task_env = _FakeTaskEnv()
    observation = _carrying_observation(-0.10)

    # The instruction sends pepper to the left basket.
    _delivery_state(task_env)[0][0] = "pepper"
    _delivery_state(task_env)[2][0] = {
        "left": np.array([-0.29, -0.04, 0.895]),
        "middle": np.array([0.00, -0.04, 0.895]),
        "right": np.array([0.29, -0.04, 0.895]),
    }

    action, arm, target = _make_deliver_action(
        task_env, 0, observation, TablePlaneMapper(TableMapConfig())
    )

    assert arm == "left"
    np.testing.assert_allclose(target[0], -0.29, atol=1e-6)
    # The closed gripper keeps its grip while being repositioned.
    np.testing.assert_allclose(action["left_ee_joint_state"], [0.1])
    np.testing.assert_allclose(action["left_ee_pose"][:3], target)


def test_make_deliver_action_rejects_an_unbound_category(monkeypatch) -> None:
    monkeypatch.setenv("P1_IDENTIFY_HELD", "0")

    task_env = _FakeTaskEnv()
    _delivery_state(task_env)[0][0] = "watch"

    with pytest.raises(ValueError, match="does not bind"):
        _make_deliver_action(
            task_env,
            0,
            _carrying_observation(-0.10),
            TablePlaneMapper(TableMapConfig()),
        )


def test_relay_moves_a_cross_body_object_between_baskets() -> None:
    # Left arm carrying an object bound for the right basket at x=+0.29.
    carrying = np.array([-0.20, -0.19, 1.05], dtype=np.float32)

    target = relay_target(carrying, 0.29, cross_reach=0.15, min_z=0.895)

    # Inside the left arm's reach, and in the gap between the middle basket at
    # x=0.0 and the right basket at x=+0.29 rather than over either mouth.
    assert arm_can_reach("left", float(target[0]), 0.15)
    np.testing.assert_allclose(target[0], 0.135, atol=1e-6)
    assert 0.0 < target[0] < 0.29
    # Depth stays in the object zone, in front of the baskets.
    np.testing.assert_allclose(target[1], -0.19, atol=1e-6)
    assert 0.895 <= target[2] <= 0.955


def test_relay_direction_follows_the_destination_side() -> None:
    carrying = np.array([0.20, -0.19, 0.95], dtype=np.float32)
    target = relay_target(carrying, -0.29, cross_reach=0.15, min_z=0.895)
    assert arm_can_reach("right", float(target[0]), 0.15)
    np.testing.assert_allclose(target[0], -0.135, atol=1e-6)


def test_cross_body_delivery_relays_instead_of_raising(monkeypatch) -> None:
    monkeypatch.setenv("P1_IDENTIFY_HELD", "0")
    monkeypatch.delenv("P1_DELIVER_MODE", raising=False)
    monkeypatch.delenv("P1_RELAY", raising=False)

    task_env = _FakeTaskEnv()
    # Left gripper closed, carrying a chocolate_bar bound for the right basket.
    observation = _carrying_observation(-0.20)
    _delivery_state(task_env)[0][0] = "chocolate_bar"
    _delivery_state(task_env)[2][0] = {
        "left": np.array([-0.29, 0.01, 0.895]),
        "middle": np.array([0.00, 0.01, 0.895]),
        "right": np.array([0.29, 0.01, 0.895]),
    }

    _, arm, target = _make_deliver_action(
        task_env, 0, observation, TablePlaneMapper(TableMapConfig())
    )

    assert arm == "left"
    np.testing.assert_allclose(target[0], 0.135, atol=1e-6)


def test_relay_can_be_disabled_for_ablation(monkeypatch) -> None:
    monkeypatch.setenv("P1_IDENTIFY_HELD", "0")
    monkeypatch.setenv("P1_RELAY", "0")

    task_env = _FakeTaskEnv()
    _delivery_state(task_env)[0][0] = "chocolate_bar"
    _delivery_state(task_env)[2][0] = {
        "left": np.array([-0.29, 0.01, 0.895]),
        "middle": np.array([0.00, 0.01, 0.895]),
        "right": np.array([0.29, 0.01, 0.895]),
    }

    with pytest.raises(ValueError, match="cannot reach"):
        _make_deliver_action(
            task_env,
            0,
            _carrying_observation(-0.20),
            TablePlaneMapper(TableMapConfig()),
        )


def _assert_grippers_untouched(action: dict, observation: dict) -> None:
    state = observation["state"]
    for arm in ("left", "right"):
        key = f"{arm}_ee_joint_state"
        np.testing.assert_allclose(
            action[key], np.asarray(state[key], dtype=np.float32), atol=0.0
        )


def test_no_guidance_action_ever_commands_a_gripper(monkeypatch) -> None:
    """The agent guides poses; grasping, releasing and relaying are Pi_05's.

    Every action this adapter writes to the simulator must leave the gripper
    channel exactly as observed, otherwise the agent is performing the
    manipulation rather than guiding it.
    """
    monkeypatch.setenv("P1_IDENTIFY_HELD", "0")
    monkeypatch.delenv("P1_DELIVER_MODE", raising=False)
    monkeypatch.delenv("P1_RELAY", raising=False)

    baskets = {
        "left": np.array([-0.29, 0.01, 0.895]),
        "middle": np.array([0.00, 0.01, 0.895]),
        "right": np.array([0.29, 0.01, 0.895]),
    }
    mapper = TablePlaneMapper(TableMapConfig())

    # Approach hover, with both grippers open.
    open_observation = {
        "instruction": OFFICIAL_INSTRUCTION,
        "state": {
            "left_ee_pose": np.array([-0.2, -0.19, 0.9, 1.0, 0.0, 0.0, 0.0]),
            "right_ee_pose": np.array([0.2, -0.19, 0.9, 1.0, 0.0, 0.0, 0.0]),
            "left_ee_joint_state": np.array([1.0]),
            "right_ee_joint_state": np.array([0.93]),
        },
    }
    action, _ = _build_hover_action(open_observation, np.array([-0.25, -0.18, 0.865]))
    _assert_grippers_untouched(action, open_observation)

    # Same-side delivery, left gripper closed on a left-basket category.
    task_env = _FakeTaskEnv()
    _delivery_state(task_env)[0][0] = "pepper"
    _delivery_state(task_env)[2][0] = baskets
    carrying = _carrying_observation(-0.20)
    action, _, _ = _make_deliver_action(task_env, 0, carrying, mapper)
    _assert_grippers_untouched(action, carrying)

    # Cross-body relay, left gripper closed on a right-basket category.
    relay_env = _FakeTaskEnv()
    _delivery_state(relay_env)[0][0] = "chocolate_bar"
    _delivery_state(relay_env)[2][0] = baskets
    action, _, _ = _make_deliver_action(relay_env, 0, carrying, mapper)
    _assert_grippers_untouched(action, carrying)

    # Final reset, which previously forced both grippers open.
    action, _, _ = _make_reset_action(_FakeTaskEnv(), 0, carrying)
    _assert_grippers_untouched(action, carrying)
