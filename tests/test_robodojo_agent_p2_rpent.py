import inspect

import pytest


def _tool_functions():
    from XPolicyLab.policy.RoboDojo_Agent_P2_RPent.planner import TOOLS_SPEC

    return {
        tool["function"]["name"]: tool["function"]
        for tool in TOOLS_SPEC
    }


def test_p2_exposes_only_atomic_motion_and_gripper_tools():
    functions = _tool_functions()

    mutating = {
        "move_to",
        "set_gripper",
        "return_home",
    }
    assert mutating <= functions.keys()
    assert {
        "pi05_act",
        "pi05_pick",
        "pregrasp",
        "pick",
        "place",
        "release",
        "rotate_wrist",
        "hold_position",
    }.isdisjoint(functions)


def test_p2_registers_only_tools_the_p1_base_can_dispatch():
    from XPolicyLab.policy.Pi_05_Agent_P1_RPent.planner import (
        TOOLS_SPEC as P1_TOOLS_SPEC,
    )

    p1_names = {tool["function"]["name"] for tool in P1_TOOLS_SPEC}
    # render was retired in the P1 base; keeping it would burn a planner turn
    # on an "unknown tool" result.
    assert "render" not in _tool_functions()
    assert _tool_functions().keys() <= p1_names


def test_geometry_step_params_explain_env_state_step():
    for name in ("view_env_state", "sample_world_xyz", "query_world_map"):
        step = _tool_functions()[name]["parameters"]["properties"]["step"]
        assert "env_state_step" in step["description"]
        assert "env_steps" in step["description"]


def test_move_to_schema_requires_explicit_quaternion():
    move_to = _tool_functions()["move_to"]

    assert set(move_to["parameters"]["required"]) == {"xyz", "arm", "quat"}
    quat = move_to["parameters"]["properties"]["quat"]
    assert quat["minItems"] == quat["maxItems"] == 4


def test_p2_move_to_rejects_missing_quaternion_before_reading_environment():
    from XPolicyLab.policy.RoboDojo_Agent_P2_RPent.tools import P2Primitives

    primitives = P2Primitives(object(), None, None)

    with pytest.raises(ValueError, match="explicit quat"):
        primitives.move_to(xyz=[0.1, 0.2, 0.3], arm="right")


def test_p2_model_has_no_policy_action_path():
    from XPolicyLab.policy.RoboDojo_Agent_P2_RPent.model import Model

    model = Model({"action_type": "joint", "env_cfg_type": "arx_x5"})

    assert model.get_action() == []
    assert model.get_action_batch([0, 1]) == [[], []]
    assert "Pi_05" not in inspect.getsource(Model)


def test_debug_without_planner_key_fails_instead_of_falling_back_to_pi05(monkeypatch):
    from XPolicyLab.policy.RoboDojo_Agent_P2_RPent import deploy

    class _UnavailablePlanner:
        @staticmethod
        def available():
            return False

    class _ModelClient:
        calls = []

        def call(self, *, func_name, **kwargs):
            self.calls.append((func_name, kwargs))

    monkeypatch.setenv("EVAL_ENV_TYPE", "debug")
    monkeypatch.setattr(deploy, "create_planner_llm", lambda: _UnavailablePlanner())
    model_client = _ModelClient()

    with pytest.raises(RuntimeError, match="planner backend"):
        deploy.eval_one_episode(object(), model_client)
    assert model_client.calls == [("reset", {})]


def test_eval_client_can_target_a_separate_robodojo_workspace():
    script = (
        __import__("pathlib").Path(__file__).parents[1]
        / "policy/RoboDojo_Agent_P2_RPent/setup_eval_env_client.sh"
    ).read_text(encoding="utf-8")

    assert 'EVAL_ROOT="${ROBODOJO_ROOT:-${BENCH_ROOT}}"' in script
    assert '"${EVAL_ROOT}/scripts/eval_policy.sh"' in script
    assert '--root_dir "${EVAL_ROOT}"' in script


def test_fixed_layout_runner_enables_metric_depth():
    script = (
        __import__("pathlib").Path(__file__).parents[1]
        / "policy/RoboDojo_Agent_P2_RPent/run_fixed_layout.sh"
    ).read_text(encoding="utf-8")

    assert "export ROBODOJO_ENABLE_METRIC_DEPTH=1" in script


@pytest.mark.parametrize(
    ("points", "expected"),
    [
        ([700, 400], [[192, 448]]),
        ([[700, 400], [1000, 1000]], [[192, 448], [479, 639]]),
    ],
)
def test_qwen_normalized_xy_is_converted_to_pixel_row_col(points, expected):
    from XPolicyLab.policy.RoboDojo_Agent_P2_RPent.tools import (
        normalized_xy_to_pixel_rc,
    )

    assert normalized_xy_to_pixel_rc(points, (480, 640)) == expected


def test_geometry_tool_schema_uses_qwen_normalized_xy_coordinates():
    functions = _tool_functions()

    assert "[x,y]" in functions["sample_world_xyz"]["description"]
    assert "0..1000" in functions["sample_world_xyz"]["description"]
    assert "[x0,y0,x1,y1]" in functions["query_world_map"]["description"]


def test_top_down_eef_targets_raise_surface_points_by_flange_offset():
    from XPolicyLab.policy.RoboDojo_Agent_P2_RPent.tools import (
        top_down_eef_targets_from_surface,
    )

    targets = top_down_eef_targets_from_surface([0.36538, -0.03354, 0.76557])

    assert targets["suggested_contact_eef_xyz"] == pytest.approx(
        [0.36538, -0.03354, 0.91057], abs=1e-5
    )
    assert targets["suggested_hover_eef_xyz"] == pytest.approx(
        [0.36538, -0.03354, 1.03057], abs=1e-5
    )


def test_failed_move_to_attaches_flange_offset_remediation(monkeypatch):
    from XPolicyLab.policy.RoboDojo_Agent_P2_RPent import tools as tools_mod
    from XPolicyLab.policy.RoboDojo_Agent_P2_RPent.tools import P2Primitives

    primitives = P2Primitives.__new__(P2Primitives)

    def _fake_parent_move_to(self, **kwargs):
        return {
            "success": False,
            "stop_reason": "plan_failed",
            "target_xyz": kwargs.get("xyz"),
        }

    monkeypatch.setattr(
        tools_mod.RpentPrimitives,
        "move_to",
        _fake_parent_move_to,
        raising=True,
    )
    result = P2Primitives.move_to(
        primitives,
        xyz=[0.36538, -0.03354, 0.76557],
        arm="right",
        quat=[-0.353523, 0.61239, -0.353524, -0.61239],
    )
    assert result["success"] is False
    assert "remediation" in result
    assert result["remediation"]["suggested_hover_eef_xyz"] == pytest.approx(
        [0.36538, -0.03354, 1.03057], abs=1e-5
    )
