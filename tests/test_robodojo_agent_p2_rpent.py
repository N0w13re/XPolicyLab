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
