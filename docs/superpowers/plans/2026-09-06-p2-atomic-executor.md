# P2 Atomic Executor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an eval-only RoboDojo P2 adapter that attempts `general_pickup`
with explicit `move_to` and `set_gripper` calls and no VLA contact policy.

**Architecture:** The adapter reuses the tested P1-RPent RGB-D, trace, motion,
and planner-client infrastructure through imports, but defines its own planner
tool surface and P2-only prompt. A no-action model satisfies the policy-server
RPC contract while `deploy.py` owns all environment actions.

**Tech Stack:** Python, pytest, RoboDojo evaluation API, CuRobo path planning,
OpenAI-compatible tool calling.

## Global Constraints

- Policy name is `RoboDojo_Agent_P2_RPent`.
- Initial task is `general_pickup`, robot is `arx_x5`, action type is `joint`.
- `move_to` requires explicit `xyz`, `arm`, and `[qw, qx, qy, qz]` quaternion.
- Do not expose or call `pi05_act`, `pregrasp`, `pick`, or `place`.
- Images remain RGB and already decoded.
- Only official environment success counts.

---

### Task 1: Lock the P2 planner boundary

**Files:**
- Create: `tests/test_robodojo_agent_p2_rpent.py`
- Create: `policy/RoboDojo_Agent_P2_RPent/planner.py`
- Create: `policy/RoboDojo_Agent_P2_RPent/prompts.py`

**Interfaces:**
- Produces: `P2Planner(primitives, qwen)` and `TOOLS_SPEC`.
- Consumes: P1-RPent planner loop and primitive implementation.

- [ ] Write tests asserting the exact mutating tool set and required `quat`.
- [ ] Run the focused test and confirm it fails because the package is absent.
- [ ] Add the minimal P2 planner subclass, prompt, restricted dispatch, and
      explicit-quaternion primitive wrapper.
- [ ] Run the focused tests and confirm they pass.

### Task 2: Add the evaluation adapter

**Files:**
- Create: `policy/RoboDojo_Agent_P2_RPent/model.py`
- Create: `policy/RoboDojo_Agent_P2_RPent/deploy.py`
- Create: `policy/RoboDojo_Agent_P2_RPent/deploy.yml`
- Create: `policy/RoboDojo_Agent_P2_RPent/{eval.sh,install.sh,setup_eval_policy_server.sh,setup_eval_env_client.sh}`
- Create: `policy/RoboDojo_Agent_P2_RPent/README.md`

**Interfaces:**
- Produces: standard XPolicyLab model and evaluation entry points.
- Consumes: `P2Planner` and P2 primitives from Task 1.

- [ ] Add a failing test proving debug mode cannot fall back to Pi_05.
- [ ] Implement the no-action model, planner-owned deploy loop, configuration,
      and scripts.
- [ ] Run Python compile and shell syntax checks.
- [ ] Run the focused unit tests.

### Task 3: Exercise `general_pickup`

**Files:**
- Verify: `policy/RoboDojo_Agent_P2_RPent/README.md`

**Interfaces:**
- Consumes: the complete adapter.
- Produces: startup evidence and, when simulator/LLM credentials are available,
  one official `general_pickup` result.

- [ ] Run the debug evaluation command for `general_pickup`.
- [ ] Run the adapter and inherited P1-RPent unit suites.
- [ ] Record any missing simulator or planner credential as an unverified
      external dependency rather than replacing P2 with Pi_05.
- [ ] Commit and push the verified branch to `zwbx/XPolicyLab`.
