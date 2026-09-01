# RPent Minimal Prompt Adaptation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the upstream RPent RoboTwin planning contract with only the substitutions required for RoboDojo and frozen Pi_05.

**Architecture:** Keep RPent's planner/tool separation: head-view grounding binds target identity and reports same-step surface geometry; the planner adds EEF/TCP clearance before `move_to`; wrist views refine geometry only through `sample_world_xyz` or `query_world_map`. Build the default prompt from the archived upstream sections so future edits cannot silently replace the RPent strategy with a shorter custom prompt.

**Tech Stack:** Python, pytest, OpenAI-compatible function calling.

## Global Constraints

- Keep images RGB end to end.
- Preserve the full upstream RPent strategy wherever RoboDojo and Pi_05 do not require a substitution.
- `ground` remains head-only and must not compute an EEF hover target.
- Wrist geometry uses the exact step, view, and pixel coordinate space.

---

### Task 1: Lock the adapted prompt and grounding contracts

**Files:**
- Modify: `tests/test_pi05_agent_p1_rpent.py`

**Interfaces:**
- Consumes: `SYSTEM_PROMPT`, `TOOLS_SPEC`, `RpentPrimitives.ground`.
- Produces: regression tests for upstream strategy preservation, wrist-tool guidance, and surface-only grounding.

- [ ] Add a test asserting that the default prompt retains upstream sections and explicitly routes wrist refinement to `sample_world_xyz` / `query_world_map`, while removing RoboTwin/LingBot names.
- [ ] Add a test asserting that the `ground` schema has no `clearance` parameter and its result has no `suggested_hover_xyz`.
- [ ] Run the focused tests and confirm they fail for the missing behavior.

### Task 2: Restore RPent's prompt with minimal substitutions

**Files:**
- Modify: `policy/Pi_05_Agent_P1_RPent/prompt_versions.py`
- Modify: `policy/Pi_05_Agent_P1_RPent/planner.py`
- Modify: `policy/Pi_05_Agent_P1_RPent/guides/GUIDE_RPENT.md`

**Interfaces:**
- Produces: `rpent_v1_system_prompt(task_name: str) -> str` and `rpent_v1_user_prompt(...) -> str`.

- [ ] Derive v1 from the archived v0 section structure.
- [ ] Substitute RoboDojo, Pi_05, local resource access, and the actual `pi05_act` horizon contract only where required.
- [ ] Restore the upstream guide's geometry, wrist, motion-failure, verification, and budget rules with the same minimal substitutions.
- [ ] Run the prompt regression tests and confirm they pass.

### Task 3: Make grounding surface-only

**Files:**
- Modify: `policy/Pi_05_Agent_P1_RPent/planner.py`
- Modify: `policy/Pi_05_Agent_P1_RPent/tools.py`
- Modify: `policy/Pi_05_Agent_P1_RPent/README.md`

**Interfaces:**
- `ground(query: str, camera: str = "head", anchor: str = "center") -> dict`.

- [ ] Remove `clearance` from the tool schema and dispatch.
- [ ] Remove `suggested_hover_xyz` from the result.
- [ ] Document that the planner must add EEF/TCP clearance and that wrist refinement uses the geometry tools.
- [ ] Run the focused test file.

### Task 4: Verify and submit

**Files:**
- Verify all modified policy and test files.

- [ ] Run `pytest -q tests/test_pi05_agent_p1_rpent.py`.
- [ ] Run shell syntax and Python compile checks for the adapter.
- [ ] Inspect the final diff for unrelated changes.
- [ ] Commit, push, and update the pull request.
