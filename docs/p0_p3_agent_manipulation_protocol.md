# P0–P3 Agentic Manipulation Protocol

Date: 2026-08-27

This note freezes the comparison we agreed to run on RoboDojo. It is not a
leaderboard submission recipe. P0 is the already-measured official-protocol
baseline. P1–P2 are the main scientific comparison.

**The axis is how close control sits to the LLM.** P0 has no LLM in the loop at
all. Each step up hands the model more of the control problem and takes away
one more non-learned or pretrained layer between it and the robot:

| | Who emits the action | What stands between the model and the robot |
| --- | --- | --- |
| P0 | Frozen VLA | Everything; there is no LLM |
| P1 | Frozen VLA | The LLM aims the arm, then hands off; it never emits an action |
| P2 | Our primitives | A hand-written primitive vocabulary (`move_to`, `set_gripper`; no grasp macro) |
| P3 | The LLM | Nothing but action decoding |

P3 is the endpoint of that axis, not an extra-data appendix. Retraining a VLA
on subgoal text used to be called P3; it is a different question and now lives
in §7 as the retrain extension, because it moves control *away* from the LLM
and back into learned weights.

Related design notes: [physical_policy_harness_design.md](physical_policy_harness_design.md)
(harness vs high-rate policy). Official seed-0 numbers live in
`experiments/robodojo-official-2026-08-25/`.

## 1. Question

On RoboDojo long-horizon / Open failures, is the missing piece

1. a frozen-VLA outer loop (which object, when to hand off, retry), or
2. a callable non-learned executor for the whole pick–place?

The first P1 bet is **not** rewriting the official instruction into short
subgoals. It is **gaze priming**: keep the official long instruction, move the
end-effector above the object that should be used next, then call frozen
Pi_05. Retraining on subgoal text is the §7 extension and is reported
separately.

P3 asks the third form of the question: can a frontier LLM be the policy
outright, with no VLA and no primitive vocabulary? A P3 number that beats P2
says the primitive layer was itself the bottleneck. A P3 number near zero while
P2 is high says the LLM can plan manipulation but cannot output the actions.

## 2. Locked shared settings

These stay fixed across P0–P2 unless a condition is explicitly marked
non-leaderboard.

- Renderer and cameras: current valid sweep (albedo on, `ROBODOJO_UNTILED_CAMERAS=1`).
  Invalid trees `eval_result_pre_renderfix/` and `eval_result_pre_untiledfix/` are
  not baselines.
- Pi_05 checkpoint for P0/P1: `RoboDojo-sim-arx_x5-joint-0`, **weights frozen**.
- Success: only RoboDojo `_result.json` / official reward. Agent-side `verify`
  is for control, not scoring.
- P0/P1 observations: RGB + proprioception + language. No object labels, layout
  JSON, or reward-script answers.
- Do not attach P1–P3 jobs to the live G05 elastic scheduler or write into the
  official `eval_result/` tree under the `Pi_05` policy name. Use a separate
  `policy_name` and/or eval root.
- Official training HDF5 has a scalar episode `instruction`, not frame-level
  subtask text ([docs](https://robodojo-benchmark.com/doc/usage/install-and-download/),
  paper Appendix I). Open tasks including `classify_objects_by_language` are
  eval-only (no official demos).

## 3. Conditions

### P0 — Bare VLA (anchor)

Official `gen_instruction`, standard `update_obs` / `get_action`, no agent.

Seed-0 Pi_05 (untiled, complete 42/42): Average SR **10.12%** vs official 6.91
(+3.21). Micro 213/2100. Useful cells:

- `classify_objects` 9/50 (Long-Horizon, teleop train+eval)
- `classify_objects_by_language` 0/50 (Open, eval-only)
- Open dimension SR 2.0%

### P1 — Frozen VLA + shell (P1-gaze first)

Same Pi_05 weights. Official `instruction` is **unchanged** for the VLA.

**P1-gaze (main column).** Hypothesis: Pi_05 already knows grasp/place if the
wrist is already looking at the right object; it fails at *which* object to
approach under a long Open instruction. The shell therefore:

1. reads head RGB (wrist RGB is for later verify, not the global map);
2. detects 2D boxes from vision + language (no GT category labels);
3. lifts those pixels into a **shared table frame** (Section 3.1);
4. calls a thin Cartesian `move_ee` to hover above the chosen $(x, y)$;
5. hands control to frozen Pi_05 with the original episode instruction;
6. **re-primes before every Pi_05 chunk** so multi-object tasks can switch
   targets as objects leave the table or enter baskets (locator prompt: NEXT
   table object, not FIRST at episode start only).

Contact (close gripper, lift, place) stays on Pi_05. The pre-move is a
deliberate P2-thin primitive; P1-gaze is therefore not “pure language shell”.
Report it as such.

**P1-text (ablation only).** Rewrite the instruction into a short same-style
sentence, still no joints. Run only after P1-gaze, so a gain is not confused
with “the VLA just needed a shorter prompt”.

If action chunks are shortened for handoff, also run a **P0-K** cell (bare
Pi_05, same $K$) so P1 is not confounded with 50-step vs 10-step chunks.

### P2 — Agent + non-learned executor

No VLA. Agent plus self-built primitives, first via **path A**: policy process
owns IK / Cartesian interpolation; environment still only sees `ee` or `joint`
over the standard websocket.

Minimum primitives (implemented by us, not shipped in `RoboDojo-eval`):

- `move_to(arm, xyz, quat)` — one Cartesian pose, with an explicit orientation
- `set_gripper(arm, width)`
- `return_home(arm)`

The built P2 (`policy/RoboDojo_Agent_P2_RPent/`) deliberately stops there and
ships **no** `pick` / `place` / pregrasp macro, so the agent has to compose
open, hover, descend, close, and lift itself. A macro would hide exactly the
step where the interesting failures live: an RGB-D surface point is not an
end-effector target, and the arx_x5 flange sits about 0.145 m above the
fingertips. Every target is measured from the depth-backed world map; a replayed
coordinate is not a P2 result.

`RoboDojo-eval` has cuRobo in `env/planner_manager/curobo_planner.py`, but it
lives in the **simulator process**. Exposing it is path C (better motion, worse
protocol comparability, easier pose leakage). Path B is the same self-written
primitives as A, with Mem_0-style long RPCs instead of per-step `get_action`.
Main table starts at A.

Perception must be labeled:

- **Main table:** RGB + optional learned detectors; no GT category labels.
- **Privileged upper bound (not on the main table):** GT or oracle 6D pose
  without category names, or later path C.

Do not ship task-named tools (`solve_classify_objects`).

### 3.1 Shared 2D→3D table map (used by P1-gaze, reusable in P2)

RoboDojo tabletop tasks share approximately the same fixture: fixed robot base,
fixed head camera, objects on one horizontal plane. Object *instances* move
every episode; the **camera–table geometry does not**. One mapper therefore
serves all table tasks. It is not a per-task script and not a learned latent
space.

**Inputs (from the live observation, not from reward scripts):**

- Head RGB, already decoded RGB by the server.
- `intrinsic_matrix` $K$ and `extrinsics_matrix` $T_{\mathrm{cam}\leftarrow\mathrm{base}}$
  (runtime name; HDF5 uses `extrinsic_matrix`).
- A table plane in the robot base: $n^\top x + d = 0$. First cut: horizontal
  $z = z_{\mathrm{table}}$ for `arx_x5`, with $z_{\mathrm{table}}$ fit once
  from a few table pixels or a known reach, then frozen.

**Map (pinhole ray ∩ table plane):**

1. Pixel $(u, v)$ → ray in camera: $K^{-1}[u, v, 1]^\top$.
2. Transform the ray into the robot base with $T$.
3. Intersect the ray with the table plane → $(x, y, z_{\mathrm{table}})$.
4. Hover target: $(x, y, z_{\mathrm{table}} + h)$ with a downward EE
   orientation. $h$ is a fixed clearance (e.g. 8–12 cm), not a per-task height.

If a frame is missing extrinsics, fall back to a **once-calibrated homography**
head-image ↔ table $(x, y)$, still shared across tasks. Do not fit a new $H$
per task.

**What language does:** bind the current step of the official instruction to 2D
boxes — both the object to pick and, where the instruction names one, the
container to place it in. Language does not output Cartesian setpoints; the
shared mapper does.

For `classify_objects_by_language` the container half is not optional. Its
reward fixes which category belongs in which basket, while `classify_objects`
accepts any consistent permutation; frozen Pi_05 scores 9/50 on the latter and
0/50 on the former. Grounding only the pick target therefore cannot close that
gap, because a correctly picked object still lands in an arbitrary basket.

**What this is not:**

- Not GT object names or layout JSON.
- Not a wrist-camera world map. Wrists move; they can confirm “object large in
  view” after hover, they do not define the shared frame.
- Not full 6D pose. Orientation for hover is a default; Pi_05 still does the
  grasp.

**Handoff:** hover is done when EE $xy$ is within a few centimeters of the
mapped point and $z$ is in the hover band (or a timeout). Then `get_action`
from Pi_05 for one action chunk; do not keep overriding joints during that
chunk. When the chunk finishes, repeat steps 1–4 for the next target before the
next chunk (`P1_REPRIME=1`, default). In `release` mode, a real open-after-close
cycle re-primes immediately once the gripper stayed closed for at least
`P1_GRASP_MIN_STEPS` (default 15); short failed grasps are throttled by
`P1_REPRIME_MIN_STEPS` (default 80). The live implementation measures EE
position after the minimum dwell and continues IK actions until the error is at
most `P1_HOVER_TOLERANCE_M` (default 3 cm), bounded by
`P1_HOVER_MAX_STEPS` (default 40). Gripper state is observed after every
simulator action. If a close→open object-release transition occurs inside a
Pi_05 chunk, the unused tail of that chunk is discarded so the next target is
localized and hovered before a fresh chunk is requested.

**Executable boundary:** snapshot every action chunk immediately after
`get_action` / `get_action_batch`. Every action field sent during that chunk
must be byte-for-byte equivalent to the snapshot. P1 may stop and discard the
unused tail of a chunk, but it may not change arm joints, EE poses, grippers, or
any other action field. Guidance actions are allowed only between chunks and
must preserve the observed gripper channels. A condition that violates either
rule is P2, regardless of its adapter name.

### P3 — The LLM is the policy

No VLA and no primitive vocabulary. The LLM occupies exactly the slot a VLA
occupies: it receives an observation and returns an action chunk, and the
harness cannot tell from the contract which one it is talking to.

```
reset(scene) -> None
act(observation) -> ActionChunk        # a VLA fills this; at P3 the LLM does
```

The only thing between the model and the simulator is decoding its tool-call
arguments into the action dict the websocket already accepts
(`left_arm_joint_state` 6, `right_arm_joint_state` 6, `left_ee_joint_state` 1,
`right_ee_joint_state` 1, or the `ee_pose` variant). No IK, no interpolation,
no `pick`/`place`. If a condition needs one of those to work, it is P2.

**Native control surface.** P3 uses each provider's own API rather than a
lowest-common-denominator one, because the point is to measure the model as its
vendor exposes it: Anthropic's `/messages` with thinking blocks and
`cache_control`, OpenAI's `/responses` with reasoning items, and
`/chat/completions` for everything else. The action space is a tool schema, so
emitting an action and calling a tool are the same act for the model.

**Action chunking.** The model returns a sequence of actions executed open-loop,
the same handoff granularity a VLA chunk has. Chunk length is a reported
condition, not a fixed constant: a one-action chunk is closed-loop LLM control
at the simulator's rate, and a 50-action chunk matches Pi_05. Report which was
used; they are different conditions.

**What P3 does not get.** No GT object names, no layout JSON, no reward-script
answers — the same observation contract as P0/P1. Privileged-pose variants are
an upper bound, never the main table.

**Executable boundary.** Every action field sent to the environment must come
from the model's tool call or from the previous observation's state for the
channels the call left unspecified. A condition that clamps, retargets, or
interpolates the model's numbers is P2 wearing a P3 name. Approval that only
*rejects* an action (safety clamp that aborts rather than edits) stays P3, and
the rejection must be recorded.

## 4. First-cut tasks

Do not run 42 cells until the causal tests move.

1. `classify_objects_by_language` first for P1-gaze (eval-only Open; P0 is 0/50).
   Then `classify_objects` to see if the same mapper helps a trained clustering
   task. P0 already contrasts clustering vs named left/middle/right grounding.
2. `general_pickup` — best first closed loop for P2 (coarse pick, Open, eval-only).
3. Optional: `press_by_number`, `organize_table` — P1 memory and step splitting.

Start with `--eval-num 10` (or 5). Promote to native 50 only if the gap is
clear.

## 5. How to read the table

| Outcome | Conclusion |
| --- | --- |
| P1-gaze > P0, full P2 not much higher | Bottleneck is *which object* / approach; VLA can finish contact |
| P1-gaze ≈ P0, P2 clearly higher | VLA is not a callable skill even when aimed at the right object |
| P1-text > P1-gaze | The model also needed a shorter prompt, not only gaze |
| P3 > P2 | The primitive vocabulary was the ceiling, not the model |
| P3 ≈ 0 while P2 is high | The LLM can plan manipulation but cannot emit actions |
| P3 rises only at chunk length 1 | It is closed-loop correction, not open-loop control |
| All low | Contact / precision; the agent taxonomy does not answer it |

Do not use `push_T` / `insert_key` as the first proof of the agent paradigm.

## 6. Implementation notes (when we build)

- New adapters: e.g. `policy/Pi_05_agent/` (P1) and a separate P2 policy
  directory. `deploy.yml` `policy_name` equals the directory name.
- Decoding, checkpoint resolution, and action dims stay on shared helpers
  (`AGENTS.md`). No extra `COLOR_BGR2RGB`.
- P1-gaze: shared `TablePlaneMapper` + detector; `move_ee` only for hover;
  Pi_05 for contact. Eval names e.g. `Pi_05_Agent_P1A50`, never `Pi_05`.
- P1-text rewrite budget (ablation): cap agent calls per episode and rewrite
  length; keep original `step_lim`.
- Isolate eval output from the live G05 sweep on the 8× A800 host.
- Prefer a separate GPU/process for the detector VLM vs Pi_05 JAX.
- P3 lives in `policy/Agent_P3/` and holds no VLA, so it needs no
  `policy_uv_env_path` and no checkpoint. Its cost is API tokens, not GPUs, so
  it can run alongside a GPU sweep.

## 7. Retrain extension (was P3)

Segment demos and train on fine-grained instructions, then let the agent invoke
that VLA. Extra data and compute are reported. This is not a column on the
P0–P3 axis: it moves control back into learned weights rather than toward the
LLM, so it answers a different question and is reported on its own.

`classify_objects_by_language` has no official training set; any number there
must say whether it is transfer from `classify_objects` or newly labeled data.

Official Mem_0 adapter ships per-segment language for only three Mn tasks:
`cover_blocks`, `press_by_number`, `imitate_sorting_sequence`. That is not a
RoboDojo-wide subtask corpus.

## 8. Status

| Item | Status |
| --- | --- |
| P0 seed-0 Pi_05 | Done (10.12% Average SR) |
| P0 G05 / Xiaomi seed-0 | In progress / queued on the official sweep; do not steal GPUs |
| Shared 2D→3D table map spec | Locked (this note, §3.1) |
| P1 code | `policy/Pi_05_Agent_P1_RPent/` running on `arrange_largest_number` |
| P2 code | Built (`policy/RoboDojo_Agent_P2_RPent/`); official success on `general_pickup` fixed layout 0, `arrange_largest_number` unsolved |
| P3 framework | Built (`policy/Agent_P3/`); no eval numbers yet |
| Retrain extension | Not started |
