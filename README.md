<div align="center">

<h1>ManipLoop</h1>

<p><strong>Can an LLM close the loop on a robot arm?</strong></p>

<p>
<a href="docs/p0_p3_agent_manipulation_protocol.md">P0–P3 protocol</a> |
<a href="https://robodojo-benchmark.com/LeaderBoard">RoboDojo</a> |
<a href="docs/harness.md">Harness reference</a>
</p>

</div>

ManipLoop measures how well a language model can control a robot arm by making
it actually run one. The model drives a manipulator in closed loop inside
RoboDojo, and the only thing that counts as success is the benchmark's own
reward. There are no multiple-choice questions, no offline trajectory scoring,
and no partial credit invented on our side.

The reason to build this is that "can a VLM do manipulation?" is not one
question. A model can name the right object and still be unable to say where to
put the gripper; it can produce a plausible waypoint and still be unable to hold
a grasp. So ManipLoop does not report one number. It reports the same task
across **four levels of how much of the control problem the model owns**, and
the interesting result is the shape of the curve across those levels, not any
single cell.

## The P0–P3 axis

The axis is **how close control sits to the LLM**. P0 has no LLM at all. Each
step up hands the model more of the control problem and removes one more
learned or hand-written layer between it and the robot.

| | Who emits the action | What stands between the model and the robot | Implemented in |
| --- | --- | --- | --- |
| **P0** | Frozen VLA | Everything; there is no LLM | any adapter in `policy/`, run under the official protocol |
| **P1** | Frozen VLA | The LLM aims the arm, then hands off; it never emits an action | `policy/Pi_05_Agent_P1_RPent/` |
| **P2** | Our primitives | A hand-written primitive vocabulary the LLM composes | `policy/RoboDojo_Agent_P2_RPent/` |
| **P3** | The LLM | Nothing but action decoding | `policy/Agent_P3/` |

What makes the axis worth anything is that each level has an **executable
boundary** — a mechanical rule for whether a run really belongs to that level.
Without those rules every condition drifts upward into whichever level sounds
most impressive.

### P0 — bare VLA, the anchor

A frozen policy under the official protocol: official instruction, standard
`update_obs` / `get_action`, no agent in the loop. This is the number every
other level is compared against, and it is deliberately somebody else's
model — the point is a fixed reference, not a strong one.

### P1 — the LLM aims, the VLA acts

Same frozen VLA weights, same unchanged official instruction. The LLM reads the
scene, decides *which* object matters next, moves the end-effector above it, and
then hands control to the VLA for one action chunk. Contact — closing the
gripper, lifting, placing — stays with the VLA.

**Boundary:** snapshot the action chunk the moment the VLA returns it. Every
action field sent to the environment during that chunk must be byte-for-byte
the snapshot. P1 may stop early and discard the unused tail of a chunk, but it
may not edit an arm joint, an end-effector pose, or a gripper channel. Guidance
motions are allowed only *between* chunks. A run that edits actions inside a
chunk is P2 no matter what its adapter is called.

### P2 — the LLM composes primitives

No VLA anywhere. The LLM plans with measurement tools and executes with
non-learned primitives, and this implementation is deliberately stricter than
"a primitive vocabulary": there is no `pick`, no `place`, and no pregrasp
helper. The model gets explicit Cartesian `move_to` — which requires an explicit
quaternion, not a default — plus `set_gripper`, `return_home`, and RGB-D
measurement tools, so a grasp is something the model has to compose out of open,
hover, descend, close, and lift.

Every motion target must be *measured*. Depth-backed pixel-to-world sampling is
the only source of coordinates; replaying a coordinate that worked last episode
is not a P2 result. The gap that catches most planners here is that an object
surface point is not an end-effector target: the arx_x5 flange sits about
0.145 m above the fingertips, and a planner that commands the surface point
directly drives the fingers through the table.

### P3 — the LLM *is* the policy

No VLA and no primitive vocabulary. The LLM occupies exactly the slot a VLA
occupies, and the harness cannot tell from the contract which one it is talking
to:

```text
reset(scene) -> None
act(observation) -> ActionChunk        # a VLA fills this; at P3 the LLM does
```

The only thing between the model and the simulator is decoding its tool-call
arguments into the action dictionary the websocket already accepts. No inverse
kinematics, no interpolation, no `pick`/`place`. Action chunk length is a
reported condition rather than a constant: a one-action chunk is closed-loop
control at the simulator's rate, a fifty-action chunk matches a VLA's handoff
granularity, and they are different conditions.

**Boundary:** every action field must come from the model's tool call, or from
the previous observation for channels the call left unspecified. A condition
that clamps, retargets, or interpolates the model's numbers is P2 wearing a P3
name. A safety check that only *rejects* an action — aborting rather than
editing it — stays P3, and each rejection is recorded.

## What every level is denied

The conditions only mean something if the model is not fed the answer, so all
four see the same observation contract: RGB, proprioception, and the official
language instruction. No ground-truth object names, no layout JSON, no reward
script. Privileged variants (oracle 6D pose, simulator-side motion planning)
are reported as upper bounds and never on the main table.

Success comes only from RoboDojo's own reward and termination. The agent's own
verification exists to steer control, never to score, and the planner is
mechanically prevented from grading itself: a `finish(status="success")` that
the environment has not confirmed is refused while the episode is still live and
step budget remains.

## How to read the results

| Outcome | Conclusion |
| --- | --- |
| P1 > P0, P2 not much higher | The bottleneck was *which object* and how to approach it; the VLA can finish contact |
| P1 ≈ P0, P2 clearly higher | The VLA is not a callable skill even when aimed at the right object |
| P3 > P2 | The primitive vocabulary was the ceiling, not the model |
| P3 ≈ 0 while P2 is high | The LLM can plan manipulation but cannot emit actions |
| P3 rises only at chunk length 1 | It is closed-loop correction, not open-loop control |
| All low | The wall is contact and precision, and this taxonomy does not answer it |

## Status

Early. The framework for all four levels exists; the sweep does not.

| Level | State |
| --- | --- |
| P0 | Measured. Seed-0 π0.5 under the official protocol, 42/42 tasks: **10.12%** average SR (`experiments/robodojo-official-2026-08-25/`) |
| P1 | Running. Official successes recorded on several RoboDojo tasks; no full sweep yet |
| P2 | Running. Official success on `general_pickup` fixed layout 0; `arrange_largest_number` still unsolved |
| P3 | Built, no eval numbers yet |

One early finding worth stating, because it shapes what the numbers will mean:
at P2 the planner's failures are rarely about *understanding the task*. They are
about arithmetic on measured coordinates — lifting 0.089 m when the success
threshold is 0.1 m, or commanding a surface point as a flange target. Model
scale shows up here more than prompt wording does: a local 4B planner tends to
re-emit the same tool call until the step budget is gone, which is why the
dispatch layer refuses byte-identical repeats outright rather than trying to
talk the model out of them.

## Running a condition

Every condition is a normal adapter, so it runs through the standard harness
entry point ([reference](docs/harness.md)):

```bash
cd policy/<CONDITION>
bash eval.sh RoboDojo <task> <ckpt_name> arx_x5 joint <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_env_or_uv_path> <eval_env_conda_env>
```

For agent conditions, a fixed layout is the useful development loop, because a
single deterministic scene makes a prompt or primitive change legible:

```bash
ROBODOJO_RUN_ID=<run-id> \
  bash policy/RoboDojo_Agent_P2_RPent/run_fixed_layout.sh \
  <layout_id> <policy_gpu_id> <env_gpu_id> /path/to/RoboDojo-eval/.venv <task>
```

Each condition's README carries its own requirements — planner backend and keys,
camera and depth settings, and what its tool surface deliberately excludes.

## Layout

```text
policy/Pi_05_Agent_P1_RPent/   P1: LLM aims, frozen π0.5 acts
policy/RoboDojo_Agent_P2_RPent/  P2: LLM composes atomic Cartesian primitives
policy/Agent_P3/               P3: LLM in the policy slot
docs/p0_p3_agent_manipulation_protocol.md   the protocol these conditions obey
docs/harness.md                the inherited XPolicyLab harness reference
experiments/                   official run records and result JSON
policy/<41 others>/            upstream XPolicyLab policy adapters
```

## Built on XPolicyLab

ManipLoop is a derivative work of
[XPolicyLab](https://github.com/XPolicyLab/XPolicyLab)
([website](https://xpolicylab.github.io/), [arXiv:2608.09892](https://arxiv.org/abs/2608.09892)),
a collaborative open-source project led by MMLab@HKU and THU, and it is
distributed under the same Apache-2.0 licence. The policy-serving harness, the
adapter contract, the standard data formats, and the 41 policy adapters under
`policy/` are XPolicyLab's work, not ManipLoop's — see
[docs/harness.md](docs/harness.md) for that documentation, kept verbatim. The
Python package is still imported as `XPolicyLab` for the same reason.

What ManipLoop adds is the P0–P3 axis and the agent conditions that implement
it: the P1, P2, and P3 adapters, the protocol in
[docs/p0_p3_agent_manipulation_protocol.md](docs/p0_p3_agent_manipulation_protocol.md),
and the run records under `experiments/`.

If the harness or the policy zoo is what helps your work, cite XPolicyLab:

```bibtex
@article{community2026xpolicylab,
  title={{XPolicyLab}: A Unified Standard and Open Ecosystem for Robot Policy Evaluation and Deployment},
  author={Community, XPolicyLab and Chen, Tianxing and Chen, Yue and Nian, Tian and Cai, Zijian and Chen, Guangyu and Lin, Wenwei and Liang, Qiwei and Xiang, Peicheng and Su, Kailun and others},
  journal={arXiv preprint arXiv:2608.09892},
  year={2026}
}
```

Adding a policy adapter, or entering the RoboDojo and RoboTwin leaderboards,
goes through upstream XPolicyLab rather than this repository.
