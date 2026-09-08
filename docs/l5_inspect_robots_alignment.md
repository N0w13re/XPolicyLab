# L5 vs inspect-robots-agent

Protocol source: [`robocurve/inspect-robots`](https://github.com/robocurve/inspect-robots)
plugin `plugins/inspect-robots-agent` (cloned for this write-up at the `main`
tip of 2026-09-08). RoboProbe keeps the RoboDojo / XPolicyLab eval harness and
the frozen name `Agent_P3`. Alignment is the **LLM-as-policy strategy**, not a
port of `inspect_robots.eval()`.

## inspect-robots-agent protocol (what we are matching)

The policy slot is still `reset()` / `act(observation) -> ActionChunk`. Inside
`act` the model is **not** a VLA: it speaks tools, and a motion layer turns one
validated call into an open-loop interpolant.

| Piece | inspect-robots-agent default |
| --- | --- |
| System prompt | `_SYSTEM_TEMPLATE`: named embodiment, small motions, required `note`, one tool call, `done` / `give_up` + hindsight, `max_llm_calls` budget |
| Observation | `Current observation.` + `Instruction:` + labeled `state[key]: dim=value` + camera PNGs (`camera 'name':`) |
| Tools | Absolute joint: `move_joints(targets, note)`; pose: `move_to`; plus `done(summary, hindsight)`, `give_up(reason, hindsight)` |
| Chunking | Interpolate from current proprioception at `max_speed_frac=0.1`, 5% range/step ceiling, 10 s cap, 10 Hz fallback |
| History | Full chat + tool results; `image_horizon=2` stubs older frames |
| Repair | Structured tool-result errors; 3 consecutive failures abort |
| Stop | `done` / `give_up` / budget-forced `give_up`; scorer judges success |
| Images | `always` (default). `on_demand` + `take_pic` is a separate mode |
| Report | `EvalLog` (llm usage, transcript, hindsight, config) |

Task set and seeds in inspect are `Task.scenes` with `init_seed`. RoboDojo
keeps its own layouts and official success flag; we do not replace that
dataset.

## Strategy table (after this change)

| Item | inspect-robots-agent | Agent_P3 now | Test |
| --- | --- | --- | --- |
| Tools `move_joints` / `done` / `give_up` | yes | yes | `test_the_default_tools_are_move_joints_done_and_give_up` |
| Required `note` on moves | yes | yes | schema `required` includes `note` |
| Named partial targets | yes | yes (`left_j0` …) | `test_the_request_labels_state_with_inspect_dimension_names` |
| Speed-limited interpolation | 0.1 / 5% / 10 s | same constants in `motion.py` | `test_interpolation_matches_inspect_speed_limits` |
| Labeled proprioception | `dim=value` | `state[joints]: left_j0=…` | same |
| Images every observation | default `always` | always | `test_every_request_carries_the_current_camera_images` |
| `image_horizon=2` | default | `P3_IMAGE_HORIZON` default 2 | `test_image_horizon_elides_older_camera_frames` |
| Tool-result repair | yes | yes | `test_a_malformed_call_comes_back_as_a_tool_result_for_repair` |
| 3-strike abort | yes | yes | `test_three_consecutive_tool_failures_end_the_turn` |
| `max_llm_calls=100` + forced `give_up` | yes | `P3_MAX_LLM_CALLS` | `test_llm_call_budget_forces_give_up` |
| `done` / `give_up` + hindsight | yes | yes | `test_done_carries_hindsight_and_stops_the_trial` |
| Embodiment notes in system prompt | `EmbodimentInfo.docs` | `EMBODIMENT_NOTES` for ARX X5 | default-tools test |
| System prompt wording | `_SYSTEM_TEMPLATE` | copied `SYSTEM_TEMPLATE` | default-tools test |
| `prior_learnings` file | `-P prior_learnings=` | `P3_PRIOR_LEARNINGS` | constructor appends file text |
| Official success | inspect scorer | RoboDojo `success` (unchanged) | harness still records fail on early stop |
| Eval runner | `inspect-robots eval()` | `eval.sh` / RoboDojo | scripts added; not a clone of Inspect |

## Intentionally not cloned (not the default agent condition)

These exist in inspect-robots but are **not** required for the default
`--policy agent` / `images=always` strategy. Shipping them would be extra
surface, not a protocol mismatch on the default path.

| Item | Why it stays out |
| --- | --- |
| `images=on_demand` + `take_pic` | Optional inspect mode; default is `always` |
| Depth renders | Needs embodiment depth; RoboDojo RGB eval does not |
| `wire=responses` / `gemini-live` / `interactions` | Extra vendors; we keep chat / messages / azure-chat |
| Core `DeltaLimitApprover` rewriting chunks | Inspect core, not the agent plugin. We clip interpolants to declared bounds the same way the plugin does |
| `EvalLog` / Rerun UI | Different product; transcript JSON + RoboDojo mp4 remain |
| Inspect scene/seed objects | RoboDojo layouts stay the task set |

A **raw 14-D `act` dump with no interpolation** is a different ablation. It is
no longer the L5 default, because it is not what inspect-robots-agent measures.
