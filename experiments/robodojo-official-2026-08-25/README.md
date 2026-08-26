# RoboDojo simulator evaluation, 2026-08-25

Closed-loop RoboDojo simulation runs for the three released checkpoints, on 8x NVIDIA
A800-SXM4-80GB. This supersedes `experiments/robodojo-baseline-2026-08-23/`, which only
recorded single forward passes and never entered the simulator.

## Status

| Policy | Checkpoint dir | Action type | Seed 0 | Seeds 1-2 |
| --- | --- | --- | --- | --- |
| Pi_05 | `RoboDojo-sim-arx_x5-joint-0` | joint | valid rerun in progress. Official cells **1/42** (`build_tower` 29/50 = 58%, three-camera videos OK). In-progress 36/252 eps (14.29%). GPU7 moved to `fill_pen_holder`. | not started |
| G05 | `RoboDojo-sim-arx_x5-joint-0` | joint | queued behind Pi_05; adapter and venv issues already fixed | not started |
| Xiaomi_Robotics_1 | `RoboDojo-sim-arx_x5-ee-0` | ee | queued; checkpoint resolves; SDPA fallback verified offline (`results/xiaomi-sdpa-preflight.json`); closed-loop smoke still pending until a GPU frees | not started |

**No valid seed-0 table exists right now.** Snapshot at 19:35 CST: 8 Isaac clients with untiled cameras; wrist videos still colorful (`results/wrist-diagnosis/untiled-rerun-live-1930.json`). Early WS `TimeoutError`s during policy warmup recovered without stopping the sweep. Two completed Pi_05/G05 tables were
quarantined because their wrist-camera observations were blank; see below. The first
attempt ran 07:41-18:07 CST and
reached 41/42 cells for Pi_05 at **SR 1.07%** against an official 6.91%, plus six
partial G05 cells. All of it was rendered without material albedo and has been
discarded; see the next section. `compare_robodojo_to_official.py` correctly
reports no results until the re-run fills cells again.

### Second renderer defect: tiled wrist cameras were blind

The post-albedo-fix run completed 42/42 cells for both Pi_05 and G05, but produced
only 1.10% and 1.57% success. Those numbers are also invalid. In the exact observation
dict sent to the policy, `cam_head` had normal mean luma around 93 while both mounted
wrist views were effectively black (mean 1-3); one side was usually a literal all-zero
array. `EvalEnv._stream_vision` records the same dict returned to the policy, so this is
direct evidence of the model input rather than a video-only encoding problem.

The defect depends on the camera backend:

- RoboDojo's tiled path with 5 or 10 parallel envs: both wrist streams blank.
- One env: all three views normal.
- Five envs with one normal render product per camera: head/left/right mean luma
  100/98/124 on the sampled episode, all with visible texture; `stack_bowls` reached
  4/5 successes.
- Ten envs with normal render products exhausts RTX ParameterBlock resources, so the
  stable workaround is five envs.

This matches NVIDIA's confirmed Isaac Sim 5.1 tiled-camera bugs: multi-camera tiled RGB
can be black in RTX real-time mode and the renderer-side fix is in Isaac Sim 6.0
([IsaacSim #367](https://github.com/isaac-sim/IsaacSim/issues/367)). Path tracing,
raising `/rtx/viewTile/limit`, reducing from 10 to 5 while retaining tiled rendering,
and splitting policy/simulator GPUs did not restore RoboDojo's mounted wrist views.
The reproducible host workaround is installed by `scripts/robodojo_sim_env.sh`:
`ROBODOJO_UNTILED_CAMERAS=1` plus `ROBODOJO_NUM_ENVS=5`.

The invalid 42-cell tables and videos were moved to
`RoboDojo-eval/eval_result_pre_untiledfix/`. Diagnostic values and the side-by-side
camera image are under `results/wrist-diagnosis/`. The valid rerun started from an
empty `eval_result/` at 18:58 CST Aug 26.

What that discarded attempt still establishes, because it is a property of the
harness rather than of the renderer: the full 54-task protocol completes on this
host in about 10.5 h on 8 cards, the paired `X`/`X_random` merge and the
50-episode `--eval-num native` budget behave as `summarize_result.py` expects, and
both G05 blockers are genuinely fixed - (1) the official `.hydra/config.yaml`
points `hf_processor_path` at a trainer-host directory, so the sidecar remap now
prefers `run_dir/hf_processor/tokenizer.json`, and (2) G05's `.venv` lacked
`msgpack-numpy` and `pydantic` for the XPolicyLab websocket server. G05 reached
six tasks with closed-loop `_result.json` files before the stop, so it should not
need debugging again on the re-run.

`imitate_sorting_sequence`, `make_kong`, and `play_tic_tac_toe` failed on the very
first sweep because `Assets/Traj` was still an LFS pointer. Those files are on disk
now and all three ran closed-loop afterwards. `robodojo.sh eval --eval-num native`
does not export `EVAL_NUM`; the Isaac client still reads 50 episodes from
`_task.yml` / `process_config` when the env var is unset, which is the official
standalone budget.

### Why a task-level scheduler replaced the static chain

`smoke_all_tasks.sh` shards tasks once, up front, from embedded runtime weights.
Measured against those weights the shards diverged badly: by 14:00 CST the GPU0 and
GPU1 shards had already drained (both cards were busy only with the Traj retries)
while the GPU2 shard still had ~145 episodes and GPU6 ~180 — eight more hours on two
cards while six others emptied. The previous plan then waited for that straggler
before starting G05 at all.

`scripts/elastic_robodojo_scheduler.py` hands out one task at a time to whichever
card is actually idle, walking a priority list across all three policies. Two
properties make it safe to run next to a live sweep:

- A card counts as free only after several consecutive polls with no
  `eval_client/main.py --device_id <gpu>` process, which also rides out the 1–3
  minute reset gap between episode batches of one task.
- A task already running anywhere (matched on `--task_name` plus `--policy_name`) is
  never launched a second time.
- Tasks the live sweep has not yet `RUN` on an unfinished shard stay reserved, and
  those shard GPUs are not given elastic jobs between two of that shard's own
  tasks. After a shard's last `RUN`, the task stays reserved until its episode
  budget is full; otherwise a free card would launch a second Isaac client whose
  newer timestamp would hide the sweep's `_result.json`. That race happened once
  (GPU7 vs GPU4 on `push_T_random`, stamp `2026-08-25_16-16-14`); the duplicate
  tree was removed before it wrote a result.
- Idle detection matches `python -u src/eval_client/main.py --device_id N` only.
  A looser `eval_client` substring made GPU6 look busy after `stack_blocks`
  finished, because diagnostic `pgrep -f` / agent shells embed that string. The
  scheduler was restarted at 17:33 CST (pid in `/tmp/elastic-sched.pid`);
  existing Isaac clients were left running. GPU6 then took G05 `classify_objects`.

Completion is judged exactly as `summarize_result.py` judges it: the newest timestamp
directory must hold the task's full `_task.yml` budget. That comparison is
lexicographic, and the sweep's directories are pinned to its
`2026-08-25_07-41-06_smoke_<task>` run id, so a task the scheduler starts now sorts
above them and a later sweep re-run cannot demote a filled cell. Once every Pi_05
task is complete the scheduler releases sweep pid 216753, handing all eight cards to
G05 and then Xiaomi.

Superseded and removed: `chain_robodojo_remaining.sh`, `coord_pi05_traj_retry.sh`,
`coord_pi05_traj_play_gpu0.sh`, `retry_pi05_traj_tasks.sh`.

Most finished cells are 0/50, which is still compatible with an overall 6.91% once the
remaining tasks fill in. Traj retries have 30 ffmpeg streams each under `_stream/`.

G05 seed-0 closed-loop is already overlapping on six cards (see Status). Xiaomi is
still queued. Policy-server Forward JSON (`results/g05-forward-gpu1.json`,
`results/xiaomi-forward-gpu1.json`) only proves the servers load; it is not
closed-loop evidence.

The per-task logs and videos stay in the RoboDojo checkout under `smoke_results/<run_id>/` and `eval_result/`, which are far too large to commit. Scheduler stdout is `logs/<policy>-seed<seed>.log`.

## The seed-0 numbers above are invalid: rendering has no material albedo

Pi_05 seed 0 landed at **1.07%** against an official **6.91%**, and the gap is not
sampling noise. It is a rendering defect on this host, so every closed-loop number
recorded before it is fixed is measured against the wrong observation distribution
and has to be re-run.

Evidence in `results/render-diagnosis/`:

- `cam_head_train_vs_rendered.png` — official recorded frame beside ours for the
  same task. The mahogany table has no wood grain, the white bowls render black,
  and only the already-black-and-white robot looks unchanged.
- `render_gap.json` — the recorded frame has `mean_rgb` `[106.3, 81.7, 72.8]`
  (channel spread **33.5**); ours has `[48.6, 48.5, 48.7]` (spread **0.21**).
  Three identical channels means no albedo reached the renderer, and mean luma is
  half the recorded value with 27.6% of pixels near black.
- `pi05-openloop-stack_bowls-f122.json` — fed an official *recorded* frame, the
  same checkpoint predicts the next 50 joint targets at **MAE 0.0239 rad**, versus
  **0.5215** for a stay-still baseline. The adapter, checkpoint, norm stats, state
  packing, CHW image layout and 25 Hz control rate are therefore all correct; only
  the pixels the simulator produces are wrong.

This also explains the per-task pattern. `stack_bowls` still reaches 36% because
bowl silhouettes survive in greyscale, while every task that needs colour, texture
or glyphs is exactly 0: `classify_objects`, `*_by_language`, `solve_equation`,
`press_by_number`, `arrange_largest_number`.

### Cause and fix

Texture streaming. `--/rtx-transient/resourcemanager/enableTextureStreaming=false`
restores albedo, and it is now part of `ROBODOJO_KIT_ARGS` in
`scripts/robodojo_sim_env.sh`. On the mahogany smoke case the centre crop goes
from `[46, 46, 46]` (spread **0.00**) to `[110, 71, 56]` (spread **54.1**), which
is the brown the BaseColor texture actually contains.

The failure is silent, which is why it survived the earlier render bisect: MDL
materials compile, the textures are real files on disk (`Mahogany_Planks_BaseColor.png`
is 5.7 MB, not an LFS pointer), Kit logs nothing above a C302 implicit-conversion
warning, and `RENDER_OK` still prints. Only the pixels are wrong. The earlier
render smoke checked frame shape, not content, so it passed throughout.
`--/rtx/materialDb/syncLoads=true` `--/rtx/hydra/materialSyncLoads=true` with 16
subframes does not help, so this is not an async-load race.

`scripts/robodojo_render_smoke.py` now prints whole-frame and centre-crop
per-channel means, so this reproduces in ~20 s without a full episode.

### Verified in a real episode

A 2-episode `stack_bowls` run with the fix (stamp `2026-08-25_18-09-33`, 1/2
successes) put `cam_head` at `[120.5, 90.3, 78.4]`, channel spread **42.1**,
against the recorded `[106.3, 81.7, 72.8]` / **33.5** — same distribution, versus
`[48.6, 48.5, 48.7]` / **0.21** before. See
`results/render-diagnosis/cam_head_train_before_after.png` (recorded, before,
after) and `render_fix_verified.json`.

The 8-card re-run itself was spot-checked at 18:53 CST on a live
`imitate_sorting_sequence` fail video: `mean_rgb` `[111.8, 87.0, 77.7]`,
channel spread **34.1** (`results/render-diagnosis/rerun_live_color.json`).
First post-fix `_result.json` is that task's first 10/50 batch (0 success so
far; not yet a reported cell).

### Consequence: everything before the fix was discarded

All eight cards were stopped at 18:07 CST — sweep pid 216753, the elastic
scheduler, and every Isaac client and policy server. Pi_05's 41/42 table and
G05's six partial cells were measured through the grey renderer, so they are void
as reproduction evidence.

Those 66 stamp directories (7.1 GB) were moved out of `eval_result/` to
`eval_result_pre_renderfix/` in the RoboDojo checkout. Leaving them in place was
not safe: `compare_robodojo_to_official.py` reads the newest stamp per task, so
any task whose re-run failed would silently fall back to its grey-renderer result
and report a filled cell. `compare` now reports no results at all, which is the
correct starting point.

Restarted at 18:33 CST with the fixed renderer, both on all eight cards:

| | pid file | log |
| --- | --- | --- |
| Pi_05 seed-0 sweep | `/tmp/pi05-sweep-renderfix.pid` (2700145) | `logs/Pi_05-seed0-renderfix.log` |
| Elastic scheduler | `/tmp/elastic-sched.pid` | `logs/elastic-scheduler.log` |

Pre-fix logs are kept as `logs/Pi_05-seed0.log` and
`logs/elastic-scheduler-prerenderfix.log`. `logs/Pi_05-seed0-renderfix-badshard.log`
is a discarded false start: `--policy-gpu-ids 0-7` is not parsed as a range, so it
serialised all 54 tasks onto one worker. The working form is
`--policy-gpu-ids 0,1,2,3,4,5,6,7`.

## Official numbers to reproduce

Overall simulation success rate from the RoboDojo leaderboard:

| Policy | Official SR |
| --- | ---: |
| G0.5 (`G05`) | 14.88% |
| Xiaomi-Robotics-1 | 13.93% |
| π0.5 (`Pi_05`) | 6.91% |

The paper additionally breaks π0.5 down by capability dimension as `score / success`:
Generalization 13.37 / 8.17%, Memory 12.40 / 5.50%, Long-Horizon 23.54 / 14.67%,
Precision 5.78 / 4.56%, Open 1.98 / 1.67%, overall 11.41 / 6.91%. G0.5 and
Xiaomi-Robotics-1 are later leaderboard entries submitted by their own labs, so only the
overall number is published for them.

## Protocol

Taken from `scripts/internal/summarize_result.py` in the RoboDojo checkout, which defines
what the leaderboard table counts:

- 54 runnable tasks, reported as 42 after each `X` / `X_random` pair is merged.
- 50 episodes per reported task: 50 for a standalone task, 25 + 25 for a paired one.
  `--eval-num native` reads those per-task counts from `task/RoboDojo/config/_task.yml`.
- Seeds 0, 1, 2. A summary cell is filled only when its episode count is complete, so
  partial runs show up as blank cells rather than as low scores.
- `success_rate = successes / count * 100`, `score = sum(scores) / count * 100`.

## Reproduce

```bash
# One-time host setup plus the full protocol for one seed across all three policies.
bash scripts/run_robodojo_official_protocol.sh \
  --seeds 0 --policies Pi_05,G05,Xiaomi_Robotics_1 --gpu-ids 0,1,2,3,4,5,6,7
```

Single task, for debugging:

```bash
bash scripts/run_robodojo_sim_eval.sh eval Pi_05 \
  --task stack_bowls --eval-num 1 --seed 0 --policy-gpu 0 --env-gpu 1
```

`scripts/run_robodojo_sim_eval.sh` sources `scripts/robodojo_sim_env.sh`, which applies the
host fixes described below, then calls `scripts/robodojo.sh` in the RoboDojo checkout.

## Host fixes this needed

Isaac Sim crashed before rendering a single frame on this machine. Three unrelated causes,
all of which had to be fixed; see `scripts/robodojo_sim_env.sh` for the code.

**No NVIDIA Vulkan ICD was registered.** `/usr/share/vulkan/icd.d/` shipped only Intel, AMD
and lavapipe manifests, so Kit reported "No device could be created" and ran with no GPU. On
this driver build the Vulkan entry points are in `libEGL_nvidia.so.0`; `libGLX_nvidia.so.0`,
which NVIDIA's own packaging normally points at, enumerates zero devices here.

**`libcuda.so.1` resolved to the CUDA forward-compatibility driver.** The kernel module is
535.261.03, but `/lib/x86_64-linux-gnu/libcuda.so.1` had been symlinked to
`libcuda.so.590.48.01` from `/usr/local/cuda-13.1/compat`. Forward compatibility covers
compute only; the Vulkan and EGL stack stays at 535. Isaac Sim's renderer shares Vulkan
images and semaphores with CUDA, and a 590 CUDA driver cannot import handles produced by a
535 Vulkan driver. The visible symptom was a cascade starting at `NVTT block-compression
failed` and `cudaErrorIllegalAddress` on the first texture upload; that poisons the CUDA
context, so everything afterwards fails too, including `vkCreateRayTracingPipelinesKHR`.

That last error is worth calling out because it is actively misleading. It reads as "this
GPU has no ray tracing hardware", which matches the common belief that A100/A800 cannot run
Isaac Sim. It is not the cause here: `vulkaninfo` shows A800 exposing
`VK_KHR_ray_tracing_pipeline`, and the renderer works once the driver mismatch is gone.
Only the simulator is redirected to the stock driver, through a conda `activate.d` hook, so
policy servers keep the CUDA runtime they were installed against.

**DLSS crashed at renderer startup.** Kit probes `NVSDK_NGX_VULKAN_Init_Ext2`
unconditionally and the bundled NGX build segfaults inside `libnvidia-ptxjitcompiler`
against this driver. `--/ngx/enabled=false` skips the probe. Anti-aliasing and render mode
are left at their defaults so output stays as close as possible to an official run; the only
other Kit override is `--/rtx/verifyDriverVersion/enabled=false`, since Kit rejects
535.261.03 outright.

`scripts/robodojo_render_smoke.py` reproduces the whole render path in about 20 seconds and
prints `RENDER_OK`, which is what made bisecting these three practical.

## Assets

`scripts/fetch_robodojo_assets.py` downloads the ~40 GB arx_x5 asset set. `git lfs pull`
holds about 0.9 MB/s against this Hub through the proxy and raising
`lfs.concurrenttransfers` does not move the aggregate; plain parallel HTTPS reaches
13 MB/s. Every file is verified against the SHA-256 in its LFS pointer before being moved
into place, so the result is identical to what `git lfs pull` would have produced.

Missing assets do not fail loudly. `stack_bowls_random` died with
`TypeError: 'NoneType' object does not support item assignment` in
`env/scene_manager/layout_manager.py`, which is `load_object_metadata` returning `None` for
an object whose metadata JSON was still an LFS pointer.

## Throughput

One worker per GPU, with the policy server and Isaac Sim sharing a card. Two episodes of
`stack_bowls` took 762s co-located versus 786s split across two GPUs, so co-location is free
here: the policy server is idle while the simulator renders. That halves wall-clock time
against the two-GPU-per-worker layout.

Using the scheduler's embedded per-task runtime weights, one seed of one policy is about
47 hours of serial work, or 5.9 hours across 8 workers. All three policies at one seed is
roughly 17.6 hours, and the complete three-seed protocol about 53 hours.
