# RoboDojo simulator evaluation, 2026-08-25

Closed-loop RoboDojo simulation runs for the three released checkpoints, on 8x NVIDIA
A800-SXM4-80GB. This supersedes `experiments/robodojo-baseline-2026-08-23/`, which only
recorded single forward passes and never entered the simulator.

## Status

| Policy | Checkpoint dir | Action type | Seed 0 | Seeds 1-2 |
| --- | --- | --- | --- | --- |
| Pi_05 | `RoboDojo-sim-arx_x5-joint-0` | joint | running (`2026-08-25_07-41-06_smoke`; Traj retry in parallel) | not started |
| G05 | `RoboDojo-sim-arx_x5-joint-0` | joint | queued after Pi_05 on GPUs 2–7 while Traj still uses 0/1 (`scripts/chain_robodojo_remaining.sh`) | not started |
| Xiaomi_Robotics_1 | `RoboDojo-sim-arx_x5-ee-0` | ee | queued after G05 | not started |

Pi_05 seed 0 snapshot (partial): see `results/pi05-seed0-partial.json` (18/42 reported
cells, 900 episodes, SR 0.00% vs official 6.91% — not a reproduction verdict). A
reported-task cell is counted only at 50 episodes.

`imitate_sorting_sequence`, `make_kong`, and `play_tic_tac_toe` failed in the first
sweep because `Assets/Traj` was still an LFS pointer. Files are on disk now. Closed-loop
retry: imitate on GPU 1 (videos under `eval_result/.../imitate_sorting_sequence/.../_stream/`),
`make_kong` on idle GPU 0, `play_tic_tac_toe` starts on GPU 1 after imitate exits
(`scripts/coord_pi05_traj_retry.sh`). `robodojo.sh eval --eval-num native` does not
export `EVAL_NUM`; the Isaac client still reads 50 episodes from `_task.yml` /
`process_config` when the env var is unset, which is the official standalone budget.

The latest `_result.json` files currently contain one closed-loop success
(`match_and_pick_from_conveyor` 1/10 while that task is still running). Most finished
cells are 0/50, which is still compatible with an overall 6.91% once the remaining
tasks fill in. Camera ffmpeg streams are writing for imitate and make_kong.

G05 and Xiaomi closed-loop sweeps have not started; their policy servers do load on this
host (`results/g05-forward-gpu1.json`, `results/xiaomi-forward-gpu1.json`, both `"status": "passed"`).

The per-task logs and videos stay in the RoboDojo checkout under `smoke_results/<run_id>/` and `eval_result/`, which are far too large to commit. Scheduler stdout is `logs/<policy>-seed<seed>.log`.

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
