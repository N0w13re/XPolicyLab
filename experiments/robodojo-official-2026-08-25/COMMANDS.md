# Reproducible commands for this run

Host: 8x NVIDIA A800-SXM4-80GB, driver 535.261.03. Policy servers and Isaac Sim
share a GPU. Apply the host fixes first; without them Kit never renders a frame.

```bash
export ROBODOJO_ROOT=/mnt/bn/robotics-data-mx/wenbo/RoboDojo-eval
source /mnt/bn/robotics-data-mx/wenbo/XPolicyLab/scripts/robodojo_sim_env.sh "$ROBODOJO_ROOT"
# installs ROBODOJO_KIT_ARGS (texture streaming off), ROBODOJO_UNTILED_CAMERAS=1,
# and ROBODOJO_NUM_ENVS=5 — required for valid wrist-camera RGB on Isaac Sim 5.1
```

Do **not** raise `ROBODOJO_NUM_ENVS` to 10 on this host. Isaac Sim is still 5.1.0;
NVIDIA [IsaacSim #367](https://github.com/isaac-sim/IsaacSim/issues/367) only
fixes tiled RTX RGB in 6.0. Untiled 10-env exhausts RTX ParameterBlock resources
(`logs/verify/wrist-10env-untiled.log`). Tiled 10-env restores black wrists.
Keep 5 env until a separate Isaac 6.0 checkout passes a 10-env three-camera smoke.


Assets (skip if `Assets/{Robots,Eval_Layout,Object,Material,Background,Room,Sensor,Traj}`
are already populated):

```bash
python3 /mnt/bn/robotics-data-mx/wenbo/XPolicyLab/scripts/fetch_robodojo_assets.py \
  --repo "$ROBODOJO_ROOT/.cache/robodojo_assets_repo" --workers 16
```

Render smoke (prints `RENDER_OK`):

```bash
python3 /mnt/bn/robotics-data-mx/wenbo/XPolicyLab/scripts/robodojo_render_smoke.py \
  --frames 3 --kit-args="$ROBODOJO_KIT_ARGS"
```

Official protocol, one seed, three policies, 8 co-located workers:

```bash
bash /mnt/bn/robotics-data-mx/wenbo/XPolicyLab/scripts/run_robodojo_official_protocol.sh \
  --seeds 0 --policies Pi_05,G05,Xiaomi_Robotics_1 --gpu-ids 0,1,2,3,4,5,6,7
```

Valid seed-0 Pi_05 sweep (started 2026-08-26 18:58 CST after the untiled-camera fix).
Use the comma form for GPU ids — `0-7` is not expanded and serialises onto one card:

```bash
bash /mnt/bn/robotics-data-mx/wenbo/XPolicyLab/scripts/run_robodojo_sim_eval.sh benchmark Pi_05 \
  --eval-num native --seed 0 \
  --policy-gpu-ids 0,1,2,3,4,5,6,7 --env-gpu-ids 0,1,2,3,4,5,6,7 \
  > /mnt/bn/robotics-data-mx/wenbo/XPolicyLab/experiments/robodojo-official-2026-08-25/logs/Pi_05-seed0-untiled.log 2>&1
# pid file: /tmp/pi05-sweep-untiled.pid
```

Elastic scheduler that fills idle cards across all three policies and kills the
sweep once Pi_05's 42-cell table is complete:

```bash
python3 /mnt/bn/robotics-data-mx/wenbo/XPolicyLab/scripts/elastic_robodojo_scheduler.py \
  --policies Pi_05,G05,Xiaomi_Robotics_1 --gpus 0,1,2,3,4,5,6,7 --seed 0 \
  --kill-pid "$(cat /tmp/pi05-sweep-untiled.pid)" --kill-pid-policy Pi_05 \
  --sweep-log /mnt/bn/robotics-data-mx/wenbo/XPolicyLab/experiments/robodojo-official-2026-08-25/logs/Pi_05-seed0-untiled.log \
  --max-attempts 3 \
  --log-dir /mnt/bn/robotics-data-mx/wenbo/XPolicyLab/experiments/robodojo-official-2026-08-25/logs/elastic-untiled \
  > /mnt/bn/robotics-data-mx/wenbo/XPolicyLab/experiments/robodojo-official-2026-08-25/logs/elastic-untiled.log 2>&1
# pid file: /tmp/elastic-untiled.pid
```

One task on one card, for a manual retry (assets must already be on disk):

```bash
bash /mnt/bn/robotics-data-mx/wenbo/XPolicyLab/scripts/run_robodojo_sim_eval.sh eval Pi_05 \
  --task stack_bowls --eval-num 5 --seed 0 --policy-gpu 0 --env-gpu 1
```

Invalid prior result trees (do not read these for the score):

- `RoboDojo-eval/eval_result_pre_renderfix/` — albedo missing
- `RoboDojo-eval/eval_result_pre_untiledfix/` — wrist cameras blank

Compare against the published leaderboard rates:

```bash
python3 /mnt/bn/robotics-data-mx/wenbo/XPolicyLab/scripts/compare_robodojo_to_official.py \
  --eval-root "$ROBODOJO_ROOT" --seed 0 \
  --json-out /mnt/bn/robotics-data-mx/wenbo/XPolicyLab/experiments/robodojo-official-2026-08-25/results/compare-seed0.json
```

Remaining-time estimate from shard weights (not a protocol score; scale 2x for `num_envs=5`):

```bash
ROBODOJO_NUM_ENVS=5 python3 /mnt/bn/robotics-data-mx/wenbo/XPolicyLab/scripts/robodojo_eval_eta.py \
  --robodojo-root "$ROBODOJO_ROOT" \
  --sweep-log /mnt/bn/robotics-data-mx/wenbo/XPolicyLab/experiments/robodojo-official-2026-08-25/logs/Pi_05-seed0-untiled.log \
  --json-out /mnt/bn/robotics-data-mx/wenbo/XPolicyLab/experiments/robodojo-official-2026-08-25/results/pi05-seed0-untiled-eta.json
```
