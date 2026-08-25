# Reproducible commands for this run

Host: 8x NVIDIA A800-SXM4-80GB, driver 535.261.03. Policy servers and Isaac Sim
share a GPU. Apply the host fixes first; without them Kit never renders a frame.

```bash
export ROBODOJO_ROOT=/mnt/bn/robotics-data-mx/wenbo/RoboDojo-eval
source /mnt/bn/robotics-data-mx/wenbo/XPolicyLab/scripts/robodojo_sim_env.sh "$ROBODOJO_ROOT"
```

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

This seed-0 Pi_05 sweep is the one launched at 2026-08-25 07:41:

```bash
bash /mnt/bn/robotics-data-mx/wenbo/XPolicyLab/scripts/run_robodojo_sim_eval.sh benchmark Pi_05 \
  --eval-num native --seed 0 \
  --policy-gpu-ids 0,1,2,3,4,5,6,7 --env-gpu-ids 0,1,2,3,4,5,6,7
```

Compare against the published leaderboard rates:

```bash
python3 /mnt/bn/robotics-data-mx/wenbo/XPolicyLab/scripts/compare_robodojo_to_official.py \
  --eval-root "$ROBODOJO_ROOT" --seed 0 \
  --json-out /mnt/bn/robotics-data-mx/wenbo/XPolicyLab/experiments/robodojo-official-2026-08-25/results/compare-seed0.json
```
