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

Retry the three Traj-backed tasks on an idle GPU (assets must already be on disk):

```bash
GPU=1 bash /mnt/bn/robotics-data-mx/wenbo/XPolicyLab/scripts/retry_pi05_traj_tasks.sh
```

After the in-flight Pi_05 sweep (pid 216753) and Traj coordinator
(`/tmp/pi05-traj-all.pid`) exit, continue G05 then Xiaomi on all 8 GPUs:

```bash
PI05_PID=216753 TRAJ_PID_FILE=/tmp/pi05-traj-all.pid \
  bash /mnt/bn/robotics-data-mx/wenbo/XPolicyLab/scripts/chain_robodojo_remaining.sh
```

Parallel Traj retry (imitate already running on GPU 1; make_kong then
`play_tic_tac_toe` on GPU 0 so play does not wait for imitate's 1600-step horizon):

```bash
IMITATE_PID=<robodojo.sh eval pid> RETRY_PARENT=<retry_pi05_traj_tasks.sh pid> \
  TRAJ_PID_FILE=/tmp/pi05-traj-all.pid \
  bash /mnt/bn/robotics-data-mx/wenbo/XPolicyLab/scripts/coord_pi05_traj_retry.sh
```

If that coordinator was started with the old "play after imitate on GPU 1" order,
overlap play onto GPU 0 as soon as make_kong exits:

```bash
MAKE_PID=<make_kong pid> IMITATE_PID=<imitate pid> OLD_COORD=<coord pid> \
  bash /mnt/bn/robotics-data-mx/wenbo/XPolicyLab/scripts/coord_pi05_traj_play_gpu0.sh
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
