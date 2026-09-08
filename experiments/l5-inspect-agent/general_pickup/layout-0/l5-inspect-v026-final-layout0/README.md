# L5 inspect-agent integration evidence

- Adapter revision: `fa2c55d`
- Task/layout: `general_pickup`, RoboDojo eval seed 0, layout 0
- Model wire: Azure Chat transport over upstream Chat client
- Policy budget: 2 LLM calls
- Process exit: 0
- Official RoboDojo result: failure, score 0.0

Command shape (credentials and endpoint omitted):

```bash
P3_MODEL=azure/<deployment> \
P3_WIRE=azure-chat \
P3_MAX_LLM_CALLS=2 \
P3_IMAGES=always \
P3_DEPTH=render \
P3_WIRE_CAPTURE=true \
ROBODOJO_RUN_ID=l5-inspect-v026-final-layout0 \
bash policy/Agent_P3/run_fixed_layout.sh 0 0 general_pickup uv
```

`p3_config.json` records the exact upstream versions/config, scene/layout and
live embodiment. `p3_transcript.json` records the policy conversation and
termination. `robodojo_result.json` is copied byte-for-byte in data content
from RoboDojo's authoritative `_result.json`. The `wire/` tree is the upstream
per-attempt replay capture, including deduplicated RGB image blobs.

This is an integration/evidence run, not a leaderboard score claim. The
two-call cap deliberately exercised structured motion repair followed by the
upstream forced-`give_up` path; RoboDojo finalized and scored that stop as a
failure.
