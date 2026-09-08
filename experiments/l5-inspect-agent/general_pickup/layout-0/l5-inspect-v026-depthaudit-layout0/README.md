# L5 integration evidence for the depth-audit revision

Second archived run, produced by the current adapter revision and by a wire that
needs no external credentials, so it can be reproduced from this checkout alone.

- Adapter revision: `3dc90a2`
- Task/layout: `general_pickup`, RoboDojo eval seed 0, layout 0
- Model wire: upstream Chat client against a locally served `Qwen3-VL-4B-Instruct`
- Policy budget: 4 LLM calls, 3 spent
- Observation: three RGB cameras plus three metric-depth renders; the transcript
  field `depth_cameras` names them, so the depth condition is recorded rather
  than assumed
- Inspect trial record: `status="error"`, both `terminated` and `truncated`
  false, exactly as upstream records a policy exception
- Official RoboDojo result: failure, score 0.0

Command shape:

```bash
P3_MODEL=local/Qwen3-VL-4B-Instruct \
P3_BASE_URL=<local server>/v1 P3_API_KEY=local P3_WIRE=chat \
P3_MAX_LLM_CALLS=4 P3_IMAGES=always P3_DEPTH=render P3_WIRE_CAPTURE=true \
ROBODOJO_RUN_ID=l5-inspect-v026-depthaudit-layout0 \
bash policy/Agent_P3/run_fixed_layout.sh 0 0 general_pickup uv
```

The 4B model kept proposing motions past the upstream playout cap, so this run
exercises the full upstream repair path: three structured tool errors in a row
end the turn, the bridge finalizes the episode as a RoboDojo failure, and
RoboDojo independently scores it. `wire/` holds three per-attempt request and
response rows and six deduplicated PNG blobs (three RGB, three depth).
`robodojo_result.json` is the data-identical copy of RoboDojo's authoritative
`_result.json`.

This is integration evidence, not a capability claim; a 4B local model is not a
leaderboard condition. The Azure run in `../l5-inspect-v026-final-depth-layout0/`
covers a frontier model on the same task and layout.
