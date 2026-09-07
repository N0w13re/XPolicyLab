# Agent_P3 — the LLM is the policy

P3 is the far end of the [P0–P3 axis](../../docs/p0_p3_agent_manipulation_protocol.md):
how close control sits to the LLM. P0 has no LLM. P1 lets it aim a frozen VLA.
P2 lets it call primitives we wrote. P3 removes both, and the model emits the
actions itself.

| | Who emits the action | What stands between the model and the robot |
| --- | --- | --- |
| P0 | Frozen VLA | Everything; there is no LLM |
| P1 | Frozen VLA | The LLM aims the arm, then hands off |
| P2 | Our primitives | `pick`, `place`, `move_ee` |
| P3 | The LLM | Nothing but action decoding |

## The contract

The LLM occupies exactly the slot a served VLA occupies, so the harness cannot
tell from the contract which one it is driving:

```python
reset() -> None
act(observation) -> ActionChunk
```

`LlmPolicy` in `policy.py` satisfies the `Policy` protocol in `types.py`, and
so would any VLA. That equality is the point of the condition: a P3 number is
comparable to a P0 number because the two answer the same two calls.

## How an action gets made

1. `deploy.py` reads the environment observation: head and wrist RGB, the joint
   state, and the remaining step budget.
2. `policy.py` builds one request. The images go in as PNG data URLs and the
   state goes in as a flat vector **in the same order the answer must use**, so
   the model never converts between two layouts.
3. The action space *is* the tool schema. `ActionSpace.tool_schema()` is
   generated from the channel table in `types.py`, so an action the model can
   express and an action the websocket accepts cannot drift apart.
4. The tool call's `actions` array is decoded straight into the websocket dict
   (`left_arm_joint_state` 6, `left_ee_joint_state` 1, `right_arm_joint_state`
   6, `right_ee_joint_state` 1) and executed in order.

Nothing clamps, retargets, or interpolates the numbers. A malformed call is
sent back to the model with the reason and it tries again, bounded by
`max_repairs`; a call that cannot be repaired ends the episode as a failure
rather than being fixed up, because fixing it up would make the condition P2.

## Chunk length is the condition, not a constant

`P3_MAX_CHUNK=1` is closed-loop LLM control at the simulator's rate: the model
sees the result of every action. `P3_MAX_CHUNK=50` matches a Pi_05 chunk and
asks whether it can commit to a whole motion blind. These are different
conditions and must be reported separately. Start at 1.

## Wires

Each provider is called on its own endpoint rather than through a compat layer,
because P3 measures a model as its vendor exposes it:

| Wire | Endpoint | Notes |
| --- | --- | --- |
| `messages` | `POST /messages` | Anthropic native; system block carries `cache_control` |
| `chat` | `POST /chat/completions` | OpenAI-compatible; also vLLM, OpenRouter, DashScope |

Requests go out over `urllib`, with no SDK and no dependency the eval
environments do not already have. Every client takes an injectable `transport`,
which is how the tests exercise the wires without a network.

Retries: 3 attempts on 408/409/429/5xx and transport errors, with
`backoff_s * 2**attempt`. Any other 4xx is our request's fault and fails at
once with the response body attached.

## Configuration

| Variable | Meaning |
| --- | --- |
| `P3_MODEL` | `provider/model`, e.g. `anthropic/claude-sonnet-4-20250514` |
| `P3_MAX_CHUNK` | Actions per turn (default 1) |
| `P3_ACTION_TYPE` | `joint` (default) or `ee` |
| `P3_WIRE` | Override the wire the prefix table chose |
| `P3_BASE_URL` | Point at a local server; wins over the prefix table |
| `P3_API_KEY_ENV` | Name of the variable holding the key |
| `P3_TRACE_DIR` | Where to write `p3_transcript.json` |
| `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, … | Per-provider keys |
| `OPENROUTER_API_KEY` | Fallback router for any model with no direct key |

Resolution order is: explicit `P3_BASE_URL`, then the prefix table in
`wire.py`, then OpenRouter. A model with no reachable key raises `ConfigError`
naming the variable to set rather than failing at request time.

## Running

```bash
export P3_MODEL=anthropic/claude-sonnet-4-20250514
export ANTHROPIC_API_KEY=...
export P3_MAX_CHUNK=1
export P3_TRACE_DIR=/tmp/xpolicylab-p3/$RUN_ID

bash policy/Agent_P3/run_fixed_layout.sh <layout> <env_gpu> arrange_largest_number
```

P3 holds no VLA, so there is no checkpoint, no `policy_uv_env_path`, and no GPU
for inference. Its cost is API tokens, which means it can run alongside a GPU
sweep without competing for one.

## Accounting

Token usage is summed per episode from whatever the wire reports
(`input_tokens`, `output_tokens`, `cache_read_input_tokens`, and
`cache_creation_input_tokens` on the Anthropic wire) and written with the turn
log to `$P3_TRACE_DIR/p3_transcript.json`, alongside `llm_calls`. Repair turns
count as calls, so a rising call-to-step ratio is the signal that the model is
struggling to emit well-formed actions rather than struggling with the task.

## What would make this not P3

- Adding IK, a planner, or a `pick`/`place` skill — that is P2.
- Clamping or retargeting the model's numbers before sending them. A safety
  check that *rejects* an action and aborts stays P3, and the rejection must be
  recorded; one that *edits* it does not.
- Feeding GT object names, layout JSON, or reward-script answers.
