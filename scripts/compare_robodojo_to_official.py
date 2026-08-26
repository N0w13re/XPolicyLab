"""Compare reproduced RoboDojo success rates against the published leaderboard numbers.

Reads the per-task `_result.json` files the simulator writes and aggregates them the same
way `scripts/internal/summarize_result.py` does, so the numbers are directly comparable to
the leaderboard:

  - `X` and `X_random` are one reported task; the first 25 episodes of each are merged.
  - Every other task contributes its first 50 episodes.
  - A reported task counts only once its full episode budget is present, so a partial run
    lowers coverage instead of silently lowering the score.

Usage:
    python scripts/compare_robodojo_to_official.py --eval-root /path/to/RoboDojo-eval
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

# Overall simulation success rate published for these checkpoints.
OFFICIAL_SR = {
    "Pi_05": 6.91,
    "G05": 14.88,
    "Xiaomi_Robotics_1": 13.93,
}

SEED_RE = re.compile(r"^(\d+)_ckpt_name=")
EPISODES_PAIRED = 25
EPISODES_STANDALONE = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", required=True, type=Path)
    parser.add_argument("--bench", default="RoboDojo")
    parser.add_argument("--embodiment", default="arx_x5")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def latest_result(run_dir: Path) -> Path | None:
    """The summarizer only reads the newest timestamp dir that actually has a result."""
    stamps = sorted(
        (d for d in run_dir.iterdir() if (d / "_result.json").is_file()),
        key=lambda d: d.name,
    )
    return stamps[-1] / "_result.json" if stamps else None


def load_entries(path: Path) -> list[tuple[bool, float]]:
    with open(path) as fh:
        data = json.load(fh)
    details = data.get("details") or {}
    ordered = sorted(details.items(), key=lambda kv: int(kv[0]))
    return [(bool(v.get("success")), float(v.get("score", 0.0))) for _, v in ordered]


def collect(root: Path, bench: str, embodiment: str, seed: int) -> dict[str, dict[str, list]]:
    """{policy: {task: [(success, score), ...]}} for one seed."""
    out: dict[str, dict[str, list]] = {}
    bench_dir = root / "eval_result" / bench
    if not bench_dir.is_dir():
        return out
    for task_dir in bench_dir.iterdir():
        if not task_dir.is_dir() or task_dir.name.startswith("_"):
            continue
        for policy_dir in task_dir.iterdir():
            if not policy_dir.is_dir():
                continue
            emb_dir = policy_dir / embodiment
            if not emb_dir.is_dir():
                continue
            for run_dir in emb_dir.iterdir():
                m = SEED_RE.match(run_dir.name)
                if not m or int(m.group(1)) != seed or not run_dir.is_dir():
                    continue
                result = latest_result(run_dir)
                if result is None:
                    continue
                out.setdefault(policy_dir.name, {})[task_dir.name] = load_entries(result)
    return out


def canonical_tasks(root: Path) -> set[str]:
    """Every task the benchmark can run, from the task modules in the RoboDojo checkout."""
    task_dir = root / "task" / "RoboDojo" / "tasks"
    if not task_dir.is_dir():
        return set()
    return {p.stem for p in task_dir.glob("*.py") if not p.stem.startswith("_")}


def reported_tasks(
    tasks: dict[str, list], canonical: set[str]
) -> tuple[dict[str, list], list[str]]:
    """Merge X/X_random pairs; return complete reported tasks and the incomplete names.

    Pairing comes from the canonical task set, not from what happens to be on disk: a base
    whose `_random` sibling has not run yet is incomplete, not a 50-episode standalone.
    """
    universe = canonical or set(tasks)
    complete: dict[str, list] = {}
    incomplete: list[str] = []
    for base in sorted(t for t in universe if not t.endswith("_random")):
        rnd = f"{base}_random"
        if rnd in universe:
            first = tasks.get(base, [])[:EPISODES_PAIRED]
            second = tasks.get(rnd, [])[:EPISODES_PAIRED]
            if len(first) == EPISODES_PAIRED and len(second) == EPISODES_PAIRED:
                complete[base] = first + second
                continue
        else:
            entries = tasks.get(base, [])[:EPISODES_STANDALONE]
            if len(entries) == EPISODES_STANDALONE:
                complete[base] = entries
                continue
        incomplete.append(base)
    return complete, incomplete


def main() -> int:
    args = parse_args()
    root = args.eval_root.resolve()
    per_policy = collect(root, args.bench, args.embodiment, args.seed)
    if not per_policy:
        print(f"no results under {root}/eval_result/{args.bench}")
        return 1

    canonical = canonical_tasks(root)
    report = {}
    rows = []
    for policy in sorted(per_policy):
        complete, incomplete = reported_tasks(per_policy[policy], canonical)
        n_ep = sum(len(v) for v in complete.values())
        successes = sum(1 for v in complete.values() for s, _ in v if s)
        score_sum = sum(sc for v in complete.values() for _, sc in v)
        sr = successes / n_ep * 100 if n_ep else float("nan")
        score = score_sum / n_ep * 100 if n_ep else float("nan")
        official = OFFICIAL_SR.get(policy)
        delta = sr - official if official is not None and n_ep else None
        rows.append((policy, len(complete), n_ep, sr, score, official, delta))
        per_task = {}
        for name, entries in sorted(per_policy[policy].items()):
            suc = sum(1 for s, _ in entries if s)
            per_task[name] = {
                "episodes": len(entries),
                "successes": suc,
                "success_rate": suc / len(entries) * 100 if entries else None,
                "score": sum(sc for _, sc in entries) / len(entries) * 100 if entries else None,
            }
        raw_eps = sum(len(v) for v in per_policy[policy].values())
        raw_succ = sum(1 for v in per_policy[policy].values() for s, _ in v if s)
        report[policy] = {
            "reported_tasks_complete": len(complete),
            "episodes": n_ep,
            "success_rate": sr,
            "score": score,
            "official_success_rate": official,
            "delta": delta,
            "incomplete_tasks": sorted(incomplete),
            "in_progress_episodes": raw_eps,
            "in_progress_successes": raw_succ,
            "in_progress_success_rate": (
                raw_succ / raw_eps * 100 if raw_eps else None
            ),
            "per_task": per_task,
        }

    width = max(len(r[0]) for r in rows)
    print(f"seed {args.seed}, embodiment {args.embodiment}")
    print(
        f"{'policy'.ljust(width)}  {'tasks':>5}  {'eps':>5}  {'SR %':>7}  "
        f"{'score':>7}  {'official':>8}  {'delta':>7}"
    )
    for policy, ntask, neps, sr, score, official, delta in rows:
        off = f"{official:.2f}" if official is not None else "-"
        dlt = f"{delta:+.2f}" if delta is not None else "-"
        print(
            f"{policy.ljust(width)}  {ntask:5d}  {neps:5d}  {sr:7.2f}  "
            f"{score:7.2f}  {off:>8}  {dlt:>7}"
        )
    print("\nA reported task counts only when its full 50-episode budget is present;")
    print("`tasks` below 42 means the sweep is still incomplete and SR is a partial average.")

    for policy in sorted(per_policy):
        info = report[policy]
        raw_eps = info.get("in_progress_episodes") or 0
        if info["reported_tasks_complete"] >= 42 or raw_eps == 0:
            continue
        raw_succ = info.get("in_progress_successes") or 0
        raw_sr = info.get("in_progress_success_rate")
        print(
            f"\nin-progress {policy} (not official): "
            f"{raw_succ}/{raw_eps} eps"
            + (f" ({raw_sr:.2f}%)" if raw_sr is not None else "")
        )
        for name, stats in sorted(info["per_task"].items()):
            print(
                f"  {name:40s} {stats['successes']:3d}/{stats['episodes']:<4d}  "
                f"sr={stats['success_rate']:.1f}%"
            )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as fh:
            json.dump({"seed": args.seed, "policies": report}, fh, indent=2, sort_keys=True)
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
