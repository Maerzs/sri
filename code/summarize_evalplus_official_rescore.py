#!/usr/bin/env python3
"""Summarize official EvalPlus re-score JSON files."""

import argparse
import json
import math
import random
from pathlib import Path


def load(path):
    data = json.loads(Path(path).read_text())
    mode = data.get("mode") or Path(path).stem
    for candidate in ["simple_retry", "code_refine", "zeroshot", "oneshot", "isr"]:
        if candidate in mode:
            mode = candidate
            break
    return data, mode, {r["task_id"]: bool(r.get("official_final_passed")) for r in data["results"]}


def mcnemar(gains, losses):
    n = gains + losses
    if n == 0:
        return 1.0
    k = min(gains, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def bootstrap_ci(a, b, iterations=10000, seed=123):
    ids = sorted(set(a) & set(b))
    rng = random.Random(seed)
    vals = [(1 if b[i] else 0, 1 if a[i] else 0) for i in ids]
    diffs = []
    for _ in range(iterations):
        total = 0
        for _ in vals:
            bv, av = vals[rng.randrange(len(vals))]
            total += bv - av
        diffs.append(100 * total / len(vals))
    diffs.sort()
    return [diffs[int(0.025 * iterations)], diffs[int(0.975 * iterations)]]


def compare(label, baseline, method):
    ids = sorted(set(baseline) & set(method))
    gains = sum((not baseline[i]) and method[i] for i in ids)
    losses = sum(baseline[i] and (not method[i]) for i in ids)
    diff = 100 * (sum(method[i] for i in ids) - sum(baseline[i] for i in ids)) / len(ids)
    return {
        "label": label,
        "n": len(ids),
        "baseline_passed": sum(baseline[i] for i in ids),
        "method_passed": sum(method[i] for i in ids),
        "diff_pp": diff,
        "gains": gains,
        "losses": losses,
        "mcnemar_p": mcnemar(gains, losses),
        "bootstrap_ci_pp": bootstrap_ci(baseline, method),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    loaded = {}
    source = {}
    for path in args.inputs:
        data, mode, by_task = load(path)
        loaded[mode] = by_task
        source[mode] = {
            "path": path,
            "local_final_passed": data.get("local_final_passed"),
            "official_final_passed": data.get("official_final_passed"),
            "disagreements": data.get("disagreements"),
        }

    rows = []
    for method in ["zeroshot", "simple_retry", "code_refine", "isr"]:
        if "oneshot" in loaded and method in loaded:
            rows.append(compare(f"{method}_vs_oneshot", loaded["oneshot"], loaded[method]))
    if "simple_retry" in loaded and "isr" in loaded:
        rows.append(compare("isr_vs_simple_retry", loaded["simple_retry"], loaded["isr"]))
    if "code_refine" in loaded and "isr" in loaded:
        rows.append(compare("isr_vs_code_refine", loaded["code_refine"], loaded["isr"]))

    summary = {
        "inputs": args.inputs,
        "source": source,
        "methods": {
            mode: {
                "total": len(vals),
                "passed": sum(vals.values()),
                "pass_rate": sum(vals.values()) / len(vals) if vals else 0,
            }
            for mode, vals in loaded.items()
        },
        "paired_comparisons": rows,
    }
    Path(args.out_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    lines = ["# Official EvalPlus Re-score Summary", "", "## Methods", ""]
    for mode, stat in summary["methods"].items():
        src = summary["source"][mode]
        lines.append(
            f"- {mode}: {stat['passed']}/{stat['total']} ({100 * stat['pass_rate']:.1f}%), "
            f"local={src['local_final_passed']}, disagreements={src['disagreements']}"
        )
    lines.extend(["", "## Paired Comparisons", ""])
    for row in rows:
        lo, hi = row["bootstrap_ci_pp"]
        lines.append(
            f"- {row['label']}: {row['method_passed']}/{row['n']} vs {row['baseline_passed']}/{row['n']}, "
            f"{row['diff_pp']:+.1f} pp, gains/losses {row['gains']}/{row['losses']}, "
            f"p={row['mcnemar_p']:.3g}, CI [{lo:+.1f},{hi:+.1f}]"
        )
    lines.append("")
    Path(args.out_md).write_text("\n".join(lines))
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
