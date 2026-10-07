#!/usr/bin/env python3
"""Summarize EvalPlus held-out ISR runs."""

import argparse
import json
import math
import random
from pathlib import Path


def load(path):
    with Path(path).open() as f:
        data = json.load(f)
    return data, {r["task_id"]: bool(r.get("final_passed")) for r in data.get("results", [])}


def mode_key(data, path):
    mode = data.get("mode")
    if mode:
        return mode
    mode = data.get("metadata", {}).get("mode", "")
    for candidate in ["simple_retry", "code_refine", "zeroshot", "oneshot", "isr"]:
        if mode.endswith(f"_{candidate}") or mode == candidate:
            return candidate
    stem = Path(path).stem
    for candidate in ["simple_retry", "code_refine", "zeroshot", "oneshot", "isr"]:
        if f"_{candidate}_" in stem or stem.endswith(f"_{candidate}"):
            return candidate
    return stem


def mcnemar_pvalue(gains, losses):
    n = gains + losses
    if n == 0:
        return 1.0
    k = min(gains, losses)
    # Exact two-sided binomial under p=0.5.
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def bootstrap_ci(a, b, iterations=10000, seed=123):
    ids = sorted(set(a) & set(b))
    if not ids:
        return [0.0, 0.0]
    rng = random.Random(seed)
    diffs = []
    n = len(ids)
    vals = [(1 if b[i] else 0, 1 if a[i] else 0) for i in ids]
    for _ in range(iterations):
        total = 0
        for _ in range(n):
            bv, av = vals[rng.randrange(n)]
            total += bv - av
        diffs.append(100 * total / n)
    diffs.sort()
    return [diffs[int(0.025 * iterations)], diffs[int(0.975 * iterations)]]


def compare(label, baseline, method):
    ids = sorted(set(baseline) & set(method))
    gains = sum((not baseline[i]) and method[i] for i in ids)
    losses = sum(baseline[i] and (not method[i]) for i in ids)
    diff = 100 * (sum(method[i] for i in ids) - sum(baseline[i] for i in ids)) / len(ids) if ids else 0
    return {
        "label": label,
        "n": len(ids),
        "baseline_passed": sum(baseline[i] for i in ids),
        "method_passed": sum(method[i] for i in ids),
        "diff_pp": diff,
        "gains": gains,
        "losses": losses,
        "mcnemar_p": mcnemar_pvalue(gains, losses),
        "bootstrap_ci_pp": bootstrap_ci(baseline, method),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    loaded = {}
    metadata = {}
    for path in args.inputs:
        data, by_task = load(path)
        key = mode_key(data, path)
        loaded[key] = by_task
        metadata[key] = data.get("metadata", {})

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
        "metadata": metadata,
        "methods": {
            k: {
                "total": len(v),
                "passed": sum(v.values()),
                "pass_rate": sum(v.values()) / len(v) if v else 0,
            }
            for k, v in loaded.items()
        },
        "paired_comparisons": rows,
    }

    Path(args.out_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    lines = ["# EvalPlus Result Summary", ""]
    lines.extend(["## Methods", ""])
    for k, v in summary["methods"].items():
        lines.append(f"- {k}: {v['passed']}/{v['total']} ({100 * v['pass_rate']:.1f}%)")
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
