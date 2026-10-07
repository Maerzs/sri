"""
Aggregate multiple ISR experiment JSON files.

Handles both full-run JSONs with result["passed"] and heldout JSONs with
result["final_passed"]. Produces per-repeat method summaries plus paired
ISR-vs-baseline deltas for oneshot, simple_retry/retry, and code_refine/code.
"""

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


BASELINE_ALIASES = {
    "oneshot": ("oneshot",),
    "retry": ("simple_retry", "retry"),
    "code": ("code_refine", "code"),
}


def load_json(path):
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected top-level JSON object")
    return data


def pass_field(record):
    if "final_passed" in record:
        return bool(record["final_passed"])
    if "passed" in record:
        return bool(record["passed"])
    if "private_passed" in record and "feedback_passed" in record:
        return bool(record["private_passed"] and record["feedback_passed"])
    raise KeyError("result record lacks final_passed/passed fields")


def infer_mode(data, path):
    metadata = data.get("metadata") or {}
    return metadata.get("mode") or data.get("mode") or Path(path).stem


def infer_repeat(data, path):
    metadata = data.get("metadata") or {}
    repeat = metadata.get("repeat_id", data.get("repeat_id"))
    if repeat is not None:
        return str(repeat)
    stem = Path(path).stem
    match = re.search(r"(?:^|[_-])(?:repeat|rep|r)(\d+)(?:$|[_-])", stem, flags=re.IGNORECASE)
    if match:
        return f"r{match.group(1)}"
    return "default"


def normalize_run(path):
    data = load_json(path)
    results = data.get("results")
    if not isinstance(results, list):
        raise ValueError(f"{path}: expected a results list")
    mode = infer_mode(data, path)
    repeat_id = infer_repeat(data, path)
    metadata = data.get("metadata") or {}
    task_pass = {}
    for rec in results:
        task_id = rec.get("task_id")
        if not task_id:
            continue
        task_pass[str(task_id)] = pass_field(rec)
    total = len(task_pass)
    passed = sum(task_pass.values())
    return {
        "path": str(path),
        "mode": mode,
        "repeat_id": repeat_id,
        "metadata": metadata,
        "task_pass": task_pass,
        "total": total,
        "passed": passed,
        "pass_rate": passed / total if total else 0.0,
    }


def mean_std(values):
    values = list(values)
    if not values:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def paired_diff(target, baseline):
    common = sorted(set(target["task_pass"]) & set(baseline["task_pass"]))
    gains = []
    losses = []
    ties = 0
    diffs = []
    for task_id in common:
        target_pass = target["task_pass"][task_id]
        baseline_pass = baseline["task_pass"][task_id]
        diff = int(target_pass) - int(baseline_pass)
        diffs.append(diff)
        if diff > 0:
            gains.append(task_id)
        elif diff < 0:
            losses.append(task_id)
        else:
            ties += 1
    n = len(common)
    mean = statistics.mean(diffs) if diffs else 0.0
    std = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
    return {
        "repeat_id": target["repeat_id"],
        "target": target["mode"],
        "baseline": baseline["mode"],
        "n": n,
        "target_pass": sum(1 for t in common if target["task_pass"][t]),
        "baseline_pass": sum(1 for t in common if baseline["task_pass"][t]),
        "diff": sum(diffs),
        "mean": mean,
        "std": std,
        "gains": len(gains),
        "losses": len(losses),
        "ties": ties,
        "gain_task_ids": gains,
        "loss_task_ids": losses,
    }


def build_report(paths):
    runs = [normalize_run(path) for path in paths]

    per_repeat_method = []
    by_repeat_mode = {}
    rates_by_mode = defaultdict(list)
    for run in runs:
        row = {
            "repeat_id": run["repeat_id"],
            "mode": run["mode"],
            "total": run["total"],
            "passed": run["passed"],
            "pass_rate": run["pass_rate"],
            "path": run["path"],
        }
        per_repeat_method.append(row)
        by_repeat_mode[(run["repeat_id"], run["mode"])] = run
        rates_by_mode[run["mode"]].append(run["pass_rate"])

    method_summary = {
        mode: mean_std(values)
        for mode, values in sorted(rates_by_mode.items())
    }

    paired = []
    for (repeat_id, mode), run in sorted(by_repeat_mode.items()):
        if mode != "isr":
            continue
        for label, aliases in BASELINE_ALIASES.items():
            baseline = None
            for alias in aliases:
                baseline = by_repeat_mode.get((repeat_id, alias))
                if baseline is not None:
                    break
            if baseline is not None:
                item = paired_diff(run, baseline)
                item["comparison"] = f"isr_vs_{label}"
                paired.append(item)

    paired_summary = {}
    for comparison in sorted({p["comparison"] for p in paired}):
        items = [p for p in paired if p["comparison"] == comparison]
        paired_summary[comparison] = {
            "diff": mean_std(p["diff"] for p in items),
            "mean": mean_std(p["mean"] for p in items),
            "gains": mean_std(p["gains"] for p in items),
            "losses": mean_std(p["losses"] for p in items),
        }

    return {
        "inputs": [str(p) for p in paths],
        "per_repeat_method": sorted(per_repeat_method, key=lambda r: (r["repeat_id"], r["mode"])),
        "method_summary": method_summary,
        "paired": paired,
        "paired_summary": paired_summary,
    }


def fmt_float(value, digits=4):
    if value is None:
        return "NA"
    if isinstance(value, float):
        if math.isnan(value):
            return "NA"
        return f"{value:.{digits}f}"
    return str(value)


def markdown_table(headers, rows):
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    out.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(out)


def render_markdown(report):
    lines = ["# Multirun aggregate", ""]

    lines.extend(["## Per repeat method", ""])
    lines.append(markdown_table(
        ["repeat", "method", "passed", "total", "pass_rate", "path"],
        [
            [
                r["repeat_id"],
                r["mode"],
                r["passed"],
                r["total"],
                fmt_float(r["pass_rate"]),
                r["path"],
            ]
            for r in report["per_repeat_method"]
        ],
    ))

    lines.extend(["", "## Method pass-rate summary", ""])
    lines.append(markdown_table(
        ["method", "n_repeats", "mean", "std"],
        [
            [mode, vals["n"], fmt_float(vals["mean"]), fmt_float(vals["std"])]
            for mode, vals in sorted(report["method_summary"].items())
        ],
    ))

    lines.extend(["", "## Paired ISR comparisons", ""])
    if report["paired"]:
        lines.append(markdown_table(
            ["repeat", "comparison", "n", "isr_pass", "baseline_pass", "diff", "mean", "std", "gains", "losses"],
            [
                [
                    p["repeat_id"],
                    p["comparison"],
                    p["n"],
                    p["target_pass"],
                    p["baseline_pass"],
                    p["diff"],
                    fmt_float(p["mean"]),
                    fmt_float(p["std"]),
                    p["gains"],
                    p["losses"],
                ]
                for p in report["paired"]
            ],
        ))
    else:
        lines.append("No paired ISR comparisons found. Provide ISR and baseline JSONs with matching repeat_id.")

    lines.extend(["", "## Paired summary", ""])
    if report["paired_summary"]:
        lines.append(markdown_table(
            ["comparison", "n_repeats", "diff_mean", "diff_std", "task_mean_mean", "task_mean_std", "gains_mean", "losses_mean"],
            [
                [
                    name,
                    vals["diff"]["n"],
                    fmt_float(vals["diff"]["mean"]),
                    fmt_float(vals["diff"]["std"]),
                    fmt_float(vals["mean"]["mean"]),
                    fmt_float(vals["mean"]["std"]),
                    fmt_float(vals["gains"]["mean"]),
                    fmt_float(vals["losses"]["mean"]),
                ]
                for name, vals in sorted(report["paired_summary"].items())
            ],
        ))
    else:
        lines.append("No paired summary available.")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="Experiment JSON files to aggregate")
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args()

    report = build_report([Path(p) for p in args.inputs])

    if args.out_json:
        args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    if args.out_md:
        args.out_md.write_text(render_markdown(report))
    if not args.out_json and not args.out_md:
        print(render_markdown(report), end="")


if __name__ == "__main__":
    main()
