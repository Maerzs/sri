"""
Generate audit tables for the acceptance-strengthening experiments.

The script is intentionally strict about completed 342-task runs. Partial
checkpoint files are reported, but excluded from manuscript-ready tables.
"""

import argparse
import json
import math
import random
import statistics
from pathlib import Path


METHOD_ORDER = ["zeroshot", "oneshot", "simple_retry", "code_refine", "isr"]
METHOD_LABEL = {
    "zeroshot": "Zero-shot",
    "oneshot": "One-shot skill",
    "simple_retry": "Simple Retry",
    "code_refine": "Code-level refine",
    "isr": "ISR",
}


def load_json(path):
    with path.open() as handle:
        return json.load(handle)


def pass_value(row):
    if "final_passed" in row:
        return bool(row["final_passed"])
    if "passed" in row:
        return bool(row["passed"])
    raise KeyError(f"no pass field for {row.get('task_id')}")


def run_map(path):
    data = load_json(path)
    metadata = data.get("metadata") or {}
    rows = data.get("results") or []
    task_pass = {row["task_id"]: pass_value(row) for row in rows if row.get("task_id")}
    passed = sum(task_pass.values())
    total = len(task_pass)
    mode = metadata.get("mode") or data.get("mode") or path.stem
    if mode.startswith("cross_task_reuse_"):
        mode = mode.replace("cross_task_reuse_", "", 1)
    return {
        "path": str(path),
        "mode": mode,
        "metadata": metadata,
        "task_pass": task_pass,
        "total": total,
        "passed": passed,
        "pass_rate": passed / total if total else 0.0,
        "complete": bool(metadata.get("complete")),
        "target_total": metadata.get("target_total"),
        "repeat_id": metadata.get("repeat_id"),
        "canonical_prefix_chars": metadata.get("canonical_prefix_chars"),
        "model_label": metadata.get("model_label"),
    }


def mcnemar_pvalue(gain, loss):
    discordant = gain + loss
    if discordant == 0:
        return 1.0
    chi_square = (abs(gain - loss) - 1) ** 2 / discordant
    return math.erfc(math.sqrt(chi_square / 2.0))


def percentile(values, pct):
    values = sorted(values)
    if not values:
        return 0.0
    idx = (len(values) - 1) * pct / 100.0
    lo = int(idx)
    hi = min(lo + 1, len(values) - 1)
    frac = idx - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def paired_stats(baseline, method, bootstrap_samples=10000, seed=0):
    common = sorted(set(baseline["task_pass"]) & set(method["task_pass"]))
    diffs = [int(method["task_pass"][task_id]) - int(baseline["task_pass"][task_id]) for task_id in common]
    gain = sum(d == 1 for d in diffs)
    loss = sum(d == -1 for d in diffs)
    rng = random.Random(seed)
    boot = []
    if diffs:
        for _ in range(bootstrap_samples):
            boot.append(100.0 * sum(diffs[rng.randrange(len(diffs))] for _ in diffs) / len(diffs))
    return {
        "n": len(common),
        "baseline_passed": sum(1 for task_id in common if baseline["task_pass"][task_id]),
        "method_passed": sum(1 for task_id in common if method["task_pass"][task_id]),
        "gain": gain,
        "loss": loss,
        "net": gain - loss,
        "diff_pp": 100.0 * sum(diffs) / len(diffs) if diffs else 0.0,
        "mcnemar_p": mcnemar_pvalue(gain, loss),
        "bootstrap_ci_low_pp": percentile(boot, 2.5),
        "bootstrap_ci_high_pp": percentile(boot, 97.5),
    }


def fmt_pct(run):
    return f"{100.0 * run['pass_rate']:.1f}\\% ({run['passed']}/{run['total']})"


def fmt_float(value, digits=1):
    return f"{value:.{digits}f}"


def fmt_p(value):
    if value < 1e-3:
        return f"{value:.2e}"
    return f"{value:.3f}"


def mean_std(values):
    values = list(values)
    return {
        "n": len(values),
        "mean": statistics.mean(values) if values else None,
        "std": statistics.stdev(values) if len(values) > 1 else 0.0 if values else None,
    }


def collect_runs(results_dir):
    patterns = [
        "haiku_heldout_*_n342_rep00.json",
        "haiku_heldout_*_n342_rep01.json",
        "haiku_prefix0_heldout_*_n342_rep00.json",
        "cross_task_reuse_*_n342_haiku_rep00.json",
        "cross_task_reuse_*_n342_*_haiku_rep00.json",
        "prefix0_heldout_*_n100_rep00.json",
    ]
    runs = {}
    for pattern in patterns:
        for path in sorted(results_dir.glob(pattern)):
            try:
                run = run_map(path)
            except Exception as exc:
                runs[str(path)] = {"path": str(path), "error": str(exc)}
                continue
            runs[str(path)] = run
    return runs


def completed_342(run):
    return run.get("complete") is True and run.get("total") == 342 and run.get("target_total") == 342


def by_mode(runs, prefix_chars=None, repeat_id=None, path_contains=None):
    out = {}
    for run in runs.values():
        if "error" in run:
            continue
        if prefix_chars is not None and run.get("canonical_prefix_chars") != prefix_chars:
            continue
        if repeat_id is not None and run.get("repeat_id") != repeat_id:
            continue
        if path_contains is not None and path_contains not in run.get("path", ""):
            continue
        mode = run.get("mode")
        if mode in METHOD_ORDER and completed_342(run):
            out[mode] = run
    return out


def comparison_block(group, baseline_name="oneshot"):
    comparisons = {}
    if baseline_name not in group:
        return comparisons
    baseline = group[baseline_name]
    for mode, run in group.items():
        if mode == baseline_name:
            continue
        comparisons[f"{mode}_vs_{baseline_name}"] = paired_stats(baseline, run)
    if "isr" in group:
        for baseline_mode in ["simple_retry", "code_refine", "zeroshot"]:
            if baseline_mode in group:
                comparisons[f"isr_vs_{baseline_mode}"] = paired_stats(group[baseline_mode], group["isr"])
    return comparisons


def build_report(results_dir):
    runs = collect_runs(results_dir)
    prefix400_rep00 = by_mode(runs, prefix_chars=400, repeat_id="haiku342_rep00")
    prefix400_rep01 = by_mode(runs, prefix_chars=400, repeat_id="haiku342_rep01")
    prefix0_rep00 = by_mode(runs, prefix_chars=0, repeat_id="haiku342_prefix0_rep00")

    reuse = {}
    for run in runs.values():
        if "error" in run or "cross_task_reuse_" not in run.get("path", ""):
            continue
        mode = run.get("mode")
        if completed_342(run):
            reuse[mode] = run

    stability = {}
    for mode in METHOD_ORDER:
        vals = [g[mode]["pass_rate"] for g in [prefix400_rep00, prefix400_rep01] if mode in g]
        if vals:
            stability[mode] = mean_std(vals)

    isr_diffs = {}
    for label, group in [("rep00", prefix400_rep00), ("rep01", prefix400_rep01)]:
        if "isr" not in group:
            continue
        for baseline in ["oneshot", "simple_retry", "code_refine"]:
            if baseline in group:
                stats = paired_stats(group[baseline], group["isr"])
                isr_diffs.setdefault(f"isr_vs_{baseline}", []).append({
                    "repeat": label,
                    "diff_pp": stats["diff_pp"],
                    "gain": stats["gain"],
                    "loss": stats["loss"],
                    "method_passed": stats["method_passed"],
                    "baseline_passed": stats["baseline_passed"],
                })

    stability_diffs = {
        key: {
            "n": len(items),
            "diff_pp_mean": statistics.mean(item["diff_pp"] for item in items),
            "diff_pp_std": statistics.stdev(item["diff_pp"] for item in items) if len(items) > 1 else 0.0,
            "items": items,
        }
        for key, items in isr_diffs.items()
    }

    reuse_comparisons = {}
    if "refined" in reuse:
        for baseline in ["random", "initial", "oneshot", "zeroshot"]:
            if baseline in reuse:
                reuse_comparisons[f"refined_vs_{baseline}"] = paired_stats(reuse[baseline], reuse["refined"])

    return {
        "results_dir": str(results_dir),
        "runs": runs,
        "prefix400_rep00": prefix400_rep00,
        "prefix400_rep01": prefix400_rep01,
        "prefix0_rep00": prefix0_rep00,
        "prefix400_rep00_comparisons": comparison_block(prefix400_rep00),
        "prefix400_rep01_comparisons": comparison_block(prefix400_rep01),
        "prefix0_rep00_comparisons": comparison_block(prefix0_rep00),
        "stability": stability,
        "stability_diffs": stability_diffs,
        "reuse": reuse,
        "reuse_comparisons": reuse_comparisons,
        "excluded_partial_or_incomplete": [
            {
                "path": run.get("path"),
                "mode": run.get("mode"),
                "total": run.get("total"),
                "target_total": run.get("target_total"),
                "complete": run.get("complete"),
                "repeat_id": run.get("repeat_id"),
                "canonical_prefix_chars": run.get("canonical_prefix_chars"),
            }
            for run in runs.values()
            if "error" not in run and not completed_342(run)
        ],
    }


def markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def render_method_table(title, group, comparisons):
    lines = [f"## {title}", ""]
    if not group:
        lines.extend(["No complete 342-task runs available.", ""])
        return lines
    rows = []
    one = group.get("oneshot")
    for mode in METHOD_ORDER:
        if mode not in group:
            continue
        run = group[mode]
        comp = comparisons.get(f"{mode}_vs_oneshot") if one and mode != "oneshot" else None
        rows.append([
            METHOD_LABEL.get(mode, mode),
            f"{run['passed']}/{run['total']}",
            f"{100.0 * run['pass_rate']:.1f}",
            "-" if comp is None else f"{comp['gain']}/{comp['loss']}",
            "-" if comp is None else fmt_float(comp["diff_pp"]),
            "-" if comp is None else fmt_p(comp["mcnemar_p"]),
        ])
    lines.append(markdown_table(
        ["method", "passed", "pass %", "gains/losses vs one-shot", "diff pp", "McNemar p"],
        rows,
    ))
    lines.append("")
    return lines


def write_markdown(path, report):
    lines = ["# Acceptance Boost Audit", ""]
    lines.extend(render_method_table(
        "Haiku Held-Out, Prefix=400, Rep00",
        report["prefix400_rep00"],
        report["prefix400_rep00_comparisons"],
    ))
    lines.extend(render_method_table(
        "Haiku Held-Out, Prefix=0, Rep00",
        report["prefix0_rep00"],
        report["prefix0_rep00_comparisons"],
    ))
    lines.extend(render_method_table(
        "Haiku Held-Out, Prefix=400, Rep01",
        report["prefix400_rep01"],
        report["prefix400_rep01_comparisons"],
    ))

    lines.extend(["## Stability Across Complete Haiku Repeats", ""])
    if report["stability"]:
        lines.append(markdown_table(
            ["method", "n", "mean pass %", "std pass %"],
            [
                [
                    METHOD_LABEL.get(mode, mode),
                    stats["n"],
                    fmt_float(100.0 * stats["mean"]),
                    fmt_float(100.0 * stats["std"]),
                ]
                for mode, stats in sorted(report["stability"].items(), key=lambda kv: METHOD_ORDER.index(kv[0]))
            ],
        ))
    else:
        lines.append("No complete repeated Haiku runs available.")
    lines.append("")

    lines.extend(["## ISR Paired Diff Stability", ""])
    if report["stability_diffs"]:
        lines.append(markdown_table(
            ["comparison", "n", "mean diff pp", "std diff pp", "per-repeat"],
            [
                [
                    key,
                    item["n"],
                    fmt_float(item["diff_pp_mean"]),
                    fmt_float(item["diff_pp_std"]),
                    "; ".join(
                        f"{row['repeat']}: {row['method_passed']}-{row['baseline_passed']} ({row['gain']}/{row['loss']})"
                        for row in item["items"]
                    ),
                ]
                for key, item in sorted(report["stability_diffs"].items())
            ],
        ))
    else:
        lines.append("No complete repeated paired comparisons available.")
    lines.append("")

    lines.extend(["## Cross-Task Reuse", ""])
    if report["reuse"]:
        lines.append(markdown_table(
            ["mode", "passed", "pass %"],
            [
                [mode, f"{run['passed']}/{run['total']}", fmt_float(100.0 * run["pass_rate"])]
                for mode, run in sorted(report["reuse"].items())
            ],
        ))
        lines.append("")
        if report["reuse_comparisons"]:
            lines.append(markdown_table(
                ["comparison", "passed", "gains/losses", "diff pp", "McNemar p", "95% CI pp"],
                [
                    [
                        name,
                        f"{stats['baseline_passed']} -> {stats['method_passed']}",
                        f"{stats['gain']}/{stats['loss']}",
                        fmt_float(stats["diff_pp"]),
                        fmt_p(stats["mcnemar_p"]),
                        f"[{fmt_float(stats['bootstrap_ci_low_pp'])}, {fmt_float(stats['bootstrap_ci_high_pp'])}]",
                    ]
                    for name, stats in sorted(report["reuse_comparisons"].items())
                ],
            ))
        else:
            lines.append("No complete reuse comparisons available.")
    else:
        lines.append("No complete cross-task reuse runs available.")
    lines.append("")

    lines.extend(["## Excluded Partial Or Incomplete Runs", ""])
    if report["excluded_partial_or_incomplete"]:
        lines.append(markdown_table(
            ["path", "mode", "total", "target", "complete", "repeat", "prefix"],
            [
                [
                    Path(row["path"]).name,
                    row["mode"],
                    row["total"],
                    row["target_total"],
                    row["complete"],
                    row["repeat_id"],
                    row["canonical_prefix_chars"],
                ]
                for row in report["excluded_partial_or_incomplete"]
            ],
        ))
    else:
        lines.append("None.")
    lines.append("")
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results/acceptance_boost"))
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    report = build_report(args.results_dir)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    write_markdown(args.out_md, report)
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
