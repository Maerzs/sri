"""
Compute paired task-level statistics used by the ISR manuscript.

The script reads local result JSON files and writes a reproducible audit report
with continuity-corrected McNemar p-values and bootstrap confidence intervals
for pass-rate differences.
"""

import argparse
import json
import math
import random
from pathlib import Path


DEFAULT_COMPARISONS = [
    {
        "name": "ISR - One-shot",
        "baseline": "oneshot_full.json",
        "method": "isr_full.json",
        "text_ref": "Table 1 and main-results paragraph",
    },
    {
        "name": "Simple Retry - One-shot",
        "baseline": "oneshot_full.json",
        "method": "simple_retry_full.json",
        "text_ref": "Isolating the source of the gain",
    },
    {
        "name": "ISR - Simple Retry",
        "baseline": "simple_retry_full.json",
        "method": "isr_full.json",
        "text_ref": "Table 1 and Simple Retry paragraph",
    },
    {
        "name": "ISR - Zero-shot",
        "baseline": "zeroshot_full.json",
        "method": "isr_full.json",
        "text_ref": "Table 1",
    },
    {
        "name": "ISR - Code-level refine",
        "baseline": "code_refine_full.json",
        "method": "isr_full.json",
        "text_ref": "Skill-vs-code paragraph",
    },
    {
        "name": "Zero-shot - One-shot",
        "baseline": "oneshot_full.json",
        "method": "zeroshot_full.json",
        "text_ref": "Default retriever comparison",
    },
    {
        "name": "Trace full - Generic no-trace",
        "baseline": "trace_ablation_full.json::generic",
        "method": "trace_ablation_full.json::full",
        "text_ref": "Failure trace content",
    },
    {
        "name": "Error-only - Trace full",
        "baseline": "trace_ablation_full.json::full",
        "method": "trace_ablation_full.json::error_only",
        "text_ref": "Failure trace content",
    },
    {
        "name": "Dense one-shot - TF-IDF one-shot",
        "baseline": "oneshot_full.json",
        "method": "emb_oneshot_full.json",
        "text_ref": "Dense retrieval paragraph",
    },
    {
        "name": "Dense one-shot - Zero-shot",
        "baseline": "zeroshot_full.json",
        "method": "emb_oneshot_full.json",
        "text_ref": "Dense retrieval table",
    },
    {
        "name": "Dense ISR - Dense one-shot",
        "baseline": "emb_oneshot_full.json",
        "method": "emb_isr_full.json",
        "text_ref": "Dense retrieval paragraph",
    },
    {
        "name": "V3.2 dense ISR - V3.2 dense one-shot",
        "baseline": "v32/emb_oneshot_full.json",
        "method": "v32/emb_isr_full.json",
        "text_ref": "Dense retrieval paragraph",
    },
    {
        "name": "Held-out Simple Retry - Held-out One-shot",
        "baseline": "heldout_oneshot_n342_seed123.json",
        "method": "heldout_simple_retry_n342_seed123.json",
        "text_ref": "Held-out audit table",
    },
    {
        "name": "Held-out ISR - Held-out One-shot",
        "baseline": "heldout_oneshot_n342_seed123.json",
        "method": "heldout_isr_n342_seed123.json",
        "text_ref": "Held-out audit table",
    },
    {
        "name": "Held-out ISR - Held-out Simple Retry",
        "baseline": "heldout_simple_retry_n342_seed123.json",
        "method": "heldout_isr_n342_seed123.json",
        "text_ref": "Held-out audit paragraph",
    },
    {
        "name": "Held-out Code-level refine - Held-out One-shot",
        "baseline": "heldout_oneshot_n342_seed123.json",
        "method": "heldout_code_refine_n342_seed123.json",
        "text_ref": "Held-out audit table",
    },
    {
        "name": "Held-out ISR - Held-out Code-level refine",
        "baseline": "heldout_code_refine_n342_seed123.json",
        "method": "heldout_isr_n342_seed123.json",
        "text_ref": "Held-out audit paragraph",
    },
]


def load_results(results_dir, spec):
    path_text, _, nested = spec.partition("::")
    path = results_dir / path_text
    data = json.loads(path.read_text())
    if nested:
        data = data[nested]
    rows = data["results"]
    out = {}
    for row in rows:
        if "passed" in row:
            passed = row["passed"]
        else:
            passed = row["final_passed"]
        out[row["task_id"]] = bool(passed)
    return out, path_text if not nested else f"{path_text}::{nested}"


def mcnemar_pvalue(gain, loss):
    discordant = gain + loss
    if discordant == 0:
        return 1.0
    chi_square = (abs(gain - loss) - 1) ** 2 / discordant
    return math.erfc(math.sqrt(chi_square / 2.0))


def percentile(values, pct):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    idx = (len(ordered) - 1) * pct / 100.0
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def paired_stats(baseline, method, *, bootstrap_samples, seed):
    keys = sorted(set(baseline) & set(method))
    if not keys:
        raise ValueError("no overlapping task ids")
    diffs = [int(method[k]) - int(baseline[k]) for k in keys]
    gain = sum(d == 1 for d in diffs)
    loss = sum(d == -1 for d in diffs)
    diff_pp = 100.0 * sum(diffs) / len(keys)

    rng = random.Random(seed)
    boot = []
    for _ in range(bootstrap_samples):
        sample_sum = sum(diffs[rng.randrange(len(diffs))] for _ in diffs)
        boot.append(100.0 * sample_sum / len(diffs))

    return {
        "n": len(keys),
        "baseline_passed": sum(1 for k in keys if baseline[k]),
        "method_passed": sum(1 for k in keys if method[k]),
        "gain": gain,
        "loss": loss,
        "net": gain - loss,
        "diff_pp": diff_pp,
        "mcnemar_p": mcnemar_pvalue(gain, loss),
        "bootstrap_ci_low_pp": percentile(boot, 2.5),
        "bootstrap_ci_high_pp": percentile(boot, 97.5),
    }


def fmt_p(value):
    if value == 0:
        return "0"
    if value < 1e-3:
        return f"{value:.2e}"
    return f"{value:.3f}"


def write_markdown(path, report):
    lines = [
        "# ISR Statistics Audit",
        "",
        "Generated from local JSON result files. McNemar p-values use the",
        "continuity-corrected chi-square form; confidence intervals are",
        f"bootstrap percentile intervals with {report['bootstrap_samples']} resamples.",
        "",
        "| Comparison | Sources | Passed | Gains/Losses | Diff pp | McNemar p | 95% CI pp | Text use |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in report["comparisons"]:
        stats = item["stats"]
        lines.append(
            "| {name} | {baseline} -> {method} | {bp}/{n} -> {mp}/{n} | "
            "{gain}/{loss} | {diff:.1f} | {p} | [{lo:.1f}, {hi:.1f}] | {text_ref} |".format(
                name=item["name"],
                baseline=item["baseline_source"],
                method=item["method_source"],
                bp=stats["baseline_passed"],
                mp=stats["method_passed"],
                n=stats["n"],
                gain=stats["gain"],
                loss=stats["loss"],
                diff=stats["diff_pp"],
                p=fmt_p(stats["mcnemar_p"]),
                lo=stats["bootstrap_ci_low_pp"],
                hi=stats["bootstrap_ci_high_pp"],
                text_ref=item["text_ref"],
            )
        )
    lines.append("")
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    comparisons = []
    for comp in DEFAULT_COMPARISONS:
        baseline, baseline_source = load_results(args.results_dir, comp["baseline"])
        method, method_source = load_results(args.results_dir, comp["method"])
        stats = paired_stats(
            baseline,
            method,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        comparisons.append({
            **comp,
            "baseline_source": baseline_source,
            "method_source": method_source,
            "stats": stats,
        })

    report = {
        "results_dir": str(args.results_dir),
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "comparisons": comparisons,
    }
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    write_markdown(args.out_md, report)
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
