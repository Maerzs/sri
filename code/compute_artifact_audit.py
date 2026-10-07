"""
Summarize saved ISR skill artifacts from the held-out audit run.

This is a local evidence script for the manuscript's artifact-audit claims. It
does not judge human readability; it only reports which tasks actually triggered
skill refinement, whether the skill text changed, and how edited-task outcomes
compare with the held-out one-shot run by task id.
"""

import argparse
import difflib
import json
import statistics
from pathlib import Path


def load_results(path):
    data = json.loads(path.read_text())
    return data, {row["task_id"]: row for row in data["results"]}


def diff_line_count(before, after):
    diff = difflib.ndiff((before or "").splitlines(), (after or "").splitlines())
    return sum(1 for line in diff if line.startswith("+ ") or line.startswith("- "))


def summarize(oneshot_rows, isr_rows):
    rows = list(isr_rows.values())
    triggered = [row for row in rows if row.get("rounds", 0) > 0]
    changed = [
        row for row in triggered
        if (row.get("initial_skill") or "") != (row.get("final_skill") or "")
    ]
    unchanged = [row for row in triggered if row not in changed]

    def outcome_stats(subset):
        ids = [row["task_id"] for row in subset]
        gain = sum(
            (not oneshot_rows[task_id]["final_passed"]) and isr_rows[task_id]["final_passed"]
            for task_id in ids
            if task_id in oneshot_rows
        )
        loss = sum(
            oneshot_rows[task_id]["final_passed"] and (not isr_rows[task_id]["final_passed"])
            for task_id in ids
            if task_id in oneshot_rows
        )
        return {
            "tasks": len(subset),
            "feedback_passed": sum(row["feedback_passed"] for row in subset),
            "private_passed": sum(row["private_passed"] for row in subset),
            "final_passed": sum(row["final_passed"] for row in subset),
            "paired_gains_vs_oneshot": gain,
            "paired_losses_vs_oneshot": loss,
            "paired_net_vs_oneshot": gain - loss,
        }

    edit_sizes = [
        diff_line_count(row.get("initial_skill"), row.get("final_skill"))
        for row in changed
    ]
    initial_lines = [len((row.get("initial_skill") or "").splitlines()) for row in changed]
    final_lines = [len((row.get("final_skill") or "").splitlines()) for row in changed]
    trace_chars = [len("\n".join(row.get("feedback_traces") or [])) for row in changed]

    examples = [
        row["task_id"]
        for row in changed
        if (not oneshot_rows[row["task_id"]]["final_passed"]) and row["final_passed"]
    ][:20]

    def numeric_summary(values):
        if not values:
            return {"count": 0, "median": 0, "mean": 0, "min": 0, "max": 0}
        return {
            "count": len(values),
            "median": statistics.median(values),
            "mean": statistics.mean(values),
            "min": min(values),
            "max": max(values),
        }

    return {
        "all_tasks": outcome_stats(rows),
        "triggered_refinement": outcome_stats(triggered),
        "changed_skill": outcome_stats(changed),
        "unchanged_skill_after_refinement": outcome_stats(unchanged),
        "edit_diff_lines": numeric_summary(edit_sizes),
        "initial_skill_lines": numeric_summary(initial_lines),
        "final_skill_lines": numeric_summary(final_lines),
        "feedback_trace_chars": numeric_summary(trace_chars),
        "changed_gain_examples": examples,
        "notes": [
            "Triggered refinement means rounds > 0 in the held-out ISR run.",
            "Changed skill means initial_skill and final_skill differ exactly as saved.",
            "Paired gains/losses compare independent held-out ISR and held-out one-shot runs by task id.",
            "This audit measures saved artifact behavior, not human readability or reuse.",
        ],
    }


def write_markdown(path, report):
    def row(label, stats):
        return (
            f"| {label} | {stats['tasks']} | {stats['feedback_passed']} | "
            f"{stats['private_passed']} | {stats['final_passed']} | "
            f"{stats['paired_gains_vs_oneshot']} | {stats['paired_losses_vs_oneshot']} | "
            f"{stats['paired_net_vs_oneshot']} |"
        )

    lines = [
        "# ISR Artifact Audit",
        "",
        "Generated from `heldout_isr_n342_seed123.json` and",
        "`heldout_oneshot_n342_seed123.json`.",
        "",
        "| Subset | Tasks | Feedback pass | Private pass | Final pass | Gains vs one-shot | Losses vs one-shot | Net |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        row("All held-out ISR tasks", report["all_tasks"]),
        row("Triggered refinement", report["triggered_refinement"]),
        row("Changed skill", report["changed_skill"]),
        row("Unchanged after refinement", report["unchanged_skill_after_refinement"]),
        "",
        "## Edit Size",
        "",
        f"- Changed skills: {report['changed_skill']['tasks']}",
        f"- Diff-line median/mean/min/max: {report['edit_diff_lines']['median']}/"
        f"{report['edit_diff_lines']['mean']:.1f}/"
        f"{report['edit_diff_lines']['min']}/"
        f"{report['edit_diff_lines']['max']}",
        f"- Initial skill lines median/mean: {report['initial_skill_lines']['median']}/"
        f"{report['initial_skill_lines']['mean']:.1f}",
        f"- Final skill lines median/mean: {report['final_skill_lines']['median']}/"
        f"{report['final_skill_lines']['mean']:.1f}",
        f"- Feedback trace characters median/mean: {report['feedback_trace_chars']['median']}/"
        f"{report['feedback_trace_chars']['mean']:.1f}",
        "",
        "## Notes",
        "",
    ]
    lines.extend(f"- {note}" for note in report["notes"])
    lines.append("")
    lines.append("## Example Edited Gains")
    lines.append("")
    lines.append(", ".join(report["changed_gain_examples"]))
    lines.append("")
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oneshot", type=Path, default=Path("results/heldout_oneshot_n342_seed123.json"))
    parser.add_argument("--isr", type=Path, default=Path("results/heldout_isr_n342_seed123.json"))
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    _, oneshot_rows = load_results(args.oneshot)
    _, isr_rows = load_results(args.isr)
    report = summarize(oneshot_rows, isr_rows)
    report["sources"] = {
        "oneshot": str(args.oneshot),
        "isr": str(args.isr),
    }

    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    write_markdown(args.out_md, report)
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
