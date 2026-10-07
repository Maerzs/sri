"""
Compute a rule-based taxonomy and leakage audit for held-out ISR artifacts.

The audit is intentionally mechanical: it reproduces the saved triggered and
changed counts, links each artifact category to paired one-shot/ISR outcomes,
and checks whether final skills copy identifiers or trace-specific terms.
"""

import argparse
import ast
import difflib
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ONESHOT = SCRIPT_DIR / "results" / "heldout_oneshot_n342_seed123.json"
DEFAULT_ISR = SCRIPT_DIR / "results" / "heldout_isr_n342_seed123.json"

TAXONOMY_LABELS = [
    "return_contract_shape",
    "input_validation_exception",
    "library_api_usage",
    "dataframe_plot_contract",
    "numeric_randomness_state",
    "filesystem_network_subprocess",
    "over_specific_or_leaky",
    "irrelevant_or_noop",
    "other",
]

LABEL_HINTS = {
    "return_contract_shape": [
        "return", "returns", "tuple", "dict", "dictionary", "list", "dataframe",
        "series", "array", "shape", "columns", "keys", "json", "object", "axes",
        "axis", "type", "output", "contract",
    ],
    "input_validation_exception": [
        "valueerror", "typeerror", "keyerror", "attributeerror", "exception",
        "raise", "raises", "invalid", "empty", "none", "nan", "missing",
        "non-numeric", "non numeric", "validate", "validation", "edge case",
    ],
    "library_api_usage": [
        "beautifulsoup", "requests", "sklearn", "scikit", "linearregression",
        "kmeans", "standardscaler", "minmaxscaler", "numpy", "pandas",
        "scipy", "networkx", "folium", "seaborn", "regex", "datetime",
        "json.loads", "json.dumps", "api",
    ],
    "dataframe_plot_contract": [
        "dataframe", "pandas", "plot", "matplotlib", "seaborn", "axes", "axis",
        "ax.", "xlabel", "ylabel", "title", "ticks", "histogram", "hist",
        "scatter", "boxplot", "heatmap", "bar chart", "figure", "subplots",
        "legend",
    ],
    "numeric_randomness_state": [
        "random", "random_state", "seed", "deterministic", "np.random",
        "rng", "precision", "round", "decimal", "float", "tolerance",
        "sqrt", "format", "numeric", "integer", "probability", "sample",
    ],
    "filesystem_network_subprocess": [
        "file", "files", "directory", "path", "folder", "open(", "read",
        "write", "csv", "zip", "tar", "http", "url", "network", "request",
        "subprocess", "socket", "database", "sqlite", "download",
    ],
    "over_specific_or_leaky": [
        "test_case_", "test_", "bigcodebench/", "/tmp/", "assertionerror",
        "traceback", "expected", "exactly", "hard-code", "hardcode",
    ],
}

LABEL_PRIORITY = [
    "over_specific_or_leaky",
    "dataframe_plot_contract",
    "filesystem_network_subprocess",
    "input_validation_exception",
    "return_contract_shape",
    "numeric_randomness_state",
    "library_api_usage",
]

STOP_TERMS = {
    "stdout", "stderr", "failed", "traceback", "file", "line", "self", "true",
    "false", "none", "exit_code", "tmp", "main", "testcases", "task_func",
    "assertionerror", "attributeerror", "valueerror", "typeerror", "keyerror",
    "error", "fail", "ran", "seconds",
}


def load_json(path):
    return json.loads(path.read_text())


def load_results(path):
    data = load_json(path)
    rows = data["results"] if isinstance(data, dict) and "results" in data else data
    return data, {row["task_id"]: row for row in rows}


def load_bigcodebench_metadata(metadata_jsonl=None, metadata_split="test"):
    if metadata_jsonl:
        rows = []
        with metadata_jsonl.open() as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    else:
        sys.path.insert(0, str(SCRIPT_DIR))
        from run_experiment import load_bigcodebench, split_data

        train, test = split_data(load_bigcodebench(), seed=123)
        if metadata_split == "train":
            rows = train
        elif metadata_split == "all":
            rows = train + test
        else:
            rows = test
    return {row["task_id"]: row for row in rows}


def text_blob(*parts):
    out = []
    for part in parts:
        if part is None:
            continue
        if isinstance(part, (list, tuple)):
            out.extend(str(x) for x in part if x is not None)
        else:
            out.append(str(part))
    return "\n".join(out)


def clean_text(text):
    text = text or ""
    text = re.sub(r"/tmp/[A-Za-z0-9_./-]+", " ", text)
    text = re.sub(r'File "[^"]+", line \d+', " ", text)
    return text


def normalize(text):
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def diff_line_count(before, after):
    diff = difflib.ndiff((before or "").splitlines(), (after or "").splitlines())
    return sum(1 for line in diff if line.startswith("+ ") or line.startswith("- "))


def added_text(before, after):
    diff = difflib.ndiff((before or "").splitlines(), (after or "").splitlines())
    return "\n".join(line[2:] for line in diff if line.startswith("+ "))


def extract_test_methods(test_code):
    if not test_code:
        return []
    try:
        tree = ast.parse(test_code)
    except SyntaxError:
        return sorted(set(re.findall(r"def\s+(test[A-Za-z0-9_]+)\s*\(", test_code)))
    methods = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
            methods.append(node.name)
    return sorted(set(methods))


def extract_literals(text):
    text = clean_text(text or "")
    strings = set()
    for quoted in re.findall(r"""['"]([^'"\n]{2,80})['"]""", text):
        q = quoted.strip()
        if q and not q.startswith("/tmp/"):
            strings.add(q)
    numbers = set()
    for num in re.findall(r"(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?(?![A-Za-z_])", text):
        stripped = num.lstrip("+-")
        if len(stripped) <= 1:
            continue
        numbers.add(num)
    return sorted(strings), sorted(numbers)


def extract_terms(text):
    text = clean_text(text or "")
    terms = set()
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_]{3,}", text.lower()):
        if token not in STOP_TERMS and not token.startswith("tmp"):
            terms.add(token)
    return terms


def contained_terms(final_skill, candidates, limit=20):
    final_norm = normalize(final_skill)
    found = []
    for item in candidates:
        if not item:
            continue
        item_norm = normalize(str(item))
        if not item_norm:
            continue
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(item_norm)}(?![A-Za-z0-9_])", final_norm):
            found.append(str(item))
    return sorted(set(found), key=lambda x: (len(x), x))[:limit]


def leakage_checks(row, task_meta):
    final_skill = row.get("final_skill") or ""
    feedback_trace = text_blob(row.get("feedback_traces") or row.get("feedback_trace") or [])
    private_trace = row.get("private_trace") or ""
    public_task_text = text_blob(
        task_meta.get("instruct_prompt"),
        task_meta.get("code_prompt"),
        task_meta.get("libs"),
        row.get("initial_skill"),
        feedback_trace,
    )

    feedback_methods = row.get("feedback_methods") or []
    private_methods = row.get("private_methods") or []
    feedback_method_hits = contained_terms(final_skill, feedback_methods)
    private_method_hits = contained_terms(final_skill, private_methods)

    task_id = row["task_id"]
    task_id_candidates = [task_id]
    if "/" in task_id:
        task_id_candidates.append(task_id.rsplit("/", 1)[-1])
    task_id_hits = contained_terms(final_skill, task_id_candidates)

    fb_strings, fb_numbers = extract_literals(feedback_trace)
    private_strings, private_numbers = extract_literals(private_trace)
    feedback_string_hits = contained_terms(final_skill, fb_strings)
    feedback_number_hits = contained_terms(final_skill, fb_numbers)
    private_string_hits = contained_terms(final_skill, private_strings)
    private_number_hits = contained_terms(final_skill, private_numbers)

    public_terms = extract_terms(public_task_text)
    private_only_terms = sorted(extract_terms(private_trace) - public_terms)
    private_only_hits = contained_terms(final_skill, private_only_terms)

    return {
        "feedback_method_name_hits": feedback_method_hits,
        "private_method_name_hits": private_method_hits,
        "task_id_hits": task_id_hits,
        "feedback_trace_string_hits": feedback_string_hits,
        "feedback_trace_number_hits": feedback_number_hits,
        "private_trace_string_hits": private_string_hits,
        "private_trace_number_hits": private_number_hits,
        "private_trace_only_term_hits": private_only_hits,
        "has_feedback_method_copy": bool(feedback_method_hits),
        "has_private_method_copy": bool(private_method_hits),
        "has_task_id_copy": bool(task_id_hits),
        "has_feedback_trace_copy": bool(feedback_string_hits or feedback_number_hits),
        "has_private_trace_copy": bool(private_string_hits or private_number_hits),
        "has_private_trace_only_term_copy": bool(private_only_hits),
    }


def label_scores(row, task_meta, leak):
    initial = row.get("initial_skill") or ""
    final = row.get("final_skill") or ""
    trace = text_blob(row.get("feedback_traces") or row.get("feedback_trace") or [])
    task_text = text_blob(task_meta.get("instruct_prompt"), task_meta.get("code_prompt"), task_meta.get("libs"))
    added = added_text(initial, final)
    combined = normalize(text_blob(added, final, trace, task_text))

    scores = {}
    for label, hints in LABEL_HINTS.items():
        score = 0
        for hint in hints:
            hint_norm = hint.lower()
            if hint_norm in combined:
                score += 2 if hint_norm in normalize(added) else 1
        scores[label] = score

    libs = normalize(task_meta.get("libs"))
    if any(x in libs for x in ["pandas", "matplotlib", "seaborn", "plotly"]):
        scores["dataframe_plot_contract"] += 2
    if any(x in libs for x in ["requests", "bs4", "beautifulsoup", "pathlib", "os", "subprocess", "sqlite"]):
        scores["filesystem_network_subprocess"] += 2
    if any(x in libs for x in ["sklearn", "scipy", "numpy", "pandas", "bs4", "requests", "folium"]):
        scores["library_api_usage"] += 1
    if any(leak[key] for key in [
        "has_feedback_method_copy", "has_private_method_copy", "has_task_id_copy",
        "has_private_trace_only_term_copy",
    ]):
        scores["over_specific_or_leaky"] += 10
    return scores


def choose_label(row, task_meta, leak):
    triggered = row.get("rounds", 0) > 0
    changed = (row.get("initial_skill") or "") != (row.get("final_skill") or "")
    if not triggered or not changed:
        return "irrelevant_or_noop", {label: 0 for label in TAXONOMY_LABELS}

    scores = label_scores(row, task_meta, leak)
    if all(scores.get(label, 0) == 0 for label in LABEL_PRIORITY):
        return "other", scores
    for label in LABEL_PRIORITY:
        if scores.get(label, 0) == max(scores.values()) and scores[label] > 0:
            return label, scores
    return "other", scores


def paired_delta(oneshot_row, isr_row):
    one = bool(oneshot_row and oneshot_row.get("final_passed"))
    isr = bool(isr_row.get("final_passed"))
    if (not one) and isr:
        return "gain"
    if one and (not isr):
        return "loss"
    return "no_change"


def safe_rate(num, den):
    return num / den if den else 0.0


def outcome_stats(rows):
    total = len(rows)
    gains = sum(1 for row in rows if row["outcome_delta"] == "gain")
    losses = sum(1 for row in rows if row["outcome_delta"] == "loss")
    return {
        "tasks": total,
        "triggered": sum(row["triggered"] for row in rows),
        "changed": sum(row["changed"] for row in rows),
        "oneshot_final_passed": sum(row["oneshot_final_passed"] for row in rows),
        "isr_feedback_passed": sum(row["isr_feedback_passed"] for row in rows),
        "isr_private_passed": sum(row["isr_private_passed"] for row in rows),
        "isr_final_passed": sum(row["isr_final_passed"] for row in rows),
        "paired_gains_vs_oneshot": gains,
        "paired_losses_vs_oneshot": losses,
        "paired_net_vs_oneshot": gains - losses,
        "isr_final_pass_rate": safe_rate(sum(row["isr_final_passed"] for row in rows), total),
        "oneshot_final_pass_rate": safe_rate(sum(row["oneshot_final_passed"] for row in rows), total),
    }


def numeric_summary(values):
    values = list(values)
    if not values:
        return {"count": 0, "median": 0, "mean": 0, "min": 0, "max": 0}
    return {
        "count": len(values),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
    }


def build_report(oneshot_rows, isr_rows, metadata):
    details = []
    missing_metadata = []
    for task_id in sorted(isr_rows):
        row = isr_rows[task_id]
        one = oneshot_rows.get(task_id)
        task_meta = metadata.get(task_id, {})
        if not task_meta:
            missing_metadata.append(task_id)
        leak = leakage_checks(row, task_meta)
        label, scores = choose_label(row, task_meta, leak)
        initial = row.get("initial_skill") or ""
        final = row.get("final_skill") or ""
        triggered = row.get("rounds", 0) > 0
        changed = initial != final
        detail = {
            "task_id": task_id,
            "taxonomy_label": label,
            "taxonomy_scores": {k: scores.get(k, 0) for k in TAXONOMY_LABELS if k != "irrelevant_or_noop"},
            "outcome_delta": paired_delta(one, row),
            "triggered": triggered,
            "changed": changed,
            "rounds": row.get("rounds", 0),
            "oneshot_final_passed": bool(one and one.get("final_passed")),
            "isr_feedback_passed": bool(row.get("feedback_passed")),
            "isr_private_passed": bool(row.get("private_passed")),
            "isr_final_passed": bool(row.get("final_passed")),
            "feedback_methods": row.get("feedback_methods") or [],
            "private_methods": row.get("private_methods") or [],
            "retrieved_task_ids": row.get("retrieved_task_ids") or [],
            "diff_lines": diff_line_count(initial, final),
            "initial_skill_lines": len(initial.splitlines()),
            "final_skill_lines": len(final.splitlines()),
            "feedback_trace_chars": len(text_blob(row.get("feedback_traces") or [])),
            "private_trace_chars": len(row.get("private_trace") or ""),
            "leakage": leak,
            "initial_skill": initial,
            "final_skill": final,
            "feedback_trace": text_blob(row.get("feedback_traces") or row.get("feedback_trace") or []),
        }
        details.append(detail)

    by_label = {}
    label_rows = defaultdict(list)
    for row in details:
        label_rows[row["taxonomy_label"]].append(row)
    for label in TAXONOMY_LABELS:
        by_label[label] = outcome_stats(label_rows.get(label, []))

    changed_rows = [row for row in details if row["changed"]]
    triggered_rows = [row for row in details if row["triggered"]]
    leakage_counts = {}
    for key in [
        "has_feedback_method_copy",
        "has_private_method_copy",
        "has_task_id_copy",
        "has_feedback_trace_copy",
        "has_private_trace_copy",
        "has_private_trace_only_term_copy",
    ]:
        leakage_counts[key] = sum(1 for row in details if row["leakage"][key])

    return {
        "summary": {
            "total_tasks": len(details),
            "triggered_refinement": len(triggered_rows),
            "changed_skill": len(changed_rows),
            "unchanged_after_trigger": sum(1 for row in triggered_rows if not row["changed"]),
            "all_tasks": outcome_stats(details),
            "triggered_tasks": outcome_stats(triggered_rows),
            "changed_skill_tasks": outcome_stats(changed_rows),
            "taxonomy_labels": TAXONOMY_LABELS,
            "taxonomy_counts": dict(Counter(row["taxonomy_label"] for row in details)),
            "outcome_linked_counts_by_label": by_label,
            "leakage_counts": leakage_counts,
            "edit_diff_lines": numeric_summary(row["diff_lines"] for row in changed_rows),
            "feedback_trace_chars_changed": numeric_summary(row["feedback_trace_chars"] for row in changed_rows),
            "missing_metadata_task_ids": missing_metadata,
            "notes": [
                "Triggered refinement is rounds > 0 in the held-out ISR file.",
                "Changed skill is exact inequality between initial_skill and final_skill.",
                "Taxonomy labels are rule-based and single-label; use the CSV sheet for human validation.",
                "Private-trace-only terms are terms in private_trace absent from task context, initial skill, and feedback trace, then present in final_skill.",
            ],
        },
        "details": details,
    }


def write_markdown(path, report, sources):
    summary = report["summary"]

    def stats_row(label, stats):
        return (
            f"| {label} | {stats['tasks']} | {stats['triggered']} | {stats['changed']} | "
            f"{stats['oneshot_final_passed']} | {stats['isr_final_passed']} | "
            f"{stats['paired_gains_vs_oneshot']} | {stats['paired_losses_vs_oneshot']} | "
            f"{stats['paired_net_vs_oneshot']} |"
        )

    lines = [
        "# Artifact Taxonomy Audit",
        "",
        f"- One-shot source: `{sources['oneshot']}`",
        f"- ISR source: `{sources['isr']}`",
        f"- Tasks: {summary['total_tasks']}",
        f"- Triggered refinement: {summary['triggered_refinement']}",
        f"- Changed skill: {summary['changed_skill']}",
        f"- Paired gains/losses/net vs one-shot: "
        f"{summary['all_tasks']['paired_gains_vs_oneshot']}/"
        f"{summary['all_tasks']['paired_losses_vs_oneshot']}/"
        f"{summary['all_tasks']['paired_net_vs_oneshot']}",
        "",
        "## Outcome Counts",
        "",
        "| Subset | Tasks | Triggered | Changed | One-shot pass | ISR pass | Gains | Losses | Net |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        stats_row("All tasks", summary["all_tasks"]),
        stats_row("Triggered tasks", summary["triggered_tasks"]),
        stats_row("Changed-skill tasks", summary["changed_skill_tasks"]),
        "",
        "## Taxonomy",
        "",
        "| Label | Tasks | Triggered | Changed | One-shot pass | ISR pass | Gains | Losses | Net |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in TAXONOMY_LABELS:
        lines.append(stats_row(label, summary["outcome_linked_counts_by_label"][label]))

    leak = summary["leakage_counts"]
    lines.extend([
        "",
        "## Leakage Checks",
        "",
        f"- Feedback method-name copies: {leak['has_feedback_method_copy']}",
        f"- Private method-name copies: {leak['has_private_method_copy']}",
        f"- Task-id copies: {leak['has_task_id_copy']}",
        f"- Feedback trace string/number copies: {leak['has_feedback_trace_copy']}",
        f"- Private trace string/number copies: {leak['has_private_trace_copy']}",
        f"- Private-trace-only term copies: {leak['has_private_trace_only_term_copy']}",
        "",
        "## Edit Size",
        "",
        f"- Diff-line median/mean/min/max: {summary['edit_diff_lines']['median']}/"
        f"{summary['edit_diff_lines']['mean']:.1f}/"
        f"{summary['edit_diff_lines']['min']}/"
        f"{summary['edit_diff_lines']['max']}",
        "",
        "## Notes",
        "",
    ])
    lines.extend(f"- {note}" for note in summary["notes"])
    lines.append("")
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oneshot", type=Path, default=DEFAULT_ONESHOT)
    parser.add_argument("--isr", type=Path, default=DEFAULT_ISR)
    parser.add_argument("--metadata-jsonl", type=Path, default=None)
    parser.add_argument("--metadata-split", choices=["test", "train", "all"], default="test")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    _, oneshot_rows = load_results(args.oneshot)
    _, isr_rows = load_results(args.isr)
    metadata = load_bigcodebench_metadata(args.metadata_jsonl, args.metadata_split)
    report = build_report(oneshot_rows, isr_rows, metadata)
    report["sources"] = {
        "oneshot": str(args.oneshot),
        "isr": str(args.isr),
        "metadata_jsonl": str(args.metadata_jsonl) if args.metadata_jsonl else "bigcode/bigcodebench v0.1.2 via run_experiment.load_bigcodebench",
        "metadata_split": args.metadata_split,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    write_markdown(args.out_md, report, report["sources"])
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")
    print(
        "triggered={triggered_refinement} changed={changed_skill} gains={gains} losses={losses}".format(
            triggered_refinement=report["summary"]["triggered_refinement"],
            changed_skill=report["summary"]["changed_skill"],
            gains=report["summary"]["all_tasks"]["paired_gains_vs_oneshot"],
            losses=report["summary"]["all_tasks"]["paired_losses_vs_oneshot"],
        )
    )


if __name__ == "__main__":
    main()
