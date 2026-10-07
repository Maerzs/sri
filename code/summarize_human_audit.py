"""
Summarize one- or two-annotator human artifact audit labels.

The input is normally the CSV produced by build_human_audit_sheet.py after
humans fill label columns. One annotator is reported as a pilot. Two annotators
receive simple agreement and Cohen kappa for taxonomy/preferred when possible.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


TAXONOMY_COLUMNS = [
    ("human_taxonomy_a", "human_taxonomy_b"),
    ("taxonomy_a", "taxonomy_b"),
    ("annotator1_taxonomy", "annotator2_taxonomy"),
    ("taxonomy_annotator1", "taxonomy_annotator2"),
]

PREFERRED_COLUMNS = [
    ("human_preferred_a", "human_preferred_b"),
    ("preferred_a", "preferred_b"),
    ("annotator1_preferred", "annotator2_preferred"),
    ("preferred_annotator1", "preferred_annotator2"),
]

LEAKAGE_COLUMNS = [
    ("human_leakage_flag_a", "human_leakage_flag_b"),
    ("leakage_a", "leakage_b"),
    ("annotator1_leakage", "annotator2_leakage"),
    ("leakage_annotator1", "leakage_annotator2"),
]

SINGLE_COLUMNS = {
    "taxonomy": ["human_taxonomy", "taxonomy_label_human", "taxonomy"],
    "preferred": ["human_preferred", "preferred"],
    "leakage": ["human_leakage_flag", "leakage_flag"],
}

RATING_COLUMNS = {
    "clarity_initial": "clarity_initial_1_5",
    "clarity_final": "clarity_final_1_5",
    "trace_issue_addressed": "trace_issue_addressed_0_2",
    "general_not_test_specific": "general_not_test_specific_0_2",
    "reusable_for_similar_task": "reusable_for_similar_task_0_2",
    "harmful_or_misleading": "harmful_or_misleading_0_2",
}

RATING_PAIR_COLUMNS = {
    "clarity_initial": ("clarity_initial_1_5_a", "clarity_initial_1_5_b"),
    "clarity_final": ("clarity_final_1_5_a", "clarity_final_1_5_b"),
    "trace_issue_addressed": ("trace_issue_addressed_0_2_a", "trace_issue_addressed_0_2_b"),
    "general_not_test_specific": ("general_not_test_specific_0_2_a", "general_not_test_specific_0_2_b"),
    "reusable_for_similar_task": ("reusable_for_similar_task_0_2_a", "reusable_for_similar_task_0_2_b"),
    "harmful_or_misleading": ("harmful_or_misleading_0_2_a", "harmful_or_misleading_0_2_b"),
}

BOOL_COLUMNS = {
    "hardcoded_trace_constant": "hardcoded_trace_constant_bool",
    "mentions_feedback_or_private_test_name": "mentions_feedback_or_private_test_name_bool",
}

BOOL_PAIR_COLUMNS = {
    "hardcoded_trace_constant": ("hardcoded_trace_constant_bool_a", "hardcoded_trace_constant_bool_b"),
    "mentions_feedback_or_private_test_name": (
        "mentions_feedback_or_private_test_name_bool_a",
        "mentions_feedback_or_private_test_name_bool_b",
    ),
}


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def nonempty(value):
    return str(value or "").strip()


def find_pair(rows, candidates):
    if not rows:
        return None
    fields = set(rows[0].keys())
    for left, right in candidates:
        if left in fields and right in fields:
            if any(nonempty(row.get(left)) or nonempty(row.get(right)) for row in rows):
                return left, right
    return None


def find_single(rows, candidates):
    if not rows:
        return None
    fields = set(rows[0].keys())
    for col in candidates:
        if col in fields and any(nonempty(row.get(col)) for row in rows):
            return col
    return None


def simple_agreement(pairs):
    usable = [(a, b) for a, b in pairs if a and b]
    if not usable:
        return None
    return {
        "n": len(usable),
        "agreement": sum(1 for a, b in usable if a == b) / len(usable),
        "matches": sum(1 for a, b in usable if a == b),
    }


def cohen_kappa(pairs):
    usable = [(a, b) for a, b in pairs if a and b]
    n = len(usable)
    if n == 0:
        return None
    labels = sorted({x for pair in usable for x in pair})
    if len(labels) < 2:
        return None
    observed = sum(1 for a, b in usable if a == b) / n
    left = Counter(a for a, _ in usable)
    right = Counter(b for _, b in usable)
    expected = sum((left[label] / n) * (right[label] / n) for label in labels)
    if expected == 1:
        return None
    return (observed - expected) / (1 - expected)


def pair_report(rows, name, columns):
    pair_cols = find_pair(rows, columns)
    if not pair_cols:
        return None
    left, right = pair_cols
    pairs = [(nonempty(row.get(left)), nonempty(row.get(right))) for row in rows]
    usable = [(a, b) for a, b in pairs if a and b]
    return {
        "columns": [left, right],
        "n_labeled_pairs": len(usable),
        "annotator1_counts": dict(Counter(a for a, _ in usable)),
        "annotator2_counts": dict(Counter(b for _, b in usable)),
        "simple_agreement": simple_agreement(pairs),
        "cohen_kappa": cohen_kappa(pairs),
    }


def single_report(rows, name, columns):
    col = find_single(rows, columns)
    if not col:
        return None
    labels = [nonempty(row.get(col)) for row in rows if nonempty(row.get(col))]
    return {
        "column": col,
        "n_labeled": len(labels),
        "counts": dict(Counter(labels)),
    }


def numeric_column_report(rows, column):
    values = []
    for row in rows:
        raw = nonempty(row.get(column))
        if not raw:
            continue
        try:
            values.append(float(raw))
        except ValueError:
            continue
    if not values:
        return None
    return {
        "column": column,
        "n_labeled": len(values),
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


def numeric_pair_report(rows, columns):
    left, right = columns
    if not rows or left not in rows[0] or right not in rows[0]:
        return None
    pairs = []
    for row in rows:
        a_raw = nonempty(row.get(left))
        b_raw = nonempty(row.get(right))
        if not a_raw and not b_raw:
            continue
        try:
            a = float(a_raw) if a_raw else None
            b = float(b_raw) if b_raw else None
        except ValueError:
            continue
        pairs.append((a, b))
    usable = [(a, b) for a, b in pairs if a is not None and b is not None]
    if not usable:
        return None
    means = [(a + b) / 2 for a, b in usable]
    exact = sum(1 for a, b in usable if a == b)
    within_one = sum(1 for a, b in usable if abs(a - b) <= 1)
    return {
        "columns": [left, right],
        "n_labeled_pairs": len(usable),
        "mean": sum(means) / len(means),
        "min": min(means),
        "max": max(means),
        "exact_agreement": exact / len(usable),
        "within_one_agreement": within_one / len(usable),
    }


def bool_column_report(rows, column):
    labels = []
    for row in rows:
        raw = nonempty(row.get(column)).lower()
        if not raw:
            continue
        if raw in {"1", "true", "yes", "y"}:
            labels.append("true")
        elif raw in {"0", "false", "no", "n"}:
            labels.append("false")
        else:
            labels.append(raw)
    if not labels:
        return None
    return {
        "column": column,
        "n_labeled": len(labels),
        "counts": dict(Counter(labels)),
    }


def bool_pair_report(rows, columns):
    left, right = columns
    if not rows or left not in rows[0] or right not in rows[0]:
        return None
    pairs = []
    for row in rows:
        a = nonempty(row.get(left)).lower()
        b = nonempty(row.get(right)).lower()
        if not a and not b:
            continue
        pairs.append((normalize_bool_label(a), normalize_bool_label(b)))
    usable = [(a, b) for a, b in pairs if a and b]
    if not usable:
        return None
    return {
        "columns": [left, right],
        "n_labeled_pairs": len(usable),
        "annotator1_counts": dict(Counter(a for a, _ in usable)),
        "annotator2_counts": dict(Counter(b for _, b in usable)),
        "simple_agreement": simple_agreement(usable),
        "cohen_kappa": cohen_kappa(usable),
    }


def normalize_bool_label(raw):
    raw = nonempty(raw).lower()
    if raw in {"1", "true", "yes", "y"}:
        return "true"
    if raw in {"0", "false", "no", "n"}:
        return "false"
    return raw


def summarize(rows):
    taxonomy_pair = pair_report(rows, "taxonomy", TAXONOMY_COLUMNS)
    preferred_pair = pair_report(rows, "preferred", PREFERRED_COLUMNS)
    leakage_pair = pair_report(rows, "leakage", LEAKAGE_COLUMNS)
    taxonomy_single = single_report(rows, "taxonomy", SINGLE_COLUMNS["taxonomy"])
    preferred_single = single_report(rows, "preferred", SINGLE_COLUMNS["preferred"])
    leakage_single = single_report(rows, "leakage", SINGLE_COLUMNS["leakage"])
    ratings = {}
    for name, columns in RATING_PAIR_COLUMNS.items():
        item = numeric_pair_report(rows, columns)
        if item is not None:
            ratings[name] = item
    for name, column in RATING_COLUMNS.items():
        if name not in ratings and (item := numeric_column_report(rows, column)) is not None:
            ratings[name] = item

    boolean_flags = {}
    for name, columns in BOOL_PAIR_COLUMNS.items():
        item = bool_pair_report(rows, columns)
        if item is not None:
            boolean_flags[name] = item
    for name, column in BOOL_COLUMNS.items():
        if name not in boolean_flags and (item := bool_column_report(rows, column)) is not None:
            boolean_flags[name] = item

    has_any_single = any([taxonomy_single, preferred_single, leakage_single, ratings, boolean_flags])
    has_pair_labels = (
        (taxonomy_pair and taxonomy_pair.get("n_labeled_pairs", 0) > 0)
        or (preferred_pair and preferred_pair.get("n_labeled_pairs", 0) > 0)
        or (leakage_pair and leakage_pair.get("n_labeled_pairs", 0) > 0)
        or any("n_labeled_pairs" in item for item in ratings.values())
        or any("n_labeled_pairs" in item for item in boolean_flags.values())
    )
    if has_pair_labels:
        mode = "two_annotator"
    elif has_any_single:
        mode = "single_annotator_pilot"
    else:
        mode = "unlabeled_template"
    report = {
        "mode": mode,
        "rows": len(rows),
        "taxonomy": taxonomy_pair or taxonomy_single,
        "preferred": preferred_pair or preferred_single,
        "leakage": leakage_pair or leakage_single,
        "ratings": ratings,
        "boolean_flags": boolean_flags,
    }
    if mode == "unlabeled_template":
        report["notes"] = [
            "The CSV has the requested audit rows and rating columns, but no human labels have been filled yet.",
            "Fill one annotator column set to produce a pilot summary, or paired columns to compute agreement.",
        ]
    elif mode == "single_annotator_pilot":
        report["notes"] = [
            "Only one annotator column set was found; this is a pilot summary, not an inter-annotator reliability report.",
            "Add paired columns such as human_taxonomy_a/human_taxonomy_b and human_preferred_a/human_preferred_b to compute agreement.",
        ]
    else:
        report["notes"] = [
            "Agreement is computed only on rows where both annotators supplied a non-empty label.",
            "Cohen kappa is omitted when fewer than two labels are present or expected agreement is degenerate.",
        ]
    return report


def write_markdown(path, report, labels_path):
    lines = [
        "# Human Artifact Audit Summary",
        "",
        f"- Labels source: `{labels_path}`",
        f"- Rows: {report['rows']}",
        f"- Mode: {report['mode']}",
        "",
    ]
    for section in ["taxonomy", "preferred", "leakage"]:
        data = report.get(section)
        lines.extend([f"## {section.title()}", ""])
        if not data:
            lines.extend(["No labels found.", ""])
            continue
        if "n_labeled_pairs" in data:
            lines.append(f"- Columns: `{data['columns'][0]}`, `{data['columns'][1]}`")
            lines.append(f"- Labeled pairs: {data['n_labeled_pairs']}")
            agreement = data.get("simple_agreement")
            if agreement:
                lines.append(f"- Simple agreement: {agreement['agreement']:.3f} ({agreement['matches']}/{agreement['n']})")
            else:
                lines.append("- Simple agreement: not available")
            kappa = data.get("cohen_kappa")
            lines.append(f"- Cohen kappa: {kappa:.3f}" if kappa is not None else "- Cohen kappa: not available")
            lines.append(f"- Annotator 1 counts: {data['annotator1_counts']}")
            lines.append(f"- Annotator 2 counts: {data['annotator2_counts']}")
        else:
            lines.append(f"- Column: `{data['column']}`")
            lines.append(f"- Labeled rows: {data['n_labeled']}")
            lines.append(f"- Counts: {data['counts']}")
        lines.append("")
    lines.extend(["## Rating Columns", ""])
    if report.get("ratings"):
        for name, data in report["ratings"].items():
            if "n_labeled_pairs" in data:
                lines.append(
                    f"- {name}: paired mean {data['mean']:.3f}, range {data['min']:.0f}-{data['max']:.0f}, "
                    f"n={data['n_labeled_pairs']}, exact agreement {data['exact_agreement']:.3f}, "
                    f"within-one {data['within_one_agreement']:.3f} (`{data['columns'][0]}`, `{data['columns'][1]}`)"
                )
            else:
                lines.append(
                    f"- {name}: mean {data['mean']:.3f}, range {data['min']:.0f}-{data['max']:.0f}, "
                    f"n={data['n_labeled']} (`{data['column']}`)"
                )
    else:
        lines.append("No numeric rating labels found.")
    lines.extend(["", "## Boolean Flags", ""])
    if report.get("boolean_flags"):
        for name, data in report["boolean_flags"].items():
            if "n_labeled_pairs" in data:
                agreement = data.get("simple_agreement")
                agreement_text = (
                    f"{agreement['agreement']:.3f} ({agreement['matches']}/{agreement['n']})"
                    if agreement else "not available"
                )
                kappa = data.get("cohen_kappa")
                kappa_text = f"{kappa:.3f}" if kappa is not None else "not available"
                lines.append(
                    f"- {name}: A {data['annotator1_counts']}, B {data['annotator2_counts']}, "
                    f"agreement {agreement_text}, "
                    f"kappa {kappa_text} "
                    f"(n={data['n_labeled_pairs']}, `{data['columns'][0]}`, `{data['columns'][1]}`)"
                )
            else:
                lines.append(f"- {name}: {data['counts']} (n={data['n_labeled']}, `{data['column']}`)")
    else:
        lines.append("No boolean flag labels found.")
    lines.append("")
    lines.extend(["## Notes", ""])
    lines.extend(f"- {note}" for note in report.get("notes", []))
    lines.append("")
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    rows = read_csv(args.labels)
    report = summarize(rows)
    report["sources"] = {"labels": str(args.labels)}

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    write_markdown(args.out_md, report, args.labels)
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")
    print(f"mode={report['mode']} rows={report['rows']}")


if __name__ == "__main__":
    main()
