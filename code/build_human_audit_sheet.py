"""
Build a blank human audit CSV from the artifact taxonomy report.

The sampler is deterministic and stratified by paired outcome against one-shot:
15 gains, 10 losses, and 15 no-change tasks for the default size of 40. If a
stratum is short, remaining rows are filled from the other available strata.
"""

import argparse
import csv
import json
import random
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TAXONOMY = SCRIPT_DIR.parent / "paper" / "ARTIFACT_TAXONOMY_AUDIT.json"

FIELDNAMES = [
    "task_id",
    "stratum",
    "taxonomy_label",
    "outcome_delta",
    "oneshot_final_passed",
    "isr_final_passed",
    "isr_feedback_passed",
    "isr_private_passed",
    "triggered",
    "changed",
    "rounds",
    "diff_lines",
    "initial_skill",
    "final_skill",
    "feedback_trace",
    "feedback_methods",
    "private_methods",
    "clarity_initial_1_5",
    "clarity_final_1_5",
    "trace_issue_addressed_0_2",
    "general_not_test_specific_0_2",
    "hardcoded_trace_constant_bool",
    "mentions_feedback_or_private_test_name_bool",
    "reusable_for_similar_task_0_2",
    "harmful_or_misleading_0_2",
    "human_taxonomy",
    "human_preferred",
    "human_preferred_initial_final_tie",
    "human_pilot_or_annotator_id",
    "human_leakage_flag",
    "human_notes",
]

CONTEXT_FIELDNAMES = FIELDNAMES[:17]
TWO_ANNOTATOR_FIELDNAMES = CONTEXT_FIELDNAMES + [
    "clarity_initial_1_5_a",
    "clarity_initial_1_5_b",
    "clarity_final_1_5_a",
    "clarity_final_1_5_b",
    "trace_issue_addressed_0_2_a",
    "trace_issue_addressed_0_2_b",
    "general_not_test_specific_0_2_a",
    "general_not_test_specific_0_2_b",
    "hardcoded_trace_constant_bool_a",
    "hardcoded_trace_constant_bool_b",
    "mentions_feedback_or_private_test_name_bool_a",
    "mentions_feedback_or_private_test_name_bool_b",
    "reusable_for_similar_task_0_2_a",
    "reusable_for_similar_task_0_2_b",
    "harmful_or_misleading_0_2_a",
    "harmful_or_misleading_0_2_b",
    "human_taxonomy_a",
    "human_taxonomy_b",
    "human_preferred_a",
    "human_preferred_b",
    "human_leakage_flag_a",
    "human_leakage_flag_b",
    "annotator_a_id",
    "annotator_b_id",
    "human_notes_a",
    "human_notes_b",
]

DEFAULT_TARGETS = {
    "gain": 15,
    "loss": 10,
    "no_change": 15,
}

TARGET_PRESETS = {
    40: DEFAULT_TARGETS,
    80: {
        "gain": 30,
        "loss": 20,
        "no_change": 30,
    },
}


def load_details(path):
    data = json.loads(path.read_text())
    return data["details"]


def compact_text(value):
    if value is None:
        return ""
    text = str(value)
    return text.replace("\r\n", "\n").replace("\r", "\n")


def sample_stratified(details, sample_size, seed):
    rng = random.Random(seed)
    by_stratum = {"gain": [], "loss": [], "no_change": []}
    for row in details:
        by_stratum[row.get("outcome_delta", "no_change")].append(row)
    for rows in by_stratum.values():
        rows.sort(key=lambda row: row["task_id"])
        rng.shuffle(rows)

    if sample_size in TARGET_PRESETS:
        targets = dict(TARGET_PRESETS[sample_size])
    else:
        scale = sample_size / 40
        targets = {
            key: int(round(value * scale))
            for key, value in DEFAULT_TARGETS.items()
        }
    drift = sample_size - sum(targets.values())
    if drift:
        for key in ["gain", "no_change", "loss"]:
            if drift == 0:
                break
            targets[key] += 1 if drift > 0 else -1
            drift += -1 if drift > 0 else 1

    selected = []
    selected_ids = set()
    for key in ["gain", "loss", "no_change"]:
        take = min(targets[key], len(by_stratum[key]))
        for row in by_stratum[key][:take]:
            selected.append((key, row))
            selected_ids.add(row["task_id"])

    if len(selected) < sample_size:
        remaining = []
        for key in ["gain", "loss", "no_change"]:
            for row in by_stratum[key]:
                if row["task_id"] not in selected_ids:
                    remaining.append((key, row))
        remaining.sort(key=lambda item: (item[0], item[1]["task_id"]))
        rng.shuffle(remaining)
        for item in remaining[: sample_size - len(selected)]:
            selected.append(item)
            selected_ids.add(item[1]["task_id"])

    selected.sort(key=lambda item: ({"gain": 0, "loss": 1, "no_change": 2}[item[0]], item[1]["task_id"]))
    return selected[:sample_size]


def to_csv_row(stratum, row):
    return {
        "task_id": row["task_id"],
        "stratum": stratum,
        "taxonomy_label": row.get("taxonomy_label", ""),
        "outcome_delta": row.get("outcome_delta", ""),
        "oneshot_final_passed": row.get("oneshot_final_passed", ""),
        "isr_final_passed": row.get("isr_final_passed", ""),
        "isr_feedback_passed": row.get("isr_feedback_passed", ""),
        "isr_private_passed": row.get("isr_private_passed", ""),
        "triggered": row.get("triggered", ""),
        "changed": row.get("changed", ""),
        "rounds": row.get("rounds", ""),
        "diff_lines": row.get("diff_lines", ""),
        "initial_skill": compact_text(row.get("initial_skill", "")),
        "final_skill": compact_text(row.get("final_skill", "")),
        "feedback_trace": compact_text(row.get("feedback_trace", "")),
        "feedback_methods": ";".join(row.get("feedback_methods") or []),
        "private_methods": ";".join(row.get("private_methods") or []),
        "clarity_initial_1_5": "",
        "clarity_final_1_5": "",
        "trace_issue_addressed_0_2": "",
        "general_not_test_specific_0_2": "",
        "hardcoded_trace_constant_bool": "",
        "mentions_feedback_or_private_test_name_bool": "",
        "reusable_for_similar_task_0_2": "",
        "harmful_or_misleading_0_2": "",
        "human_taxonomy": "",
        "human_preferred": "",
        "human_preferred_initial_final_tie": "",
        "human_pilot_or_annotator_id": "",
        "human_leakage_flag": "",
        "human_notes": "",
    }


def to_two_annotator_row(stratum, row):
    base = to_csv_row(stratum, row)
    return {field: base.get(field, "") for field in CONTEXT_FIELDNAMES} | {
        field: "" for field in TWO_ANNOTATOR_FIELDNAMES if field not in CONTEXT_FIELDNAMES
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--taxonomy-json", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--sample-size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--two-annotator", action="store_true",
                        help="Write a clean paired A/B label template with no single-annotator pilot fields.")
    args = parser.parse_args()

    details = load_details(args.taxonomy_json)
    selected = sample_stratified(details, args.sample_size, args.seed)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = TWO_ANNOTATOR_FIELDNAMES if args.two_annotator else FIELDNAMES
    with args.out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for stratum, row in selected:
            if args.two_annotator:
                writer.writerow(to_two_annotator_row(stratum, row))
            else:
                writer.writerow(to_csv_row(stratum, row))

    counts = {}
    for stratum, _ in selected:
        counts[stratum] = counts.get(stratum, 0) + 1
    print(f"Wrote {args.out_csv}")
    print("sample_counts=" + json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
