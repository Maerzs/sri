#!/usr/bin/env python3
"""Re-score saved EvalPlus artifacts with EvalPlus' official checker.

The online runner uses a lightweight local differential checker so feedback
traces can be generated during ISR. This script is an offline audit: it reads
the final code saved by the runner and evaluates it with EvalPlus
``check_correctness``. Set EVALPLUS_MAX_MEMORY_BYTES=-1 on macOS if resource
limits are incompatible with the local Python build.
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("EVALPLUS_MAX_MEMORY_BYTES", "-1")

from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash, get_mbpp_plus, get_mbpp_plus_hash
from evalplus.eval import PASS
from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
from evalplus.evaluate import check_correctness, get_groundtruth


def load_problems(dataset):
    if dataset == "humaneval":
        problems = get_human_eval_plus()
        expected = get_groundtruth(problems, get_human_eval_plus_hash(), [])
    elif dataset == "mbpp":
        problems = get_mbpp_plus()
        expected = get_groundtruth(problems, get_mbpp_plus_hash(), MBPP_OUTPUT_NOT_NONE_TASKS)
    else:
        raise ValueError(f"unsupported dataset: {dataset}")
    return problems, expected


def final_code(record):
    if record.get("codes"):
        return record["codes"][-1]
    if record.get("initial_code"):
        return record["initial_code"]
    if record.get("final_code"):
        return record["final_code"]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fast-check", action="store_true")
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text())
    dataset = data.get("metadata", {}).get("dataset") or data.get("dataset")
    problems, expected = load_problems(dataset)

    rows = []
    for idx, record in enumerate(data.get("results", [])):
        if args.limit is not None and idx >= args.limit:
            break
        code = final_code(record)
        if not code:
            rows.append({
                "task_id": record.get("task_id"),
                "official_base_passed": False,
                "official_plus_passed": False,
                "official_final_passed": False,
                "error": "missing saved final code",
            })
            continue
        task_id = record["task_id"]
        result = check_correctness(
            dataset,
            idx,
            problems[task_id],
            code,
            expected[task_id],
            base_only=False,
            fast_check=args.fast_check,
        )
        base_status, base_details = result["base"]
        plus_status, plus_details = result["plus"]
        rows.append({
            "task_id": task_id,
            "mode": record.get("mode"),
            "local_base_passed": bool(record.get("base_passed")),
            "local_plus_passed": bool(record.get("plus_passed")),
            "local_final_passed": bool(record.get("final_passed")),
            "official_base_passed": base_status == PASS,
            "official_plus_passed": plus_status == PASS,
            "official_final_passed": base_status == PASS and plus_status == PASS,
            "official_base_details_count": len(base_details) if base_details is not None else 0,
            "official_plus_details_count": len(plus_details) if plus_details is not None else 0,
        })
        print(
            f"[{idx + 1}] {task_id}: local={record.get('final_passed')} "
            f"official={rows[-1]['official_final_passed']}",
            flush=True,
        )

    local_passed = sum(bool(r.get("local_final_passed")) for r in rows)
    official_passed = sum(bool(r.get("official_final_passed")) for r in rows)
    disagreements = sum(bool(r.get("local_final_passed")) != bool(r.get("official_final_passed")) for r in rows)
    summary = {
        "input": args.input,
        "dataset": dataset,
        "mode": data.get("mode"),
        "total": len(rows),
        "local_final_passed": local_passed,
        "official_final_passed": official_passed,
        "disagreements": disagreements,
        "results": rows,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved: {args.out}")
    print(f"Official final pass: {official_passed}/{len(rows)}; disagreements={disagreements}")


if __name__ == "__main__":
    main()
