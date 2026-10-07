#!/usr/bin/env python3
"""EvalPlus held-out audit runner for ISR.

Base tests generate feedback traces. Plus tests are final-only audit tests and
are never passed into skill or code refinement prompts.
"""

import argparse
import base64
import json
import os
import pickle
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import evalplus
from evalplus.data import (
    get_human_eval_plus,
    get_human_eval_plus_hash,
    get_mbpp_plus,
    get_mbpp_plus_hash,
)
from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS

from run_experiment import (
    API_KEY,
    DATA_SEED,
    RESULTS_DIR,
    TOP_K,
    build_metadata,
    build_retriever,
    get_client,
    load_bigcodebench,
    parse_canonical_prefix_chars,
    refine_code,
    solve_task,
    split_data,
    synthesize_skill,
)


DATASETS = ("humaneval", "mbpp")
MODES = ("zeroshot", "oneshot", "simple_retry", "code_refine", "isr")
BCB_TRAIN_CACHE = RESULTS_DIR / "evalplus" / f"bigcodebench_train_seed{DATA_SEED}_minimal.json"


def load_evalplus(dataset):
    if dataset == "humaneval":
        problems = get_human_eval_plus()
    elif dataset == "mbpp":
        problems = get_mbpp_plus()
    else:
        raise ValueError(f"unsupported dataset: {dataset}")
    tasks = [normalize_task(dataset, p) for _, p in sorted(problems.items())]
    return problems, tasks


def get_evalplus_hash(dataset):
    if dataset == "humaneval":
        return get_human_eval_plus_hash()
    if dataset == "mbpp":
        return get_mbpp_plus_hash()
    raise ValueError(f"unsupported dataset: {dataset}")


def normalize_task(dataset, problem):
    prompt = problem["prompt"]
    entry_point = problem["entry_point"]
    return {
        "task_id": problem["task_id"],
        "dataset": dataset,
        "instruct_prompt": (
            "Complete the Python function so that it satisfies the prompt and robust EvalPlus-style checks. "
            "Return only a correct implementation."
        ),
        "code_prompt": prompt,
        "entry_point": entry_point,
        "canonical_solution": problem.get("canonical_solution", ""),
        "libs": infer_libs(prompt),
        "prompt": prompt,
    }


def infer_libs(prompt):
    libs = []
    for line in prompt.splitlines():
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            libs.append(stripped)
    return "\n".join(libs) if libs else "standard library"


def _pickle_b64(value):
    return base64.b64encode(pickle.dumps(value)).decode("ascii")


def run_eval_script(dataset, problem, code, inputs):
    payload = {
        "dataset": dataset,
        "prompt": problem["prompt"],
        "canonical_solution": problem["canonical_solution"],
        "candidate_code": code,
        "entry_point": problem["entry_point"],
        "inputs": inputs,
        "atol": problem.get("atol", 0),
    }
    script = r'''
import base64
import json
import math
import pickle
import sys
import traceback

import numpy as np

payload = pickle.loads(base64.b64decode(sys.stdin.read().encode("ascii")))
dataset = payload["dataset"]
entry_point = payload["entry_point"]
inputs = payload["inputs"]
atol = payload["atol"]

MBPP_OUTPUT_NOT_NONE_TASKS = {"check_str", "text_match_three", "text_starta_endb"}
MBPP_OUTPUT_SET_EQ_TASKS = {
    "similar_elements", "find_char_long", "common_in_nested_lists", "extract_singly",
    "larg_nnum", "intersection_array", "find_dissimilar", "Diff",
}

def _surface_Area(base_edge, height):
    slant_height = math.sqrt((base_edge / 2) ** 2 + height**2)
    return round(base_edge**2 + 4 * (base_edge * slant_height) / 2)

def _digit_distance_nums(num1, num2):
    a, b = str(num1), str(num2)
    n = max(len(a), len(b))
    return sum(abs(int(x) - int(y)) for x, y in zip(a.zfill(n), b.zfill(n)))

def _poly(xs, x):
    return sum([coeff * math.pow(x, i) for i, coeff in enumerate(xs)])

def is_floats(x):
    if isinstance(x, float):
        return True
    if isinstance(x, (list, tuple)) and x:
        return all(isinstance(i, float) for i in x)
    if isinstance(x, np.ndarray):
        return x.dtype == np.float64 or x.dtype == np.float32
    return False

def compare(out, exp, inp):
    try:
        exact_match = out == exp
        if isinstance(exact_match, np.ndarray):
            exact_match = bool(np.array_equal(out, exp))
        else:
            exact_match = bool(exact_match)
    except BaseException:
        exact_match = False
    if dataset == "mbpp":
        if entry_point == "are_equivalent":
            exact_match = bool(exact_match)
        elif entry_point == "sum_div":
            exact_match = exact_match or out == 0
        elif entry_point == "surface_Area":
            exact_match = exact_match or abs(out - _surface_Area(*inp)) <= atol
        elif entry_point == "digit_distance_nums":
            exact_match = exact_match or out == _digit_distance_nums(*inp)
        elif entry_point in MBPP_OUTPUT_SET_EQ_TASKS:
            try:
                exact_match = set(out) == set(exp)
            except BaseException:
                exact_match = False
        elif entry_point in MBPP_OUTPUT_NOT_NONE_TASKS:
            exact_match = out == exp if isinstance(out, bool) else exp == (out is not None)
    if dataset == "humaneval" and entry_point == "find_zero":
        return abs(_poly(*inp, out)) <= atol
    if atol == 0 and is_floats(exp):
        local_atol = 1e-6
    else:
        local_atol = atol
    if exact_match:
        return True
    if local_atol != 0:
        if type(out) != type(exp):
            return False
        if isinstance(exp, (list, tuple)) and len(out) != len(exp):
            return False
        return bool(np.allclose(out, exp, rtol=1e-07, atol=local_atol))
    return False

def load_fn(src):
    ns = {}
    exec(src, ns)
    return ns[entry_point]

try:
    ref_fn = load_fn(payload["prompt"] + "\n" + payload["canonical_solution"])
    cand_fn = load_fn(payload["prompt"] + "\n" + payload["candidate_code"])
    details = []
    errors = []
    for idx, inp in enumerate(inputs):
        try:
            exp = ref_fn(*inp)
            out = cand_fn(*inp)
            ok = compare(out, exp, inp)
            details.append(bool(ok))
            if not ok and len(errors) < 5:
                errors.append({"index": idx, "expected": repr(exp)[:300], "got": repr(out)[:300]})
        except BaseException as exc:
            details.append(False)
            if len(errors) < 5:
                errors.append({"index": idx, "error": repr(exc), "traceback": traceback.format_exc(limit=2)})
    print(json.dumps({"ok": all(details), "details": details, "errors": errors}))
except BaseException as exc:
    print(json.dumps({"ok": False, "details": [False for _ in inputs], "errors": [{"fatal": repr(exc), "traceback": traceback.format_exc(limit=3)}]}))
'''
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            input=_pickle_b64(payload),
            capture_output=True,
            text=True,
            timeout=max(10, min(60, len(inputs) // 10 + 10)),
            env={**os.environ, "MPLBACKEND": "Agg"},
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "details": [False for _ in inputs], "errors": [{"timeout": True}]}
    if proc.returncode != 0:
        return {
            "ok": False,
            "details": [False for _ in inputs],
            "errors": [{"returncode": proc.returncode, "stderr": proc.stderr[:1000]}],
        }
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        return {
            "ok": False,
            "details": [False for _ in inputs],
            "errors": [{"parse_error": proc.stdout[:1000], "stderr": proc.stderr[:1000]}],
        }


def make_trace(result, split):
    details = result.get("details") or []
    if result.get("ok") and all(details):
        return ""
    failed = [i for i, ok in enumerate(details) if not ok]
    shown = ",".join(str(i) for i in failed[:20])
    more = "" if len(failed) <= 20 else f",...(+{len(failed) - 20})"
    err = json.dumps(result.get("errors", [])[:3], ensure_ascii=False)[:1000]
    return f"{split}_status=fail; failed_input_indices=[{shown}{more}]; total_inputs={len(details)}; errors={err}"


def validate_base(dataset, problems, expected, task_id, code):
    result = run_eval_script(dataset, problems[task_id], code, problems[task_id]["base_input"])
    return bool(result.get("ok")), make_trace(result, "base")


def validate_plus(dataset, problems, expected, task_id, code):
    base_result = run_eval_script(dataset, problems[task_id], code, problems[task_id]["base_input"])
    plus_result = run_eval_script(dataset, problems[task_id], code, problems[task_id]["plus_input"])
    return bool(base_result.get("ok")), bool(plus_result.get("ok")), make_trace(plus_result, "plus")


def task_text(task):
    return "\n".join([task["instruct_prompt"], task["code_prompt"], task["libs"]])


def minimal_bcb_task(task):
    return {
        "task_id": task.get("task_id"),
        "instruct_prompt": task.get("instruct_prompt", ""),
        "code_prompt": task.get("code_prompt", ""),
        "libs": task.get("libs", ""),
        "canonical_solution": task.get("canonical_solution", ""),
    }


def load_train_pool_cached():
    if BCB_TRAIN_CACHE.exists():
        return json.loads(BCB_TRAIN_CACHE.read_text())
    all_bcb = load_bigcodebench()
    train, _ = split_data(all_bcb, seed=DATA_SEED)
    minimal = [minimal_bcb_task(task) for task in train]
    BCB_TRAIN_CACHE.parent.mkdir(parents=True, exist_ok=True)
    BCB_TRAIN_CACHE.write_text(json.dumps(minimal, indent=2, ensure_ascii=False))
    return minimal


def select_tasks(tasks, offset, limit, task_ids):
    if not task_ids:
        return tasks[offset: offset + limit]
    by_id = {task["task_id"]: task for task in tasks}
    selected = []
    missing = []
    for task_id in task_ids:
        if task_id in by_id:
            selected.append(by_id[task_id])
        else:
            missing.append(task_id)
    if missing:
        raise ValueError(f"Unknown task ids: {missing}")
    return selected


def run_task(client, dataset, problems, expected, task, train_pool, retriever, mode,
             max_rounds, canonical_prefix_chars, save_artifacts):
    started_at = time.time()
    problem = problems[task["task_id"]]
    record = {
        "task_id": task["task_id"],
        "dataset": dataset,
        "mode": mode,
        "rounds": 0,
        "trace_policy": "base tests only are visible to retry/refinement; plus tests are final-only",
        "base_input_count": len(problem["base_input"]),
        "plus_input_count": len(problem["plus_input"]),
    }

    if mode == "zeroshot":
        code = solve_task(client, task, skill=None)
        base_passed, base_trace = validate_base(dataset, problems, expected, task["task_id"], code)
        final_base_passed, plus_passed, plus_trace = validate_plus(dataset, problems, expected, task["task_id"], code)
        record.update({
            "base_passed": base_passed,
            "plus_passed": plus_passed,
            "final_passed": final_base_passed and plus_passed,
        })
        if save_artifacts:
            record.update({"initial_code": code, "base_trace": base_trace, "plus_trace_final_only": plus_trace})
        record["elapsed_seconds"] = time.time() - started_at
        return record

    retrieved = retriever.retrieve(task_text(task), train_pool, k=TOP_K)
    skill = synthesize_skill(client, task, retrieved, canonical_prefix_chars=canonical_prefix_chars)
    code = solve_task(client, task, skill=skill)
    base_passed, base_trace = validate_base(dataset, problems, expected, task["task_id"], code)

    final_skill = skill
    final_code = code
    skills = [skill]
    codes = [code]
    base_traces = [base_trace]

    if mode == "oneshot" or base_passed:
        pass
    elif mode == "simple_retry":
        code = solve_task(client, task, skill=skill)
        base_passed, base_trace = validate_base(dataset, problems, expected, task["task_id"], code)
        final_code = code
        codes.append(code)
        base_traces.append(base_trace)
        record["rounds"] = 1
    elif mode == "code_refine":
        for t in range(max_rounds):
            code = refine_code(client, task, skill, code, base_trace)
            base_passed, base_trace = validate_base(dataset, problems, expected, task["task_id"], code)
            final_code = code
            codes.append(code)
            base_traces.append(base_trace)
            record["rounds"] = t + 1
            if base_passed:
                break
    elif mode == "isr":
        for t in range(max_rounds):
            skill = synthesize_skill(
                client, task, retrieved, prev_skill=skill, failure_trace=base_trace,
                canonical_prefix_chars=canonical_prefix_chars,
            )
            code = solve_task(client, task, skill=skill)
            base_passed, base_trace = validate_base(dataset, problems, expected, task["task_id"], code)
            final_skill = skill
            final_code = code
            skills.append(skill)
            codes.append(code)
            base_traces.append(base_trace)
            record["rounds"] = t + 1
            if base_passed:
                break
    else:
        raise ValueError(f"unsupported mode: {mode}")

    final_base_passed, plus_passed, plus_trace = validate_plus(dataset, problems, expected, task["task_id"], final_code)
    record.update({
        "base_passed": final_base_passed,
        "plus_passed": plus_passed,
        "final_passed": final_base_passed and plus_passed,
    })
    if save_artifacts:
        record.update({
            "initial_skill": skills[0] if skills else None,
            "final_skill": final_skill,
            "skills": skills,
            "codes": codes,
            "base_traces": base_traces,
            "plus_trace_final_only": plus_trace,
            "retrieved_task_ids": [r["task_id"] for r in retrieved],
        })
    record["elapsed_seconds"] = time.time() - started_at
    return record


def error_record(task, dataset, mode, error):
    return {
        "task_id": task.get("task_id", "unknown"),
        "dataset": dataset,
        "mode": mode,
        "rounds": 0,
        "base_passed": False,
        "plus_passed": False,
        "final_passed": False,
        "error": str(error),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-rounds", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--save-artifacts", action="store_true")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--repeat-id", type=str, default=None)
    parser.add_argument("--canonical-prefix-chars", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--task-ids", type=str, default="")
    parser.add_argument("--task-timeout-seconds", type=int, default=0)
    args = parser.parse_args()

    canonical_prefix_chars = parse_canonical_prefix_chars(args.canonical_prefix_chars)
    if not API_KEY:
        print("ERROR: LLM_API_KEY not set")
        sys.exit(1)

    print(f"Loading EvalPlus {args.dataset}...", flush=True)
    problems, tasks = load_evalplus(args.dataset)
    expected = None
    task_ids = [item.strip() for item in args.task_ids.split(",") if item.strip()]
    target_subset = select_tasks(tasks, args.offset, args.limit, task_ids)

    print("Loading BigCodeBench train retrieval pool...", flush=True)
    train = load_train_pool_cached()
    retriever = build_retriever([task_text(t) for t in train])

    out_path = Path(args.out) if args.out else (
        RESULTS_DIR / "evalplus" / f"{args.dataset}_{args.mode}_n{len(target_subset)}.json"
    )

    def build_summary(results, complete):
        total = len(results)
        base_passed = sum(1 for r in results if r.get("base_passed"))
        plus_passed = sum(1 for r in results if r.get("plus_passed"))
        final_passed = sum(1 for r in results if r.get("final_passed"))
        return {
            "metadata": {
                **build_metadata(
                    f"evalplus_{args.dataset}_{args.mode}",
                    canonical_prefix_chars,
                    repeat_id=args.repeat_id,
                ),
                "complete": complete,
                "dataset": args.dataset,
                "benchmark_family": "EvalPlus",
                "evalplus_version": getattr(evalplus, "__version__", "unknown"),
                "evalplus_dataset_hash": get_evalplus_hash(args.dataset),
                "evaluator": "local differential checker with EvalPlus-compatible special oracles",
                "feedback_tests": "base_input",
                "final_private_tests": "plus_input",
                "plus_trace_visible_to_refinement": False,
                "target_total": len(target_subset),
                "target_offset": args.offset,
                "target_task_ids": task_ids,
                "concurrency": args.concurrency,
                "task_timeout_seconds": args.task_timeout_seconds,
            },
            "mode": args.mode,
            "total": total,
            "base_passed": base_passed,
            "base_pass_rate": base_passed / total if total else 0,
            "plus_passed": plus_passed,
            "plus_pass_rate": plus_passed / total if total else 0,
            "final_passed": final_passed,
            "final_pass_rate": final_passed / total if total else 0,
            "results": results,
        }

    def write_results(results, complete):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(build_summary(results, complete), f, indent=2, ensure_ascii=False)

    def maybe_checkpoint(results_by_index):
        if args.checkpoint_every <= 0:
            return
        completed = len(results_by_index)
        if completed and completed % args.checkpoint_every == 0:
            write_results([results_by_index[i] for i in sorted(results_by_index)], complete=False)

    def run_indexed(i, task):
        client = get_client()
        t0 = time.time()
        try:
            rec = run_task(
                client, args.dataset, problems, expected, task, train, retriever,
                args.mode, args.max_rounds, canonical_prefix_chars, args.save_artifacts,
            )
        except Exception as e:
            rec = error_record(task, args.dataset, args.mode, e)
        return i, task["task_id"], rec, time.time() - t0

    def run_indexed_with_timeout(i, task):
        if args.task_timeout_seconds <= 0:
            return run_indexed(i, task)

        def _timeout(_signum, _frame):
            raise TimeoutError(f"task timed out after {args.task_timeout_seconds}s")

        old_handler = signal.signal(signal.SIGALRM, _timeout)
        signal.alarm(args.task_timeout_seconds)
        try:
            return run_indexed(i, task)
        except Exception as e:
            rec = error_record(task, args.dataset, args.mode, e)
            return i, task["task_id"], rec, args.task_timeout_seconds
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    results_by_index = {}
    if args.concurrency <= 1:
        for i, task in enumerate(target_subset, start=1):
            i, tid, rec, elapsed = run_indexed_with_timeout(i, task)
            results_by_index[i] = rec
            mark = "✓" if rec.get("final_passed") else "✗"
            print(f"[{i}/{len(target_subset)}] {tid} {mark} base={rec.get('base_passed')} plus={rec.get('plus_passed')} rounds={rec.get('rounds')} [{elapsed:.1f}s]", flush=True)
            maybe_checkpoint(results_by_index)
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {executor.submit(run_indexed, i, task): (i, task) for i, task in enumerate(target_subset, start=1)}
            for fut in as_completed(futures):
                i, task = futures[fut]
                i, tid, rec, elapsed = fut.result()
                results_by_index[i] = rec
                mark = "✓" if rec.get("final_passed") else "✗"
                print(f"[{i}/{len(target_subset)}] {tid} {mark} base={rec.get('base_passed')} plus={rec.get('plus_passed')} rounds={rec.get('rounds')} [{elapsed:.1f}s]", flush=True)
                maybe_checkpoint(results_by_index)

    results = [results_by_index[i] for i in sorted(results_by_index)]
    write_results(results, complete=True)
    summary = build_summary(results, complete=True)
    print(f"Saved: {out_path}")
    print(f"Final pass: {summary['final_pass_rate']:.3f} ({summary['final_passed']}/{summary['total']}); base={summary['base_pass_rate']:.3f}; plus={summary['plus_pass_rate']:.3f}")


if __name__ == "__main__":
    main()
