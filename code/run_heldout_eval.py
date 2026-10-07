"""
Held-out feedback/final-test evaluation for ISR on BigCodeBench.

This runner splits each task's unittest methods into two disjoint groups:
- feedback tests: used to produce the trace available to repair methods
- private tests: never shown in traces; used only for final reporting

Final success requires passing BOTH feedback and private test methods. The split
is deterministic per task_id and seed. The script can also store per-task
artifacts (skills, traces, generated code) for auditability.
"""

import argparse
import ast
import copy
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_experiment import (
    API_KEY,
    BASE_URL,
    DATA_SEED,
    RESULTS_DIR,
    TOP_K,
    build_metadata,
    build_retriever,
    get_client,
    load_bigcodebench,
    parse_canonical_prefix_chars,
    split_data,
    solve_task,
    synthesize_skill,
    validate_code,
    refine_code,
)


def _stable_int(text):
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:12], 16)


def split_test_methods(test_code, task_id, seed=123, feedback_frac=0.5):
    """Return deterministic feedback/private method-name lists."""
    tree = ast.parse(test_code)
    methods = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "TestCases":
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name.startswith("test"):
                    methods.append(item.name)
            break
    if len(methods) < 2:
        raise ValueError(f"{task_id} has fewer than two test methods")
    methods = sorted(methods, key=lambda m: _stable_int(f"{seed}:{task_id}:{m}"))
    n_feedback = max(1, min(len(methods) - 1, round(len(methods) * feedback_frac)))
    feedback = sorted(methods[:n_feedback])
    private = sorted(methods[n_feedback:])
    return feedback, private


class FilterTestMethods(ast.NodeTransformer):
    def __init__(self, keep):
        self.keep = set(keep)

    def visit_ClassDef(self, node):
        if node.name != "TestCases":
            return node
        new_body = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name.startswith("test"):
                if item.name in self.keep:
                    new_body.append(item)
            else:
                new_body.append(item)
        if not new_body:
            new_body = [ast.Pass()]
        node.body = new_body
        return node


def subset_test_code(test_code, keep_methods):
    tree = ast.parse(test_code)
    tree = FilterTestMethods(keep_methods).visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def with_test_subset(task, keep_methods):
    task2 = copy.deepcopy(task)
    task2["test"] = subset_test_code(task["test"], keep_methods)
    return task2


def validate_on_subset(code, task, keep_methods):
    return validate_code(code, with_test_subset(task, keep_methods))


def clean_record(value):
    if value is None:
        return None
    return value if isinstance(value, (str, int, float, bool, list, dict)) else str(value)


def error_record(task, mode, error):
    return {
        "task_id": task.get("task_id", "unknown"),
        "mode": mode,
        "feedback_methods": [],
        "private_methods": [],
        "rounds": 0,
        "feedback_passed": False,
        "private_passed": False,
        "final_passed": False,
        "error": str(error),
    }


def run_task(client, task, pool, retriever, mode, max_rounds, split_seed, feedback_frac,
             save_artifacts, canonical_prefix_chars, exclude_retrieval_task_id=None):
    feedback_methods, private_methods = split_test_methods(
        task["test"], task["task_id"], seed=split_seed, feedback_frac=feedback_frac
    )
    record = {
        "task_id": task["task_id"],
        "mode": mode,
        "feedback_methods": feedback_methods,
        "private_methods": private_methods,
        "rounds": 0,
    }

    if mode == "zeroshot":
        code = solve_task(client, task, skill=None)
        feedback_passed, feedback_trace = validate_on_subset(code, task, feedback_methods)
        private_passed, private_trace = validate_on_subset(code, task, private_methods)
        record.update({
            "feedback_passed": feedback_passed,
            "private_passed": private_passed,
            "final_passed": feedback_passed and private_passed,
        })
        if save_artifacts:
            record.update({
                "initial_code": code,
                "feedback_trace": feedback_trace,
                "private_trace": private_trace,
            })
        return record

    query_text = f"{task['instruct_prompt']}\n{task['code_prompt']}\n{task['libs']}"
    retrieve_k = min(len(pool), TOP_K + 5) if exclude_retrieval_task_id else TOP_K
    retrieved = retriever.retrieve(query_text, pool, k=retrieve_k)
    if exclude_retrieval_task_id:
        retrieved = [r for r in retrieved if r.get("task_id") != exclude_retrieval_task_id][:TOP_K]
    else:
        retrieved = retrieved[:TOP_K]
    skill = synthesize_skill(
        client, task, retrieved, canonical_prefix_chars=canonical_prefix_chars
    )
    code = solve_task(client, task, skill=skill)
    feedback_passed, trace = validate_on_subset(code, task, feedback_methods)

    final_code = code
    final_skill = skill
    traces = [trace]
    skills = [skill]
    codes = [code]

    if mode == "oneshot" or feedback_passed:
        pass
    elif mode == "simple_retry":
        code = solve_task(client, task, skill=skill)
        feedback_passed, trace = validate_on_subset(code, task, feedback_methods)
        final_code = code
        codes.append(code)
        traces.append(trace)
        record["rounds"] = 1
    elif mode == "code_refine":
        for t in range(max_rounds):
            code = refine_code(client, task, skill, code, trace)
            feedback_passed, trace = validate_on_subset(code, task, feedback_methods)
            final_code = code
            codes.append(code)
            traces.append(trace)
            record["rounds"] = t + 1
            if feedback_passed:
                break
    elif mode == "isr":
        for t in range(max_rounds):
            skill = synthesize_skill(
                client, task, retrieved, prev_skill=skill, failure_trace=trace,
                canonical_prefix_chars=canonical_prefix_chars,
            )
            code = solve_task(client, task, skill=skill)
            feedback_passed, trace = validate_on_subset(code, task, feedback_methods)
            final_skill = skill
            final_code = code
            skills.append(skill)
            codes.append(code)
            traces.append(trace)
            record["rounds"] = t + 1
            if feedback_passed:
                break
    else:
        raise ValueError(f"unsupported mode: {mode}")

    private_passed, private_trace = validate_on_subset(final_code, task, private_methods)
    final_passed = feedback_passed and private_passed
    record.update({
        "feedback_passed": feedback_passed,
        "private_passed": private_passed,
        "final_passed": final_passed,
    })
    if save_artifacts:
        record.update({
            "initial_skill": skills[0] if skills else None,
            "final_skill": final_skill,
            "skills": skills,
            "codes": codes,
            "feedback_traces": traces,
            "private_trace": private_trace,
            "retrieved_task_ids": [r["task_id"] for r in retrieved],
        })
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["zeroshot", "oneshot", "simple_retry", "isr", "code_refine"], required=True)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-rounds", type=int, default=1)
    parser.add_argument("--split-seed", type=int, default=123)
    parser.add_argument("--feedback-frac", type=float, default=0.5)
    parser.add_argument("--eval-split", choices=["test", "train"], default="test",
                        help="Evaluate held-out protocol on the BigCodeBench test or train split. "
                             "When evaluating train tasks, the current task is excluded from retrieval.")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--save-artifacts", action="store_true")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--repeat-id", type=str, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=10,
                        help="Write partial JSON after every N completed tasks; 0 disables checkpointing.")
    parser.add_argument("--canonical-prefix-chars", type=int, default=None,
                        help="Chars of retrieved canonical_solution to include; "
                             "env CANONICAL_PREFIX_CHARS or default 400. Use 0 to omit code blocks.")
    args = parser.parse_args()
    canonical_prefix_chars = parse_canonical_prefix_chars(args.canonical_prefix_chars)

    if not API_KEY:
        print("ERROR: LLM_API_KEY not set")
        sys.exit(1)

    all_tasks = load_bigcodebench()
    train, test = split_data(all_tasks, seed=DATA_SEED)
    eval_pool = test if args.eval_split == "test" else train
    test_subset = eval_pool[args.offset:args.offset + args.limit]
    pool_texts = [f"{t['instruct_prompt']}\n{t['code_prompt']}\n{t['libs']}" for t in train]
    retriever = build_retriever(pool_texts)
    out_path = args.out or str(
        RESULTS_DIR / f"heldout_{args.mode}_n{len(test_subset)}_seed{args.split_seed}.json"
    )

    def build_summary(results, complete):
        total = len(results)
        final_passed = sum(1 for r in results if r["final_passed"])
        feedback_passed = sum(1 for r in results if r["feedback_passed"])
        private_passed = sum(1 for r in results if r["private_passed"])
        return {
            "metadata": {
                **build_metadata(
                    args.mode,
                    canonical_prefix_chars,
                    repeat_id=args.repeat_id,
                    split_seed=args.split_seed,
                ),
                "complete": complete,
                "eval_split": args.eval_split,
                "target_total": len(test_subset),
            },
            "mode": args.mode,
            "total": total,
            "final_passed": final_passed,
            "final_pass_rate": final_passed / total if total else 0,
            "feedback_passed": feedback_passed,
            "feedback_pass_rate": feedback_passed / total if total else 0,
            "private_passed": private_passed,
            "private_pass_rate": private_passed / total if total else 0,
            "max_rounds": args.max_rounds,
            "split_seed": args.split_seed,
            "feedback_frac": args.feedback_frac,
            "concurrency": args.concurrency,
            "results": results,
        }

    def write_results(results, complete):
        with open(out_path, "w") as f:
            json.dump(build_summary(results, complete), f, indent=2, ensure_ascii=False, default=clean_record)

    def maybe_checkpoint(results_by_index):
        if args.checkpoint_every <= 0:
            return
        completed = len(results_by_index)
        if completed and completed % args.checkpoint_every == 0:
            partial = [results_by_index[i] for i in sorted(results_by_index)]
            write_results(partial, complete=False)

    def run_indexed(i, task):
        client = get_client()
        t0 = time.time()
        try:
            rec = run_task(
                client, task, train, retriever, args.mode, args.max_rounds,
                args.split_seed, args.feedback_frac, args.save_artifacts,
                canonical_prefix_chars,
                exclude_retrieval_task_id=task["task_id"] if args.eval_split == "train" else None,
            )
        except Exception as e:
            rec = error_record(task, args.mode, e)
        return i, task["task_id"], rec, time.time() - t0

    results_by_index = {}
    if args.concurrency <= 1:
        for i, task in enumerate(test_subset, start=1):
            i, tid, rec, elapsed = run_indexed(i, task)
            results_by_index[i] = rec
            mark = "✓" if rec["final_passed"] else "✗"
            print(
                f"[{i}/{len(test_subset)}] {tid} {mark} "
                f"feedback={rec['feedback_passed']} private={rec['private_passed']} "
                f"rounds={rec['rounds']} [{elapsed:.1f}s]",
                flush=True,
            )
            maybe_checkpoint(results_by_index)
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [
                executor.submit(run_indexed, i, task)
                for i, task in enumerate(test_subset, start=1)
            ]
            for fut in as_completed(futures):
                i, tid, rec, elapsed = fut.result()
                results_by_index[i] = rec
                mark = "✓" if rec["final_passed"] else "✗"
                print(
                    f"[{i}/{len(test_subset)}] {tid} {mark} "
                    f"feedback={rec['feedback_passed']} private={rec['private_passed']} "
                    f"rounds={rec['rounds']} [{elapsed:.1f}s]",
                    flush=True,
                )
                maybe_checkpoint(results_by_index)

    results = [results_by_index[i] for i in sorted(results_by_index)]

    summary = build_summary(results, complete=True)
    write_results(results, complete=True)
    final_passed = summary["final_passed"]
    total = summary["total"]
    print(f"Saved: {out_path}")
    print(
        f"Final pass: {summary['final_pass_rate']:.3f} ({final_passed}/{total}); "
        f"feedback: {summary['feedback_pass_rate']:.3f}; "
        f"private: {summary['private_pass_rate']:.3f}"
    )


if __name__ == "__main__":
    main()
