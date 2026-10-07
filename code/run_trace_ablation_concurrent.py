"""
Concurrent Trace ablation on FULL test split (342 tasks).

Compares 3 refinement signal modes (T=1):
- full: stdout + stderr + exit code (paper default)
- error_only: just the stderr error line
- generic: no failure info, just "the previous attempt failed, try again"

Each task is independent; we run all (task, mode) combinations
in a shared thread pool to maximize throughput.
"""

import os, sys, json, time, threading, argparse, re
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_experiment import (
    load_bigcodebench, split_data, TFIDFRetriever,
    solve_task, validate_code,
    RESULTS_DIR, API_KEY, BASE_URL, TOP_K,
    SYNTH_MODEL, TEMPERATURE,
)
# Import synthesize_skill from run_ablation (has trace_mode support)
from run_ablation import synthesize_skill as synthesize_skill_with_mode
from openai import OpenAI


print_lock = threading.Lock()
TRACE_MODES = ["full", "error_only", "generic"]


def parse_log(log_path):
    """Parse log lines like:  [42] full BigCodeBench/X... ✓"""
    if not os.path.exists(log_path):
        return {}  # (mode, tid) -> {"passed": bool}
    completed = {}
    with open(log_path) as f:
        for line in f:
            m = re.match(r"\s*\[\d+\]\s+(\w+)\s+(BigCodeBench/\d+)\.\.\.\s+(✓|✗|ERROR)", line)
            if not m:
                continue
            mode, tid, mark = m.group(1), m.group(2), m.group(3)
            completed[(mode, tid)] = {"passed": mark == "✓"}
    return completed


def run_one_trace(client, task, pool, retriever, mode):
    """Synthesize → solve → on fail, refine with trace_mode → re-solve."""
    query_text = f"{task['instruct_prompt']}\n{task['code_prompt']}\n{task['libs']}"
    retrieved = retriever.retrieve(query_text, pool, k=TOP_K)

    skill = synthesize_skill_with_mode(client, task, retrieved)
    code = solve_task(client, task, skill=skill)
    passed, trace = validate_code(code, task)
    if passed:
        return {"passed": True, "rounds": 0}

    # Refine with given trace_mode
    skill = synthesize_skill_with_mode(client, task, retrieved,
                                       prev_skill=skill, failure_trace=trace,
                                       trace_mode=mode)
    code = solve_task(client, task, skill=skill)
    passed, trace = validate_code(code, task)
    return {"passed": passed, "rounds": 1 if passed else 0}


def run_with_lock(client_factory, task, pool, retriever, mode, idx, total, log_fp):
    tid = task["task_id"]
    client = client_factory()
    try:
        t0 = time.time()
        r = run_one_trace(client, task, pool, retriever, mode)
        elapsed = time.time() - t0
        mark = f"✓" if r["passed"] else "✗"
        line = f"  [{idx}] {mode:10s} {tid}... {mark} [{elapsed:.1f}s]"
        with print_lock:
            print(line, flush=True)
            log_fp.write(line + "\n")
            log_fp.flush()
        return {"mode": mode, "task_id": tid, "passed": r["passed"], "rounds": r["rounds"]}
    except Exception as e:
        line = f"  [{idx}] {mode:10s} {tid}... ERROR: {e}"
        with print_lock:
            print(line, flush=True)
            log_fp.write(line + "\n")
            log_fp.flush()
        return {"mode": mode, "task_id": tid, "passed": False, "error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--log", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    if not API_KEY:
        print("ERROR: LLM_API_KEY not set"); sys.exit(1)

    log_path = args.log or str(RESULTS_DIR / "fullrun_trace_ablation.log")
    out_path = args.out or str(RESULTS_DIR / "trace_ablation_full.json")

    print("Loading BigCodeBench...", flush=True)
    all_tasks = load_bigcodebench()
    train, test = split_data(all_tasks, seed=123)
    print(f"Train: {len(train)}, Test: {len(test)}", flush=True)

    print("Building TF-IDF retrieval index...", flush=True)
    pool_texts = [f"{t['instruct_prompt']}\n{t['code_prompt']}\n{t['libs']}" for t in train]
    retriever = TFIDFRetriever(pool_texts)
    print("Done.", flush=True)

    completed = parse_log(log_path)
    print(f"Already completed: {len(completed)} / {len(test)*len(TRACE_MODES)}", flush=True)

    # Build remaining (mode, task) work list
    work = []
    for mode in TRACE_MODES:
        for t in test:
            if (mode, t["task_id"]) not in completed:
                work.append((mode, t))
    total_combinations = len(test) * len(TRACE_MODES)
    print(f"Remaining: {len(work)} / {total_combinations}", flush=True)

    if work:
        client_factory = lambda: OpenAI(base_url=BASE_URL, api_key=API_KEY)
        with open(log_path, "a") as log_fp:
            log_fp.write(f"\n=== START trace ablation concurrency={args.concurrency} ===\n")
            log_fp.flush()
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool_exec:
                futures = []
                for i, (mode, task) in enumerate(work):
                    futures.append(pool_exec.submit(
                        run_with_lock, client_factory, task, train, retriever,
                        mode, len(completed) + i + 1, total_combinations, log_fp,
                    ))
                for fut in as_completed(futures):
                    fut.result()

    # Aggregate
    final = parse_log(log_path)
    summary = {}
    n = len(test)
    for mode in TRACE_MODES:
        per_task = []
        passed = 0
        for t in test:
            tid = t["task_id"]
            key = (mode, tid)
            if key in final:
                r = final[key]
                per_task.append({"task_id": tid, "passed": r["passed"]})
                if r["passed"]:
                    passed += 1
            else:
                per_task.append({"task_id": tid, "passed": False, "error": "not_run"})
        summary[mode] = {
            "trace_mode": mode, "total": n, "passed": passed,
            "pass_rate": passed / n, "results": per_task,
        }

    print("\n" + "─" * 60)
    for mode in TRACE_MODES:
        s = summary[mode]
        print(f"  {mode:12s} Pass@1 = {s['pass_rate']:.4f} ({s['passed']}/{s['total']})")
    print("─" * 60)

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
