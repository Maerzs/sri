"""
Concurrent Trace ablation on FULL test split (342 tasks) - V2 OPTIMIZED.

Key optimization: each task runs the INITIAL synth+solve ONCE, then shares
the failure trace across all 3 trace modes. This cuts API calls by ~3x
vs naive implementation.

For each task:
  1. retrieve, synth, solve, validate -> initial result + failure_trace
  2. if initial passes: all 3 modes inherit the pass (no refine needed)
  3. if initial fails: refine 3 times (full / error_only / generic) and re-solve

Each task is processed atomically by one worker thread.
"""

import os, sys, json, time, threading, argparse, re
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_experiment import (
    load_bigcodebench, split_data, TFIDFRetriever,
    solve_task, validate_code,
    RESULTS_DIR, API_KEY, BASE_URL, TOP_K,
)
from run_ablation import synthesize_skill as synthesize_skill_with_mode
from openai import OpenAI


print_lock = threading.Lock()
TRACE_MODES = ["full", "error_only", "generic"]


def parse_log(log_path):
    """Lines: '  [42] BigCodeBench/X... full=✓ error_only=✗ generic=✓'."""
    if not os.path.exists(log_path):
        return {}
    completed = {}  # tid -> {"full": bool, "error_only": bool, "generic": bool}
    with open(log_path) as f:
        for line in f:
            m = re.match(r"\s*\[\d+\]\s+(BigCodeBench/\d+)\.\.\.\s+(.+)", line)
            if not m:
                continue
            tid = m.group(1)
            payload = m.group(2)
            modes_passed = {}
            for mode in TRACE_MODES:
                mm = re.search(rf"{mode}=(✓|✗|ERR)", payload)
                if mm:
                    modes_passed[mode] = (mm.group(1) == "✓")
            if len(modes_passed) == len(TRACE_MODES):
                completed[tid] = modes_passed
    return completed


def run_one_task_all_modes(client, task, pool, retriever):
    """
    Run synth + solve once. If pass: all modes inherit pass.
    If fail: try refining with each of the 3 modes from the SAME failure trace.
    """
    query_text = f"{task['instruct_prompt']}\n{task['code_prompt']}\n{task['libs']}"
    retrieved = retriever.retrieve(query_text, pool, k=TOP_K)

    init_skill = synthesize_skill_with_mode(client, task, retrieved)
    init_code = solve_task(client, task, skill=init_skill)
    init_passed, init_trace = validate_code(init_code, task)

    if init_passed:
        # All modes pass at round 0
        return {mode: True for mode in TRACE_MODES}

    # Initial failure → try each refinement mode independently
    results = {}
    for mode in TRACE_MODES:
        try:
            refined_skill = synthesize_skill_with_mode(
                client, task, retrieved,
                prev_skill=init_skill, failure_trace=init_trace,
                trace_mode=mode,
            )
            refined_code = solve_task(client, task, skill=refined_skill)
            passed, _ = validate_code(refined_code, task)
            results[mode] = passed
        except Exception as e:
            results[mode] = False
    return results


def run_with_lock(client_factory, task, pool, retriever, idx, total, log_fp):
    tid = task["task_id"]
    client = client_factory()
    try:
        t0 = time.time()
        modes_pass = run_one_task_all_modes(client, task, pool, retriever)
        elapsed = time.time() - t0
        marks = " ".join(f"{m}={'✓' if modes_pass[m] else '✗'}" for m in TRACE_MODES)
        line = f"  [{idx}] {tid}... {marks} [{elapsed:.1f}s]"
        with print_lock:
            print(line, flush=True)
            log_fp.write(line + "\n")
            log_fp.flush()
        return {"task_id": tid, **modes_pass}
    except Exception as e:
        marks = " ".join(f"{m}=ERR" for m in TRACE_MODES)
        line = f"  [{idx}] {tid}... {marks} ({e})"
        with print_lock:
            print(line, flush=True)
            log_fp.write(line + "\n")
            log_fp.flush()
        return {"task_id": tid, **{m: False for m in TRACE_MODES}, "error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--log", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    if not API_KEY:
        print("ERROR: LLM_API_KEY not set"); sys.exit(1)

    log_path = args.log or str(RESULTS_DIR / "fullrun_trace_ablation_v2.log")
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
    print(f"Already completed: {len(completed)} / {len(test)}", flush=True)
    remaining = [t for t in test if t["task_id"] not in completed]
    total = len(test)
    print(f"Remaining tasks: {len(remaining)} / {total}", flush=True)

    if remaining:
        client_factory = lambda: OpenAI(base_url=BASE_URL, api_key=API_KEY)
        with open(log_path, "a") as log_fp:
            log_fp.write(f"\n=== START trace ablation v2 concurrency={args.concurrency} ===\n")
            log_fp.flush()
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool_exec:
                futures = []
                for i, task in enumerate(remaining):
                    futures.append(pool_exec.submit(
                        run_with_lock, client_factory, task, train, retriever,
                        len(completed) + i + 1, total, log_fp,
                    ))
                for fut in as_completed(futures):
                    fut.result()

    final = parse_log(log_path)
    summary = {}
    for mode in TRACE_MODES:
        per_task = []
        passed = 0
        for t in test:
            tid = t["task_id"]
            if tid in final:
                p = final[tid][mode]
                per_task.append({"task_id": tid, "passed": p})
                if p:
                    passed += 1
            else:
                per_task.append({"task_id": tid, "passed": False, "error": "not_run"})
        summary[mode] = {
            "trace_mode": mode, "total": total, "passed": passed,
            "pass_rate": passed / total, "results": per_task,
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
