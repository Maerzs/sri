"""
Concurrent Rounds ablation on FULL test split (342 tasks).

T=0 corresponds to one-shot (we already have full data: 39.2% from oneshot_full.json).
T=1 corresponds to ISR default (we already have full data: 53.8% from isr_full.json).
This script runs ONLY T=2 and T=3 by default to save API calls.

Each (T, task) is independent in the sense that we run the full
pipeline up to T refinement rounds for each task.
"""

import os, sys, json, time, threading, argparse, re
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_experiment import (
    load_bigcodebench, split_data, TFIDFRetriever,
    synthesize_skill, solve_task, validate_code,
    RESULTS_DIR, API_KEY, BASE_URL, TOP_K,
)
from openai import OpenAI


print_lock = threading.Lock()


def parse_log(log_path):
    if not os.path.exists(log_path):
        return {}
    completed = {}
    with open(log_path) as f:
        for line in f:
            m = re.match(r"\s*\[\d+\]\s+T=(\d+)\s+(BigCodeBench/\d+)\.\.\.\s+(✓|✗|ERROR)", line)
            if not m:
                continue
            T, tid, mark = int(m.group(1)), m.group(2), m.group(3)
            if mark == "✓":
                rm = re.search(r"round (\d+)", line)
                rounds = int(rm.group(1)) if rm else 0
                completed[(T, tid)] = {"passed": True, "rounds": rounds}
            else:
                completed[(T, tid)] = {"passed": False, "rounds": 0}
    return completed


def run_with_T(client, task, pool, retriever, T):
    """Run task with up to T refinement rounds."""
    query_text = f"{task['instruct_prompt']}\n{task['code_prompt']}\n{task['libs']}"
    retrieved = retriever.retrieve(query_text, pool, k=TOP_K)
    skill = synthesize_skill(client, task, retrieved)

    code = solve_task(client, task, skill=skill)
    passed, trace = validate_code(code, task)
    rounds_used = 0

    if passed or T == 0:
        return {"passed": passed, "rounds": rounds_used}

    for t in range(T):
        skill = synthesize_skill(client, task, retrieved, prev_skill=skill, failure_trace=trace)
        code = solve_task(client, task, skill=skill)
        passed, trace = validate_code(code, task)
        rounds_used = t + 1
        if passed:
            return {"passed": True, "rounds": rounds_used}

    return {"passed": False, "rounds": rounds_used}


def run_with_lock(client_factory, task, pool, retriever, T, idx, total, log_fp):
    tid = task["task_id"]
    client = client_factory()
    try:
        t0 = time.time()
        r = run_with_T(client, task, pool, retriever, T)
        elapsed = time.time() - t0
        mark = f"✓ (round {r['rounds']})" if r["passed"] else "✗"
        line = f"  [{idx}] T={T} {tid}... {mark} [{elapsed:.1f}s]"
        with print_lock:
            print(line, flush=True)
            log_fp.write(line + "\n")
            log_fp.flush()
        return {"T": T, "task_id": tid, "passed": r["passed"], "rounds": r["rounds"]}
    except Exception as e:
        line = f"  [{idx}] T={T} {tid}... ERROR: {e}"
        with print_lock:
            print(line, flush=True)
            log_fp.write(line + "\n")
            log_fp.flush()
        return {"T": T, "task_id": tid, "passed": False, "error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--T-values", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--log", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    if not API_KEY:
        print("ERROR: LLM_API_KEY not set"); sys.exit(1)

    log_path = args.log or str(RESULTS_DIR / "fullrun_rounds_ablation.log")
    out_path = args.out or str(RESULTS_DIR / "rounds_ablation_full.json")

    print("Loading BigCodeBench...", flush=True)
    all_tasks = load_bigcodebench()
    train, test = split_data(all_tasks, seed=123)
    print(f"Train: {len(train)}, Test: {len(test)}", flush=True)

    print("Building TF-IDF retrieval index...", flush=True)
    pool_texts = [f"{t['instruct_prompt']}\n{t['code_prompt']}\n{t['libs']}" for t in train]
    retriever = TFIDFRetriever(pool_texts)
    print("Done.", flush=True)

    completed = parse_log(log_path)
    print(f"Already completed: {len(completed)} / {len(test)*len(args.T_values)}", flush=True)

    work = []
    for T in args.T_values:
        for t in test:
            if (T, t["task_id"]) not in completed:
                work.append((T, t))
    total_combos = len(test) * len(args.T_values)
    print(f"Remaining: {len(work)} / {total_combos}", flush=True)

    if work:
        client_factory = lambda: OpenAI(base_url=BASE_URL, api_key=API_KEY)
        with open(log_path, "a") as log_fp:
            log_fp.write(f"\n=== START rounds ablation T={args.T_values} concurrency={args.concurrency} ===\n")
            log_fp.flush()
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool_exec:
                futures = []
                for i, (T, task) in enumerate(work):
                    futures.append(pool_exec.submit(
                        run_with_lock, client_factory, task, train, retriever,
                        T, len(completed) + i + 1, total_combos, log_fp,
                    ))
                for fut in as_completed(futures):
                    fut.result()

    # Aggregate
    final = parse_log(log_path)
    summary = {}
    n = len(test)
    for T in args.T_values:
        per_task = []
        passed = 0
        recovered = 0
        for t in test:
            tid = t["task_id"]
            key = (T, tid)
            if key in final:
                r = final[key]
                per_task.append({"task_id": tid, "passed": r["passed"], "rounds": r["rounds"]})
                if r["passed"]:
                    passed += 1
                    if r["rounds"] > 0:
                        recovered += 1
            else:
                per_task.append({"task_id": tid, "passed": False, "error": "not_run"})
        summary[f"T={T}"] = {
            "max_rounds": T, "total": n, "passed": passed,
            "pass_rate": passed / n, "recovered": recovered, "results": per_task,
        }

    print("\n" + "─" * 60)
    for T in args.T_values:
        s = summary[f"T={T}"]
        print(f"  T={T} Pass@1 = {s['pass_rate']:.4f} ({s['passed']}/{s['total']})  recovered={s['recovered']}")
    print("─" * 60)

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
