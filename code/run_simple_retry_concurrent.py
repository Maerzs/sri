"""
Concurrent Simple Retry baseline.

Same as one-shot synthesis, but on failure: retry with the same skill
(no failure trace, no refinement). This isolates "extra attempt" gains
from "failure-guided refinement" gains.
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
            m = re.match(r"\s*\[\d+/\d+\]\s+(BigCodeBench/\d+)\.\.\.\s+(✓|✗|ERROR)", line)
            if not m:
                continue
            tid, mark = m.group(1), m.group(2)
            if mark == "✓":
                rm = re.search(r"round (\d+)", line)
                rounds = int(rm.group(1)) if rm else 0
                completed[tid] = {"passed": True, "rounds": rounds}
            else:
                completed[tid] = {"passed": False, "rounds": 0}
    return completed


def run_simple_retry(client, task, pool, retriever):
    """One-shot synthesis + simple retry on failure (NO failure trace, NO skill refinement)."""
    query_text = f"{task['instruct_prompt']}\n{task['code_prompt']}\n{task['libs']}"
    retrieved = retriever.retrieve(query_text, pool, k=TOP_K)
    skill = synthesize_skill(client, task, retrieved)

    code = solve_task(client, task, skill=skill)
    passed, trace = validate_code(code, task)
    if passed:
        return {"passed": True, "rounds": 0, "skill": skill}

    # Simple retry: same skill, regenerate code (temperature is 0, but solver
    # may still produce different output due to non-determinism on the API side;
    # the real ablation question is "does a 2nd attempt with same guidance help?")
    code = solve_task(client, task, skill=skill)
    passed, trace = validate_code(code, task)
    return {"passed": passed, "rounds": 1 if passed else 0, "skill": skill}


def run_with_lock(client_factory, task, pool, retriever, idx, total, log_fp):
    tid = task["task_id"]
    client = client_factory()
    try:
        t0 = time.time()
        r = run_simple_retry(client, task, pool, retriever)
        elapsed = time.time() - t0
        mark = f"✓ (round {r['rounds']})" if r["passed"] else "✗"
        line = f"  [{idx}/{total}] {tid}... {mark} [{elapsed:.1f}s]"
        with print_lock:
            print(line, flush=True)
            log_fp.write(line + "\n")
            log_fp.flush()
        return {"task_id": tid, "passed": r["passed"], "rounds": r["rounds"]}
    except Exception as e:
        line = f"  [{idx}/{total}] {tid}... ERROR: {e}"
        with print_lock:
            print(line, flush=True)
            log_fp.write(line + "\n")
            log_fp.flush()
        return {"task_id": tid, "passed": False, "error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--log", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    if not API_KEY:
        print("ERROR: LLM_API_KEY not set"); sys.exit(1)

    log_path = args.log or str(RESULTS_DIR / "fullrun_simple_retry.log")
    out_path = args.out or str(RESULTS_DIR / "simple_retry_full.json")

    print("Loading BigCodeBench...", flush=True)
    all_tasks = load_bigcodebench()
    train, test = split_data(all_tasks, seed=123)
    print(f"Train: {len(train)}, Test: {len(test)}", flush=True)

    print("Building TF-IDF retrieval index...", flush=True)
    pool_texts = [f"{t['instruct_prompt']}\n{t['code_prompt']}\n{t['libs']}" for t in train]
    retriever = TFIDFRetriever(pool_texts)
    print("Done.", flush=True)

    completed = parse_log(log_path)
    print(f"Already completed: {len(completed)}", flush=True)
    remaining = [t for t in test if t["task_id"] not in completed]
    total = len(test)
    print(f"Remaining: {len(remaining)} / {total}", flush=True)

    if remaining:
        starting_idx = len(completed) + 1
        client_factory = lambda: OpenAI(base_url=BASE_URL, api_key=API_KEY)

        with open(log_path, "a") as log_fp:
            log_fp.write(f"\n=== START simple_retry concurrency={args.concurrency} ===\n")
            log_fp.flush()
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futures = [pool.submit(run_with_lock, client_factory, task, train, retriever,
                                        starting_idx + i, total, log_fp)
                           for i, task in enumerate(remaining)]
                for fut in as_completed(futures):
                    fut.result()

    final = parse_log(log_path)
    full_results = []
    passed_count = 0
    recovered = 0
    for t in test:
        tid = t["task_id"]
        if tid in final:
            r = final[tid]
            full_results.append({"task_id": tid, "passed": r["passed"], "rounds": r["rounds"]})
            if r["passed"]:
                passed_count += 1
                if r["rounds"] > 0:
                    recovered += 1
        else:
            full_results.append({"task_id": tid, "passed": False, "error": "not_run"})

    pass_rate = passed_count / total
    print("─" * 50)
    print(f"Mode: simple_retry")
    print(f"Pass@1: {pass_rate:.4f} ({passed_count}/{total})")
    print(f"Recovered by retry: {recovered}")
    print("─" * 50)
    with open(out_path, "w") as f:
        json.dump({"mode": "simple_retry", "total": total, "passed": passed_count,
                   "pass_rate": pass_rate, "recovered": recovered,
                   "concurrency": args.concurrency, "results": full_results}, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
