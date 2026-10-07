"""
Concurrent ISR runner with resume support.

Reads existing log to identify already-completed tasks, then runs the
remaining tasks with a thread pool. Appends to the existing log.

Usage:
    python run_experiment_concurrent.py --mode isr --concurrency 16
"""

import os
import sys
import json
import time
import random
import threading
import argparse
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Reuse all the logic from run_experiment.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_experiment import (
    load_bigcodebench, split_data, TFIDFRetriever, build_retriever,
    run_single_task, RESULTS_DIR, API_KEY, BASE_URL,
    DATA_SEED, SOLVER_MODEL, SYNTH_MODEL,
    build_metadata, parse_canonical_prefix_chars,
)
from openai import OpenAI


def parse_log(log_path):
    """Extract already-completed task_ids and their outcome from a log."""
    if not os.path.exists(log_path):
        return {}
    completed = {}  # task_id -> {"passed": bool, "rounds": int}
    with open(log_path) as f:
        for line in f:
            # Format: "  [N/342] BigCodeBench/X... ✓ (round R)" or "... ✗"
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


# Lock for printing & logging progress so threads don't interleave
print_lock = threading.Lock()


def run_with_lock(client_factory, task, pool, retriever, mode, max_rounds,
                  idx, total, log_fp, canonical_prefix_chars):
    """Run one task; print + log under lock."""
    tid = task["task_id"]
    client = client_factory()
    try:
        t0 = time.time()
        r = run_single_task(client, task, pool, retriever,
                            mode=mode, max_rounds=max_rounds,
                            canonical_prefix_chars=canonical_prefix_chars)
        elapsed = time.time() - t0
        if r["passed"]:
            mark = f"✓ (round {r['rounds']})"
        else:
            mark = "✗"
        line = f"  [{idx}/{total}] {tid}... {mark} [{elapsed:.1f}s]"
        with print_lock:
            print(line, flush=True)
            log_fp.write(line + "\n")
            log_fp.flush()
        return {
            "task_id": tid,
            "passed": r["passed"],
            "rounds": r.get("rounds", 0),
        }
    except Exception as e:
        line = f"  [{idx}/{total}] {tid}... ERROR: {e}"
        with print_lock:
            print(line, flush=True)
            log_fp.write(line + "\n")
            log_fp.flush()
        return {"task_id": tid, "passed": False, "error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["zeroshot", "oneshot", "isr", "code_refine"], required=True)
    parser.add_argument("--max-rounds", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--log", type=str, default=None,
                        help="Existing log file to resume from / append to. "
                             "Default: results/fullrun_{mode}.log")
    parser.add_argument("--out", type=str, default=None,
                        help="Final JSON output. Default: results/{mode}_full.json")
    parser.add_argument("--repeat-id", type=str, default=None)
    parser.add_argument("--canonical-prefix-chars", type=int, default=None,
                        help="Chars of retrieved canonical_solution to include; "
                             "env CANONICAL_PREFIX_CHARS or default 400. Use 0 to omit code blocks.")
    args = parser.parse_args()
    canonical_prefix_chars = parse_canonical_prefix_chars(args.canonical_prefix_chars)

    if not API_KEY:
        print("ERROR: LLM_API_KEY not set")
        sys.exit(1)

    log_path = args.log or str(RESULTS_DIR / f"fullrun_{args.mode}.log")
    out_path = args.out or str(RESULTS_DIR / f"{args.mode}_full.json")

    # Load tasks
    print("Loading BigCodeBench...", flush=True)
    all_tasks = load_bigcodebench()
    train, test = split_data(all_tasks, seed=DATA_SEED)
    print(f"Train: {len(train)}, Test: {len(test)}", flush=True)

    import os as _os
    _kind = _os.environ.get("RETRIEVER", "tfidf")
    print(f"Building retrieval index (RETRIEVER={_kind})...", flush=True)
    pool_texts = [f"{t['instruct_prompt']}\n{t['code_prompt']}\n{t['libs']}" for t in train]
    retriever = build_retriever(pool_texts)
    print("Done.", flush=True)

    # Determine remaining tasks from log
    completed = parse_log(log_path)
    print(f"Already completed in {log_path}: {len(completed)} tasks", flush=True)

    remaining = [t for t in test if t["task_id"] not in completed]
    total = len(test)
    print(f"Remaining: {len(remaining)} / {total}", flush=True)

    if not remaining:
        print("Nothing to do; all done.", flush=True)
    else:
        # Build per-task index for nice numbering across concurrent runs
        starting_idx = len(completed) + 1

        def client_factory():
            return OpenAI(base_url=BASE_URL, api_key=API_KEY)

        with open(log_path, "a") as log_fp:
            log_fp.write(
                f"\n=== RESUMING with concurrency={args.concurrency} mode={args.mode} "
                f"max_rounds={args.max_rounds} canonical_prefix_chars={canonical_prefix_chars} "
                f"repeat_id={args.repeat_id} ===\n"
            )
            log_fp.flush()

            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futures = []
                for i, task in enumerate(remaining):
                    futures.append(pool.submit(
                        run_with_lock,
                        client_factory, task, train, retriever,
                        args.mode, args.max_rounds,
                        starting_idx + i, total, log_fp, canonical_prefix_chars,
                    ))
                # Wait for all
                new_results = []
                for fut in as_completed(futures):
                    new_results.append(fut.result())

    # Aggregate ALL results (existing log + new) into final JSON
    final_completed = parse_log(log_path)
    full_results = []
    passed_count = 0
    recovered_count = 0
    for t in test:
        tid = t["task_id"]
        if tid in final_completed:
            r = final_completed[tid]
            full_results.append({"task_id": tid, "passed": r["passed"], "rounds": r["rounds"]})
            if r["passed"]:
                passed_count += 1
                if r["rounds"] > 0:
                    recovered_count += 1
        else:
            full_results.append({"task_id": tid, "passed": False, "error": "not_run"})

    pass_rate = passed_count / total

    print("\n" + "─" * 50)
    print(f"Mode: {args.mode}")
    print(f"Pass@1: {pass_rate:.4f} ({passed_count}/{total})")
    if args.mode in ("isr", "code_refine"):
        failed_oneshot = total - (passed_count - recovered_count)
        print(f"Recovered by refinement: {recovered_count} / {failed_oneshot} initially-failed")
    print("─" * 50)

    with open(out_path, "w") as f:
        json.dump({
            "metadata": build_metadata(
                args.mode,
                canonical_prefix_chars,
                repeat_id=args.repeat_id,
            ),
            "mode": args.mode,
            "total": total,
            "passed": passed_count,
            "pass_rate": pass_rate,
            "recovered": recovered_count,
            "max_rounds": args.max_rounds,
            "concurrency": args.concurrency,
            "results": full_results,
        }, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
