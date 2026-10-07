"""
Cross-task skill reuse runner for BigCodeBench heldout ISR artifacts.

The source JSON is expected to be a run_heldout_eval.py output with saved
artifacts. Source skills are selected from records with rounds > 0,
private_passed == True, and both initial_skill/final_skill present.
"""

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_experiment import (  # noqa: E402
    API_KEY,
    DATA_SEED,
    RESULTS_DIR,
    TFIDFRetriever,
    build_metadata,
    get_client,
    parse_canonical_prefix_chars,
    load_bigcodebench,
    solve_task,
    split_data,
    synthesize_skill,
    validate_code,
)


MODES = ("refined", "initial", "random", "refined_synth", "initial_synth", "random_synth", "oneshot", "zeroshot")
LEAKAGE_KEYS = (
    "has_feedback_method_copy",
    "has_private_method_copy",
    "has_task_id_copy",
    "has_feedback_trace_copy",
    "has_private_trace_copy",
    "has_private_trace_only_term_copy",
)


def task_text(task):
    return "\n".join(
        [
            str(task.get("instruct_prompt", "")),
            str(task.get("code_prompt", "")),
            str(task.get("libs", "")),
        ]
    )


def load_source_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data, {}
    return data.get("results", []), {k: v for k, v in data.items() if k != "results"}


def load_taxonomy_filter(path):
    if not path:
        return {}, {}
    with open(path, "r") as f:
        data = json.load(f)
    details = data.get("details", [])
    by_task = {row["task_id"]: row for row in details if row.get("task_id")}
    blocked = {}
    for task_id, row in by_task.items():
        leak = row.get("leakage") or {}
        hits = [key for key in LEAKAGE_KEYS if leak.get(key)]
        if hits:
            blocked[task_id] = hits
    return by_task, blocked


def source_record_eligible(rec, policy):
    if rec.get("rounds", 0) <= 0:
        return False
    initial_skill = rec.get("initial_skill")
    final_skill = rec.get("final_skill")
    if not initial_skill or not final_skill:
        return False
    if policy == "strict":
        return rec.get("private_passed") is True
    if policy == "changed":
        return initial_skill != final_skill
    if policy == "strict-v2":
        return rec.get("private_passed") is True and initial_skill != final_skill
    raise ValueError(f"unsupported source pool policy: {policy}")


def build_source_pool(source_records, task_by_id, source_limit=None, taxonomy_by_task=None,
                      leakage_blocked=None, policy="strict"):
    taxonomy_by_task = taxonomy_by_task or {}
    leakage_blocked = leakage_blocked or {}
    pool = []
    skipped_leakage = 0
    for rec in source_records:
        tid = rec.get("task_id")
        if not tid or tid not in task_by_id:
            continue
        if not source_record_eligible(rec, policy):
            continue
        initial_skill = rec.get("initial_skill")
        final_skill = rec.get("final_skill")
        if not initial_skill or not final_skill:
            continue
        if tid in leakage_blocked:
            if policy == "strict-v2":
                hard_block = {
                    "has_private_method_copy",
                    "has_task_id_copy",
                    "has_private_trace_copy",
                    "has_private_trace_only_term_copy",
                }
                if not hard_block.intersection(leakage_blocked[tid]):
                    pass
                else:
                    skipped_leakage += 1
                    continue
            else:
                skipped_leakage += 1
                continue
        if tid in leakage_blocked and policy != "strict-v2":
            skipped_leakage += 1
            continue
        tax = taxonomy_by_task.get(tid) or {}
        pool.append(
            {
                "task_id": tid,
                "task": task_by_id[tid],
                "metadata_text": task_text(task_by_id[tid]),
                "initial_skill": initial_skill,
                "final_skill": final_skill,
                "rounds": rec.get("rounds"),
                "taxonomy_label": tax.get("taxonomy_label"),
                "outcome_delta": tax.get("outcome_delta"),
            }
        )
        if source_limit is not None and len(pool) >= source_limit:
            break
    return pool, skipped_leakage


def pick_retrieved_sources(target_task, source_pool, retriever, top_k=1):
    candidates = source_pool
    if len(source_pool) == 1 and source_pool[0]["task_id"] == target_task["task_id"]:
        return []

    # Retrieve a little deeper so same-task matches from heldout artifacts can
    # be skipped without rebuilding the vectorizer for every target.
    k = min(len(source_pool), max(10, top_k))
    while True:
        retrieved = retriever.retrieve(task_text(target_task), candidates, k=k)
        selected = [source for source in retrieved if source["task_id"] != target_task["task_id"]][:top_k]
        if len(selected) >= top_k or k >= len(source_pool):
            return selected
        if k >= len(source_pool):
            return selected
        k = min(len(source_pool), k * 2)


def pick_retrieved_source(target_task, source_pool, retriever):
    sources = pick_retrieved_sources(target_task, source_pool, retriever, top_k=1)
    return sources[0] if sources else None


def pick_random_source(target_task, source_pool, rng):
    candidates = [s for s in source_pool if s["task_id"] != target_task["task_id"]]
    if not candidates:
        return None
    return rng.choice(candidates)


def pick_random_sources(target_task, source_pool, rng, top_k=3):
    candidates = [s for s in source_pool if s["task_id"] != target_task["task_id"]]
    if not candidates:
        return []
    return rng.sample(candidates, k=min(top_k, len(candidates)))


def make_oneshot_evidence(target_task, train_pool, train_retriever):
    retrieved = train_retriever.retrieve(task_text(target_task), train_pool, k=3)
    return retrieved


def source_to_evidence(source, skill_kind):
    skill_text = source["final_skill"] if skill_kind == "final_skill" else source["initial_skill"]
    return {
        "task_id": source["task_id"],
        "instruct_prompt": source["task"].get("instruct_prompt", ""),
        "code_prompt": source["task"].get("code_prompt", ""),
        "libs": source["task"].get("libs", ""),
        "canonical_solution": (
            "Reusable skill guidance from a different source task:\n"
            + skill_text[:1200]
        ),
    }


def run_target(task, mode, source_pool, source_retriever, train_pool, train_retriever,
               seed, index, canonical_prefix_chars):
    client = get_client()
    source_task_id = None
    skill = None
    skill_kind = None

    if mode == "zeroshot":
        skill = None
    elif mode == "oneshot":
        retrieved = make_oneshot_evidence(task, train_pool, train_retriever)
        skill = synthesize_skill(
            client, task, retrieved, canonical_prefix_chars=canonical_prefix_chars
        )
        skill_kind = "synthesized"
    elif mode == "random":
        rng = random.Random(f"{seed}:{index}:{task['task_id']}")
        source = pick_random_source(task, source_pool, rng)
        if source is None:
            raise ValueError(f"no eligible random source for {task['task_id']}")
        source_task_id = source["task_id"]
        skill = source["final_skill"]
        skill_kind = "final_skill"
    elif mode in ("refined", "initial"):
        source = pick_retrieved_source(task, source_pool, source_retriever)
        if source is None:
            raise ValueError(f"no eligible retrieved source for {task['task_id']}")
        source_task_id = source["task_id"]
        if mode == "refined":
            skill = source["final_skill"]
            skill_kind = "final_skill"
        else:
            skill = source["initial_skill"]
            skill_kind = "initial_skill"
    elif mode in ("refined_synth", "initial_synth", "random_synth"):
        rng = random.Random(f"{seed}:{index}:{task['task_id']}")
        if mode == "random_synth":
            sources = pick_random_sources(task, source_pool, rng, top_k=3)
            skill_kind = "final_skill"
        else:
            sources = pick_retrieved_sources(task, source_pool, source_retriever, top_k=3)
            skill_kind = "final_skill" if mode == "refined_synth" else "initial_skill"
        if not sources:
            raise ValueError(f"no eligible source evidence for {task['task_id']}")
        source_task_id = ",".join(s["task_id"] for s in sources)
        evidence = [source_to_evidence(s, skill_kind) for s in sources]
        skill = synthesize_skill(
            client, task, evidence, canonical_prefix_chars=canonical_prefix_chars
        )
        skill_kind = f"{skill_kind}_synthesized"
    else:
        raise ValueError(f"unsupported mode: {mode}")

    code = solve_task(client, task, skill=skill)
    passed, trace = validate_code(code, task)
    result = {
        "task_id": task["task_id"],
        "source_task_id": source_task_id,
        "mode": mode,
        "passed": passed,
    }
    if skill_kind:
        result["skill_kind"] = skill_kind
    if not passed:
        result["failure_trace"] = trace
    return result


def run_target_timed(task, mode, source_pool, source_retriever, train_pool, train_retriever,
                     seed, index, canonical_prefix_chars):
    t0 = time.time()
    rec = run_target(
        task, mode, source_pool, source_retriever, train_pool, train_retriever,
        seed, index, canonical_prefix_chars,
    )
    return rec, time.time() - t0


def default_out_path(mode, total, seed):
    return RESULTS_DIR / f"cross_task_reuse_{mode}_n{total}_seed{seed}.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-json", required=True, help="Heldout ISR-style JSON with saved skills.")
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--source-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--repeat-id", type=str, default=None)
    parser.add_argument("--taxonomy-json", type=str, default=None,
                        help="Optional ARTIFACT_TAXONOMY_AUDIT.json; source tasks with leakage hits are skipped.")
    parser.add_argument("--source-pool-policy", choices=["strict", "changed", "strict-v2"], default="strict",
                        help="strict keeps rounds>0, private_passed=true, nonempty skills. "
                             "changed keeps any rounds>0 changed skill after leakage filtering; use only for exploratory diagnostics. "
                             "strict-v2 also requires changed skills and only hard-blocks private/task-id leakage.")
    parser.add_argument("--canonical-prefix-chars", type=int, default=None,
                        help="Recorded for metadata and used by one-shot synthesis via env/default.")
    parser.add_argument("--checkpoint-every", type=int, default=25,
                        help="Write partial JSON after every N completed tasks; 0 disables checkpointing.")
    args = parser.parse_args()
    canonical_prefix_chars = parse_canonical_prefix_chars(args.canonical_prefix_chars)

    if args.limit < 0:
        raise ValueError("--limit must be >= 0")
    if args.offset < 0:
        raise ValueError("--offset must be >= 0")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")
    if args.source_limit is not None and args.source_limit < 1:
        raise ValueError("--source-limit must be >= 1 when provided")
    if args.limit > 0 and not API_KEY:
        print("ERROR: LLM_API_KEY not set")
        sys.exit(1)

    print("Loading BigCodeBench...", flush=True)
    all_tasks = load_bigcodebench()
    train, test = split_data(all_tasks, seed=DATA_SEED)
    task_by_id = {t["task_id"]: t for t in all_tasks}

    source_records, source_metadata = load_source_json(args.source_json)
    taxonomy_by_task, leakage_blocked = load_taxonomy_filter(args.taxonomy_json)
    source_pool, skipped_leakage = build_source_pool(
        source_records,
        task_by_id,
        args.source_limit,
        taxonomy_by_task=taxonomy_by_task,
        leakage_blocked=leakage_blocked,
        policy=args.source_pool_policy,
    )
    if args.mode in ("refined", "initial", "random", "refined_synth", "initial_synth", "random_synth") and not source_pool:
        raise ValueError("source pool is empty after filtering")

    source_retriever = None
    if args.mode in ("refined", "initial", "refined_synth", "initial_synth"):
        source_retriever = TFIDFRetriever([s["metadata_text"] for s in source_pool])

    train_retriever = None
    if args.mode == "oneshot":
        train_retriever = TFIDFRetriever([task_text(t) for t in train])

    test_subset = test[args.offset : args.offset + args.limit]
    out_path = Path(args.out) if args.out else default_out_path(args.mode, len(test_subset), args.seed)

    print(
        f"Mode={args.mode} targets={len(test_subset)} offset={args.offset} "
        f"source_pool={len(source_pool)} concurrency={args.concurrency}",
        flush=True,
    )

    def build_summary(results, complete):
        passed = sum(1 for r in results if r.get("passed"))
        total = len(results)
        return {
            "metadata": {
                **build_metadata(
                    f"cross_task_reuse_{args.mode}",
                    canonical_prefix_chars,
                    repeat_id=args.repeat_id,
                ),
                "complete": complete,
                "mode": args.mode,
                "source_json": args.source_json,
                "source_json_total_results": len(source_records),
                "source_json_metadata": source_metadata,
                "source_pool_size": len(source_pool),
                "source_pool_policy": args.source_pool_policy,
                "source_pool_skipped_leakage": skipped_leakage,
                "source_limit": args.source_limit,
                "taxonomy_json": args.taxonomy_json,
                "target_split": f"BigCodeBench test split from split_data(seed={DATA_SEED})",
                "target_offset": args.offset,
                "target_limit": args.limit,
                "target_total": len(test_subset),
                "total_completed": total,
                "seed": args.seed,
                "concurrency": args.concurrency,
                "passed": passed,
                "pass_rate": passed / total if total else 0,
            },
            "results": results,
        }

    def write_results(results, complete):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(build_summary(results, complete), f, indent=2, ensure_ascii=False)

    def maybe_checkpoint(results_by_index):
        if args.checkpoint_every <= 0:
            return
        completed = len(results_by_index)
        if completed and completed % args.checkpoint_every == 0:
            write_results([results_by_index[i] for i in sorted(results_by_index)], complete=False)

    results_by_index = {}
    if test_subset:
        if args.concurrency == 1:
            for i, task in enumerate(test_subset, start=1):
                t0 = time.time()
                try:
                    rec = run_target(
                        task,
                        args.mode,
                        source_pool,
                        source_retriever,
                        train,
                        train_retriever,
                        args.seed,
                        i + args.offset,
                        canonical_prefix_chars,
                    )
                except Exception as e:
                    rec = {
                        "task_id": task["task_id"],
                        "source_task_id": None,
                        "mode": args.mode,
                        "passed": False,
                        "error": str(e),
                    }
                results_by_index[i] = rec
                mark = "✓" if rec["passed"] else "✗"
                src = rec.get("source_task_id") or "-"
                print(f"[{i}/{len(test_subset)}] {task['task_id']} {mark} source={src} [{time.time() - t0:.1f}s]", flush=True)
                maybe_checkpoint(results_by_index)
        else:
            with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                futures = {
                    executor.submit(
                        run_target_timed,
                        task,
                        args.mode,
                        source_pool,
                        source_retriever,
                        train,
                        train_retriever,
                        args.seed,
                        i + args.offset,
                        canonical_prefix_chars,
                    ): (i, task)
                    for i, task in enumerate(test_subset, start=1)
                }
                for fut in as_completed(futures):
                    i, task = futures[fut]
                    try:
                        rec, elapsed = fut.result()
                    except Exception as e:
                        rec = {
                            "task_id": task["task_id"],
                            "source_task_id": None,
                            "mode": args.mode,
                            "passed": False,
                            "error": str(e),
                        }
                        elapsed = 0.0
                    results_by_index[i] = rec
                    mark = "✓" if rec["passed"] else "✗"
                    src = rec.get("source_task_id") or "-"
                    print(f"[{i}/{len(test_subset)}] {task['task_id']} {mark} source={src} [{elapsed:.1f}s]", flush=True)
                    maybe_checkpoint(results_by_index)

    results = [results_by_index[i] for i in sorted(results_by_index)]
    summary = build_summary(results, complete=True)
    write_results(results, complete=True)
    passed = summary["metadata"]["passed"]
    total = summary["metadata"]["total_completed"]
    print(f"Saved: {out_path}", flush=True)
    print(f"Pass rate: {summary['metadata']['pass_rate']:.3f} ({passed}/{total})", flush=True)


if __name__ == "__main__":
    main()
