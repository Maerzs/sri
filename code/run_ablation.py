"""
Run ISR ablation experiments on BigCodeBench.

Ablations:
1. Refinement rounds: T=0,1,2,3
2. Failure trace content: full, error-only, action-only
3. Retrieval size: k=1,3,5

Usage:
    python run_ablation.py --ablation rounds --limit 30
    python run_ablation.py --ablation trace --limit 30
    python run_ablation.py --ablation retrieval --limit 30
"""

import os
import sys
import json
import time
import random
import subprocess
import tempfile
import numpy as np
from pathlib import Path

# Load env

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

# --- Config ---
BASE_URL = os.environ.get("LLM_BASE_URL", "https://llm.invalley.co/v1")
API_KEY = os.environ.get("LLM_API_KEY", "")
SOLVER_MODEL = "deepseek-v4-pro"
SYNTH_MODEL = "deepseek-v4-pro"
TEMPERATURE = 0.0

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def get_client():
    return OpenAI(base_url=BASE_URL, api_key=API_KEY)


# --- Data Loading ---
def load_bigcodebench():
    from datasets import load_dataset
    ds = load_dataset("bigcode/bigcodebench", split="v0.1.2")
    return list(ds)


def split_data(tasks, seed=123):
    rng = random.Random(seed)
    indices = list(range(len(tasks)))
    rng.shuffle(indices)
    train = [tasks[i] for i in indices[:798]]
    test = [tasks[i] for i in indices[798:]]
    return train, test


# --- Retrieval ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine


class TFIDFRetriever:
    def __init__(self, corpus_texts):
        self.vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
        self.matrix = self.vectorizer.fit_transform(corpus_texts)

    def retrieve(self, query, pool, k=3):
        query_vec = self.vectorizer.transform([query])
        sims = sklearn_cosine(query_vec, self.matrix).flatten()
        top_idx = np.argsort(sims)[-k:][::-1]
        return [pool[i] for i in top_idx]


# --- Skill Synthesis ---
SYNTH_SYSTEM = """You synthesize coding skills into SKILL.md for a target coding task. Use retrieved evidence to identify reusable patterns and failure modes. Do NOT copy concrete solutions or test cases."""

SYNTH_TEMPLATE = """Create a SKILL.md for this coding task.

## Target Task
Instruction: {instruction}

Code prompt:
```python
{code_prompt}
```

Libraries: {libs}

## Retrieved Evidence
{evidence}

## Output Format
Produce a concise SKILL.md with:
- When to use
- Likely failure modes
- Recommended approach (general, not the exact solution)
- Checklist before submitting"""

REFINE_TEMPLATE_FULL = """The previous skill led to a failed execution. Refine it.

## Target Task
Instruction: {instruction}

Code prompt:
```python
{code_prompt}
```

Libraries: {libs}

## Previous Skill
{prev_skill}

## Failure Trace
{failure_trace}

## Instructions
1. Identify what went wrong based on the failure trace.
2. Produce a REFINED SKILL.md that fixes the issue.
Focus on the specific error, not generic advice."""

REFINE_TEMPLATE_ERROR_ONLY = """The previous skill led to a failed execution. Refine it.

## Target Task
Instruction: {instruction}

Code prompt:
```python
{code_prompt}
```

## Previous Skill
{prev_skill}

## Error Message
{error_msg}

## Instructions
Produce a REFINED SKILL.md that fixes the error above."""

REFINE_TEMPLATE_GENERIC = """The previous skill led to a failed execution. Try again.

## Target Task
Instruction: {instruction}

Code prompt:
```python
{code_prompt}
```

## Previous Skill
{prev_skill}

## Instructions
The previous attempt failed. Produce a better SKILL.md."""


def synthesize_skill(client, task, retrieved, prev_skill=None, failure_trace=None, trace_mode="full"):
    """Synthesize or refine a skill."""
    if prev_skill and failure_trace:
        if trace_mode == "full":
            user_msg = REFINE_TEMPLATE_FULL.format(
                instruction=task["instruct_prompt"][:1000],
                code_prompt=task["code_prompt"],
                libs=task["libs"],
                prev_skill=prev_skill,
                failure_trace=failure_trace[:1500]
            )
        elif trace_mode == "error_only":
            # Extract just the error message
            error_msg = ""
            if "stderr:" in failure_trace:
                error_msg = failure_trace.split("stderr:")[1][:500]
            elif "TIMEOUT" in failure_trace:
                error_msg = "Execution timed out"
            else:
                error_msg = failure_trace[:300]
            user_msg = REFINE_TEMPLATE_ERROR_ONLY.format(
                instruction=task["instruct_prompt"][:1000],
                code_prompt=task["code_prompt"],
                prev_skill=prev_skill,
                error_msg=error_msg
            )
        elif trace_mode == "generic":
            user_msg = REFINE_TEMPLATE_GENERIC.format(
                instruction=task["instruct_prompt"][:1000],
                code_prompt=task["code_prompt"],
                prev_skill=prev_skill
            )
        else:
            raise ValueError(f"Unknown trace_mode: {trace_mode}")
    else:
        evidence_parts = []
        for i, r in enumerate(retrieved):
            evidence_parts.append(
                f"### Example {i+1} (similar task)\n"
                f"Instruction: {r['instruct_prompt'][:200]}\n"
                f"Solution approach:\n```python\n{r['canonical_solution'][:400]}\n```\n"
            )
        user_msg = SYNTH_TEMPLATE.format(
            instruction=task["instruct_prompt"][:1000],
            code_prompt=task["code_prompt"],
            libs=task["libs"],
            evidence="\n".join(evidence_parts)
        )

    resp = client.chat.completions.create(
        model=SYNTH_MODEL,
        messages=[
            {"role": "system", "content": SYNTH_SYSTEM},
            {"role": "user", "content": user_msg}
        ],
        temperature=TEMPERATURE,
        max_tokens=1500
    )
    return resp.choices[0].message.content


# --- Solver ---
SOLVER_SYSTEM = """You are a Python code generation expert. Complete the given function. Output ONLY the complete Python code (including imports and function definition). No explanations, no markdown fences."""

SOLVER_TEMPLATE_WITH_SKILL = """Complete this Python function.

## Instruction
{instruction}

## Code to complete
```python
{code_prompt}
```

## Guidance (SKILL.md)
{skill}

## Rules
- Output ONLY valid Python code
- Include all necessary imports
- Include the function definition
- Do NOT include test code or example usage"""

SOLVER_TEMPLATE_NO_SKILL = """Complete this Python function.

## Instruction
{instruction}

## Code to complete
```python
{code_prompt}
```

## Rules
- Output ONLY valid Python code
- Include all necessary imports
- Include the function definition
- Do NOT include test code or example usage"""


def solve_task(client, task, skill=None):
    if skill:
        user_msg = SOLVER_TEMPLATE_WITH_SKILL.format(
            instruction=task["instruct_prompt"][:1500],
            code_prompt=task["code_prompt"],
            skill=skill[:1500]
        )
    else:
        user_msg = SOLVER_TEMPLATE_NO_SKILL.format(
            instruction=task["instruct_prompt"][:1500],
            code_prompt=task["code_prompt"]
        )

    resp = client.chat.completions.create(
        model=SOLVER_MODEL,
        messages=[
            {"role": "system", "content": SOLVER_SYSTEM},
            {"role": "user", "content": user_msg}
        ],
        temperature=TEMPERATURE,
        max_tokens=3000
    )
    code = resp.choices[0].message.content

    if "</think>" in code:
        code = code.split("</think>", 1)[-1].strip()

    if "```python" in code:
        parts = code.split("```python")
        code_block = parts[1] if len(parts) > 1 else parts[0]
        if "```" in code_block:
            code_block = code_block.split("```")[0]
        code = code_block.strip()
    elif "```" in code:
        parts = code.split("```")
        if len(parts) >= 2:
            code = parts[1].strip()

    return code


# --- Validation ---
def validate_code(code, task):
    test_code = task["test"]
    full_script = code + "\n\n" + test_code + "\n\n"
    full_script += "import unittest\n"
    full_script += "import sys\n"
    full_script += "loader = unittest.TestLoader()\n"
    full_script += "suite = loader.loadTestsFromTestCase(TestCases)\n"
    full_script += "runner = unittest.TextTestRunner(stream=sys.stderr, verbosity=0)\n"
    full_script += "result = runner.run(suite)\n"
    full_script += "if result.wasSuccessful():\n"
    full_script += '    print("PASSED")\n'
    full_script += "else:\n"
    full_script += '    failures = [str(f[1])[:200] for f in result.failures + result.errors]\n'
    full_script += '    print("FAILED: " + "; ".join(failures[:3]))\n'

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, dir='/tmp') as f:
        f.write(full_script)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, timeout=10,
            env={**os.environ, "PYTHONPATH": "", "MPLBACKEND": "Agg"}
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        passed = "PASSED" in stdout and result.returncode == 0
        failure_trace = ""
        if not passed:
            failure_trace = f"stdout: {stdout[:500]}\nstderr: {stderr[:500]}\nexit_code: {result.returncode}"

        return passed, failure_trace

    except subprocess.TimeoutExpired:
        return False, "TIMEOUT: Execution exceeded 10 seconds"
    except Exception as e:
        return False, f"EXECUTION ERROR: {str(e)}"
    finally:
        os.unlink(tmp_path)


# --- Ablation: Rounds ---
def run_rounds_ablation(client, test_subset, train, retriever, max_rounds_list=[0, 1, 2, 3]):
    """Test different max refinement rounds."""
    results = {}

    for T in max_rounds_list:
        print(f"\n{'='*60}")
        print(f"Ablation: rounds | T={T} | Tasks: {len(test_subset)}")
        print(f"{'='*60}")

        task_results = []
        passed_count = 0

        for i, task in enumerate(test_subset):
            tid = task["task_id"]
            print(f"  [{i+1}/{len(test_subset)}] {tid}...", end=" ", flush=True)

            try:
                # Retrieve
                query_text = f"{task['instruct_prompt']}\n{task['code_prompt']}\n{task['libs']}"
                retrieved = retriever.retrieve(query_text, train, k=3)

                # Initial synthesis
                skill = synthesize_skill(client, task, retrieved)
                code = solve_task(client, task, skill=skill)
                passed, trace = validate_code(code, task)

                rounds_used = 0
                if not passed and T > 0:
                    for t in range(T):
                        skill = synthesize_skill(client, task, retrieved, prev_skill=skill, failure_trace=trace)
                        code = solve_task(client, task, skill=skill)
                        passed, trace = validate_code(code, task)
                        rounds_used = t + 1
                        if passed:
                            break

                if passed:
                    passed_count += 1
                    print(f"✓ (round {rounds_used})")
                else:
                    print("✗")

                task_results.append({"task_id": tid, "passed": passed, "rounds": rounds_used})
            except Exception as e:
                print(f"ERROR: {e}")
                task_results.append({"task_id": tid, "passed": False, "error": str(e)})

            time.sleep(0.5)

        pass_rate = passed_count / len(test_subset)
        print(f"\n  T={T}: Pass@1 = {pass_rate:.3f} ({passed_count}/{len(test_subset)})")

        results[f"T={T}"] = {
            "max_rounds": T,
            "total": len(test_subset),
            "passed": passed_count,
            "pass_rate": pass_rate,
            "results": task_results
        }

    return results


# --- Ablation: Trace Content ---
def run_trace_ablation(client, test_subset, train, retriever, trace_modes=["full", "error_only", "generic"]):
    """Test different failure trace content."""
    results = {}

    for mode in trace_modes:
        print(f"\n{'='*60}")
        print(f"Ablation: trace | mode={mode} | Tasks: {len(test_subset)}")
        print(f"{'='*60}")

        task_results = []
        passed_count = 0
        recovered = 0

        for i, task in enumerate(test_subset):
            tid = task["task_id"]
            print(f"  [{i+1}/{len(test_subset)}] {tid}...", end=" ", flush=True)

            try:
                query_text = f"{task['instruct_prompt']}\n{task['code_prompt']}\n{task['libs']}"
                retrieved = retriever.retrieve(query_text, train, k=3)

                skill = synthesize_skill(client, task, retrieved)
                code = solve_task(client, task, skill=skill)
                passed, trace = validate_code(code, task)

                if not passed:
                    # Refine with specified trace mode
                    skill = synthesize_skill(client, task, retrieved, prev_skill=skill,
                                           failure_trace=trace, trace_mode=mode)
                    code = solve_task(client, task, skill=skill)
                    passed, trace = validate_code(code, task)
                    if passed:
                        recovered += 1

                if passed:
                    passed_count += 1
                    print("✓")
                else:
                    print("✗")

                task_results.append({"task_id": tid, "passed": passed})
            except Exception as e:
                print(f"ERROR: {e}")
                task_results.append({"task_id": tid, "passed": False, "error": str(e)})

            time.sleep(0.5)

        pass_rate = passed_count / len(test_subset)
        print(f"\n  {mode}: Pass@1 = {pass_rate:.3f} ({passed_count}/{len(test_subset)}), recovered={recovered}")

        results[mode] = {
            "trace_mode": mode,
            "total": len(test_subset),
            "passed": passed_count,
            "pass_rate": pass_rate,
            "recovered": recovered,
            "results": task_results
        }

    return results


# --- Ablation: Retrieval Size ---
def run_retrieval_ablation(client, test_subset, train, retriever, k_values=[1, 3, 5]):
    """Test different retrieval sizes with ISR."""
    results = {}

    for k in k_values:
        print(f"\n{'='*60}")
        print(f"Ablation: retrieval | k={k} | Tasks: {len(test_subset)}")
        print(f"{'='*60}")

        task_results = []
        passed_count = 0

        for i, task in enumerate(test_subset):
            tid = task["task_id"]
            print(f"  [{i+1}/{len(test_subset)}] {tid}...", end=" ", flush=True)

            try:
                query_text = f"{task['instruct_prompt']}\n{task['code_prompt']}\n{task['libs']}"
                retrieved = retriever.retrieve(query_text, train, k=k)

                skill = synthesize_skill(client, task, retrieved)
                code = solve_task(client, task, skill=skill)
                passed, trace = validate_code(code, task)

                if not passed:
                    skill = synthesize_skill(client, task, retrieved, prev_skill=skill, failure_trace=trace)
                    code = solve_task(client, task, skill=skill)
                    passed, trace = validate_code(code, task)

                if passed:
                    passed_count += 1
                    print("✓")
                else:
                    print("✗")

                task_results.append({"task_id": tid, "passed": passed})
            except Exception as e:
                print(f"ERROR: {e}")
                task_results.append({"task_id": tid, "passed": False, "error": str(e)})

            time.sleep(0.5)

        pass_rate = passed_count / len(test_subset)
        print(f"\n  k={k}: Pass@1 = {pass_rate:.3f} ({passed_count}/{len(test_subset)})")

        results[f"k={k}"] = {
            "k": k,
            "total": len(test_subset),
            "passed": passed_count,
            "pass_rate": pass_rate,
            "results": task_results
        }

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", choices=["rounds", "trace", "retrieval"], required=True)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()

    if not API_KEY:
        print("ERROR: LLM_API_KEY not set")
        sys.exit(1)

    client = get_client()

    print("Loading BigCodeBench...")
    all_tasks = load_bigcodebench()
    print(f"Total tasks: {len(all_tasks)}")

    train, test = split_data(all_tasks, seed=123)
    print(f"Train: {len(train)}, Test: {len(test)}")

    print("Building TF-IDF retrieval index...")
    pool_texts = [f"{t['instruct_prompt']}\n{t['code_prompt']}\n{t['libs']}" for t in train]
    retriever = TFIDFRetriever(pool_texts)
    print("Done.")

    test_subset = test[args.offset:args.offset + args.limit]

    if args.ablation == "rounds":
        results = run_rounds_ablation(client, test_subset, train, retriever)
    elif args.ablation == "trace":
        results = run_trace_ablation(client, test_subset, train, retriever)
    elif args.ablation == "retrieval":
        results = run_retrieval_ablation(client, test_subset, train, retriever)

    # Save
    out_file = RESULTS_DIR / f"ablation_{args.ablation}_n{len(test_subset)}.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
