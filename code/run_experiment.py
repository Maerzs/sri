"""
Run ISR experiments on BigCodeBench (full version with tests).

Usage:
    python run_experiment.py --mode [zeroshot|oneshot|isr|all] --limit 50
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
from dataclasses import dataclass, field

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

# --- Config ---
BASE_URL = os.environ.get("LLM_BASE_URL", "https://llm.invalley.co/v1")
API_KEY = os.environ.get("LLM_API_KEY", os.environ.get("API_KEY", ""))
# Model can be overridden via env vars to swap models without editing code.
# LLM_MODEL sets both at once; SOLVER_MODEL / SYNTH_MODEL override individually.
_DEFAULT_MODEL = os.environ.get("LLM_MODEL", "deepseek-v4-pro")
SOLVER_MODEL = os.environ.get("SOLVER_MODEL", _DEFAULT_MODEL)
SYNTH_MODEL = os.environ.get("SYNTH_MODEL", _DEFAULT_MODEL)
TOP_K = 3
TEMPERATURE = 0.0
DATA_SEED = 123
DEFAULT_CANONICAL_PREFIX_CHARS = 400

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def parse_canonical_prefix_chars(value=None):
    """Resolve canonical solution prefix length from CLI/env/default."""
    raw = os.environ.get("CANONICAL_PREFIX_CHARS", DEFAULT_CANONICAL_PREFIX_CHARS)
    if value is not None:
        raw = value
    try:
        chars = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"canonical_prefix_chars must be an integer, got {raw!r}") from exc
    if chars < 0:
        raise ValueError("canonical_prefix_chars must be >= 0")
    return chars


def get_retriever_label():
    return os.environ.get("RETRIEVER", "tfidf").lower()


def build_metadata(mode, canonical_prefix_chars, repeat_id=None, split_seed=None):
    model_label = os.environ.get("MODEL_LABEL") or (
        SOLVER_MODEL if SOLVER_MODEL == SYNTH_MODEL else f"solver={SOLVER_MODEL};synth={SYNTH_MODEL}"
    )
    metadata = {
        "model_label": model_label,
        "solver_model": SOLVER_MODEL,
        "synth_model": SYNTH_MODEL,
        "retriever": get_retriever_label(),
        "canonical_prefix_chars": canonical_prefix_chars,
        "data_seed": DATA_SEED,
        "split_seed": split_seed,
        "repeat_id": repeat_id,
        "temperature": TEMPERATURE,
        "mode": mode,
    }
    return metadata


def get_client():
    timeout = float(os.environ.get("LLM_REQUEST_TIMEOUT", "180"))
    return OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=timeout)


# --- Data Loading ---
def load_bigcodebench():
    """Load full BigCodeBench with tests and canonical solutions."""
    from datasets import load_dataset
    ds = load_dataset("bigcode/bigcodebench", split="v0.1.2")
    return list(ds)


def split_data(tasks, seed=123):
    """Split 1140 tasks into 798 train / 342 test (matching SkillTTA)."""
    rng = random.Random(seed)
    indices = list(range(len(tasks)))
    rng.shuffle(indices)
    train = [tasks[i] for i in indices[:798]]
    test = [tasks[i] for i in indices[798:]]
    return train, test


# --- Retrieval (TF-IDF based, no external embedding API needed) ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine


class TFIDFRetriever:
    """Simple TF-IDF retriever. Fast, free, no API calls."""

    def __init__(self, corpus_texts):
        self.vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
        self.matrix = self.vectorizer.fit_transform(corpus_texts)

    def retrieve(self, query, pool, k=TOP_K):
        query_vec = self.vectorizer.transform([query])
        sims = sklearn_cosine(query_vec, self.matrix).flatten()
        top_idx = np.argsort(sims)[-k:][::-1]
        return [pool[i] for i in top_idx]


# --- Dense embedding retriever (Qwen3-Embedding-8B via aiping.cn) ---
import hashlib
import threading

EMB_BASE_URL = os.environ.get("EMB_BASE_URL", "https://www.aiping.cn/api/v1")
EMB_API_KEY = os.environ.get("EMB_API_KEY", "")
EMB_MODEL = os.environ.get("EMB_MODEL", "Qwen3-Embedding-8B")
_EMB_CACHE_DIR = Path(__file__).parent / "emb_cache"
_EMB_CACHE_DIR.mkdir(exist_ok=True)


def _emb_client():
    return OpenAI(base_url=EMB_BASE_URL, api_key=EMB_API_KEY)


def _embed_batch(client, texts, max_retries=4):
    """Embed a list of texts with retry. Returns np.ndarray [n, d]."""
    for attempt in range(max_retries):
        try:
            resp = client.embeddings.create(model=EMB_MODEL, input=texts)
            return np.array([d.embedding for d in resp.data], dtype=np.float32)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 * (attempt + 1))


class EmbeddingRetriever:
    """Dense retriever using Qwen3-Embedding-8B. Pool embeddings are cached on disk
    keyed by a hash of the corpus so we only pay the encode cost once."""

    def __init__(self, corpus_texts, batch_size=16):
        self.client = _emb_client()
        self._lock = threading.Lock()
        key = hashlib.sha1(("\u241f".join(corpus_texts) + "|" + EMB_MODEL).encode("utf-8")).hexdigest()[:16]
        cache_file = _EMB_CACHE_DIR / f"pool_{EMB_MODEL.replace('/','_')}_{len(corpus_texts)}_{key}.npy"
        if cache_file.exists():
            self.matrix = np.load(cache_file)
            print(f"  [emb] loaded cached pool embeddings {self.matrix.shape} from {cache_file.name}", flush=True)
        else:
            print(f"  [emb] encoding {len(corpus_texts)} pool texts with {EMB_MODEL} (batch={batch_size})...", flush=True)
            embs = []
            for i in range(0, len(corpus_texts), batch_size):
                chunk = [t[:4000] for t in corpus_texts[i:i + batch_size]]
                embs.append(_embed_batch(self.client, chunk))
                print(f"  [emb] {min(i + batch_size, len(corpus_texts))}/{len(corpus_texts)}", flush=True)
            self.matrix = np.vstack(embs)
            np.save(cache_file, self.matrix)
            print(f"  [emb] cached pool embeddings {self.matrix.shape} -> {cache_file.name}", flush=True)
        # pre-normalize for cosine via dot product
        norms = np.linalg.norm(self.matrix, axis=1, keepdims=True)
        self._norm_matrix = self.matrix / np.clip(norms, 1e-8, None)

    def retrieve(self, query, pool, k=TOP_K):
        with self._lock:
            q = _embed_batch(self.client, [query[:4000]])[0]
        qn = q / max(np.linalg.norm(q), 1e-8)
        sims = self._norm_matrix @ qn
        top_idx = np.argsort(sims)[-k:][::-1]
        return [pool[i] for i in top_idx]


def build_retriever(corpus_texts):
    """Select retriever via RETRIEVER env var: 'tfidf' (default) or 'embedding'."""
    kind = os.environ.get("RETRIEVER", "tfidf").lower()
    if kind in ("embedding", "emb", "dense", "qwen3"):
        return EmbeddingRetriever(corpus_texts)
    return TFIDFRetriever(corpus_texts)


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

REFINE_TEMPLATE = """The previous skill led to a failed execution. Refine it.

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


def synthesize_skill(client, task, retrieved, prev_skill=None, failure_trace=None, canonical_prefix_chars=None):
    """Synthesize or refine a skill."""
    canonical_prefix_chars = parse_canonical_prefix_chars(canonical_prefix_chars)
    if prev_skill and failure_trace:
        user_msg = REFINE_TEMPLATE.format(
            instruction=task["instruct_prompt"][:1000],
            code_prompt=task["code_prompt"],
            libs=task["libs"],
            prev_skill=prev_skill,
            failure_trace=failure_trace[:1500]
        )
    else:
        evidence_parts = []
        for i, r in enumerate(retrieved):
            part = (
                f"### Example {i+1} (similar task)\n"
                f"Instruction: {r['instruct_prompt'][:200]}\n"
                f"Libraries: {r['libs']}\n"
            )
            if canonical_prefix_chars > 0:
                part += (
                    "Solution approach:\n"
                    f"```python\n{r['canonical_solution'][:canonical_prefix_chars]}\n```\n"
                )
            evidence_parts.append(part)
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
    """Generate code for a task."""
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

    # Strip thinking tags (DeepSeek reasoning models output <think>...</think>)
    if "</think>" in code:
        code = code.split("</think>", 1)[-1].strip()

    # Extract code from markdown fences (take the FIRST python code block)
    if "```python" in code:
        parts = code.split("```python")
        # Take the first code block after ```python
        code_block = parts[1] if len(parts) > 1 else parts[0]
        if "```" in code_block:
            code_block = code_block.split("```")[0]
        code = code_block.strip()
    elif "```" in code:
        parts = code.split("```")
        if len(parts) >= 2:
            code = parts[1].strip()

    return code


# --- Code-level refinement (Self-Refine / Reflexion-style baseline) ---
SOLVER_REFINE_TEMPLATE = """Your previous code failed its tests. Fix the code directly.

## Instruction
{instruction}

## Code to complete
```python
{code_prompt}
```

## Guidance (SKILL.md)
{skill}

## Your previous (failed) code
```python
{prev_code}
```

## Failure trace
{failure_trace}

## Rules
- Diagnose the failure from the trace, then output a CORRECTED complete Python program.
- Output ONLY valid Python code (imports + function definition); no tests, no markdown, no commentary."""


def refine_code(client, task, skill, prev_code, failure_trace):
    """Code-level refinement: feed the trace back to the SOLVER to rewrite the
    code, keeping the skill fixed. This is the Self-Refine/Reflexion-style
    counterpart to ISR, which instead refines the skill."""
    user_msg = SOLVER_REFINE_TEMPLATE.format(
        instruction=task["instruct_prompt"][:1500],
        code_prompt=task["code_prompt"],
        skill=(skill or "(none)")[:1500],
        prev_code=prev_code[:2000],
        failure_trace=failure_trace[:1500],
    )
    resp = client.chat.completions.create(
        model=SOLVER_MODEL,
        messages=[
            {"role": "system", "content": SOLVER_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        temperature=TEMPERATURE,
        max_tokens=3000,
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
    """Run generated code against task's test suite."""
    test_code = task["test"]
    entry_point = task["entry_point"]

    # Build full test script
    # BigCodeBench uses unittest.TestCase, run with unittest
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
        return False, "TIMEOUT: Execution exceeded 30 seconds"
    except Exception as e:
        return False, f"EXECUTION ERROR: {str(e)}"
    finally:
        os.unlink(tmp_path)


# --- Main Pipeline ---
def run_single_task(client, task, pool, retriever, mode="isr", max_rounds=1, canonical_prefix_chars=None):
    """Run a single task through the pipeline."""
    canonical_prefix_chars = parse_canonical_prefix_chars(canonical_prefix_chars)

    if mode == "zeroshot":
        code = solve_task(client, task, skill=None)
        passed, trace = validate_code(code, task)
        return {"passed": passed, "rounds": 0, "failure_trace": trace}

    # Retrieve
    query_text = f"{task['instruct_prompt']}\n{task['code_prompt']}\n{task['libs']}"
    retrieved = retriever.retrieve(query_text, pool, k=TOP_K)

    # Initial synthesis
    skill = synthesize_skill(client, task, retrieved, canonical_prefix_chars=canonical_prefix_chars)

    # Solve
    code = solve_task(client, task, skill=skill)
    passed, trace = validate_code(code, task)

    if passed or mode == "oneshot":
        return {"passed": passed, "rounds": 0, "failure_trace": trace, "skill": skill}

    if mode == "code_refine":
        # Code-level refinement (Self-Refine/Reflexion-style): keep the skill
        # fixed and feed the trace back to the SOLVER to rewrite the code.
        for t in range(max_rounds):
            code = refine_code(client, task, skill, code, trace)
            passed, trace = validate_code(code, task)
            if passed:
                return {"passed": True, "rounds": t + 1, "failure_trace": "", "skill": skill}
        return {"passed": False, "rounds": max_rounds, "failure_trace": trace, "skill": skill}

    # ISR: refine the SKILL on failure
    for t in range(max_rounds):
        skill = synthesize_skill(
            client, task, retrieved, prev_skill=skill, failure_trace=trace,
            canonical_prefix_chars=canonical_prefix_chars,
        )
        code = solve_task(client, task, skill=skill)
        passed, trace = validate_code(code, task)
        if passed:
            return {"passed": True, "rounds": t + 1, "failure_trace": "", "skill": skill}

    return {"passed": False, "rounds": max_rounds, "failure_trace": trace, "skill": skill}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["zeroshot", "oneshot", "isr", "code_refine", "all"], default="all")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--max-rounds", type=int, default=1)
    parser.add_argument("--offset", type=int, default=0, help="Skip first N test tasks")
    parser.add_argument("--canonical-prefix-chars", type=int, default=None,
                        help="Chars of retrieved canonical_solution to include; "
                             "env CANONICAL_PREFIX_CHARS or default 400. Use 0 to omit code blocks.")
    args = parser.parse_args()
    canonical_prefix_chars = parse_canonical_prefix_chars(args.canonical_prefix_chars)

    if not API_KEY:
        print("ERROR: LLM_API_KEY not set")
        sys.exit(1)

    client = get_client()

    # Load data
    print("Loading BigCodeBench...")
    all_tasks = load_bigcodebench()
    print(f"Total tasks: {len(all_tasks)}")

    train, test = split_data(all_tasks, seed=DATA_SEED)
    print(f"Train: {len(train)}, Test: {len(test)}")

    # Build retrieval index (TF-IDF, instant)
    print("Building TF-IDF retrieval index...")
    pool_texts = [f"{t['instruct_prompt']}\n{t['code_prompt']}\n{t['libs']}" for t in train]
    retriever = TFIDFRetriever(pool_texts)
    print("Done.")

    # Run experiments
    modes = [args.mode] if args.mode != "all" else ["zeroshot", "oneshot", "isr"]
    test_subset = test[args.offset:args.offset + args.limit]

    for mode in modes:
        print(f"\n{'='*60}")
        print(f"Running: {mode} | Tasks: {len(test_subset)} | Max rounds: {args.max_rounds}")
        print(f"{'='*60}")

        results = []
        passed_count = 0
        recovered_count = 0  # tasks that failed oneshot but passed after refinement

        for i, task in enumerate(test_subset):
            tid = task["task_id"]
            print(f"  [{i+1}/{len(test_subset)}] {tid}...", end=" ", flush=True)

            try:
                r = run_single_task(client, task, train, retriever,
                                   mode=mode, max_rounds=args.max_rounds,
                                   canonical_prefix_chars=canonical_prefix_chars)
                if r["passed"]:
                    passed_count += 1
                    if r.get("rounds", 0) > 0:
                        recovered_count += 1
                    print(f"✓ (round {r['rounds']})")
                else:
                    print("✗")

                results.append({
                    "task_id": tid,
                    "passed": r["passed"],
                    "rounds": r.get("rounds", 0),
                })
            except Exception as e:
                print(f"ERROR: {e}")
                results.append({"task_id": tid, "passed": False, "error": str(e)})

            time.sleep(0.5)

        # Summary
        total = len(test_subset)
        pass_rate = passed_count / total if total > 0 else 0

        print(f"\n{'─'*40}")
        print(f"Mode: {mode}")
        print(f"Pass@1: {pass_rate:.3f} ({passed_count}/{total})")
        if mode == "isr":
            print(f"Recovered by refinement: {recovered_count}/{total - (passed_count - recovered_count)} failed tasks")
        print(f"{'─'*40}")

        # Save
        out_file = RESULTS_DIR / f"{mode}_n{total}.json"
        with open(out_file, 'w') as f:
            json.dump({
                "metadata": build_metadata(mode, canonical_prefix_chars),
                "mode": mode,
                "total": total,
                "passed": passed_count,
                "pass_rate": pass_rate,
                "recovered": recovered_count,
                "max_rounds": args.max_rounds,
                "results": results
            }, f, indent=2)
        print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
