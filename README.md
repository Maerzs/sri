# Iterative Skill Refinement (ISR)

Code for **Failure-Guided Task-Local Skill Refinement for LLM Code Agents**. ISR uses a failed execution trace to revise a task-local textual skill, then asks the same solver to generate a fresh program from the original task context. The failed program is not passed to the skill refiner or the second solver.

This repository contains source code only. The manuscript, benchmark data, API credentials, caches, generated programs, and experiment outputs are not included.

## Repository layout

| Path | Purpose |
| --- | --- |
| `code/run_experiment.py` | BigCodeBench data loading, retrieval, synthesis, solving, and validation |
| `code/run_heldout_eval.py` | Deterministic method-level feedback/held-out evaluation |
| `code/run_experiment_concurrent.py` | Full-split zero-shot, one-shot, ISR, and code-repair runs |
| `code/run_simple_retry_concurrent.py` | Same-skill solver retry control |
| `code/run_trace_ablation_concurrent.py`, `code/run_trace_ablation_v2.py` | Trace-content controls |
| `code/run_rounds_ablation_concurrent.py`, `code/run_ablation.py` | Refinement-round and retrieval ablations |
| `code/run_evalplus_heldout.py`, `code/rescore_evalplus_official.py` | HumanEval+/MBPP+ boundary evaluation |
| `code/run_cross_task_reuse.py` | Cross-task skill reuse check |
| `code/compute_*.py`, `code/summarize_*.py`, `code/aggregate_multirun.py` | Audits and result summaries |

## Setup

Use Python 3.10 or newer. Install the experiment dependencies in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Set `LLM_API_KEY`, `LLM_BASE_URL`, and `LLM_MODEL` in `.env` for an OpenAI-compatible chat endpoint. `SOLVER_MODEL` and `SYNTH_MODEL` can override the shared model. The paper's main experiments used a DeepSeek model through an OpenAI-compatible endpoint; model availability and behavior depend on your provider. TF-IDF retrieval is the default. For dense retrieval, set `RETRIEVER=embedding` plus `EMB_API_KEY`, `EMB_BASE_URL`, and `EMB_MODEL`.

The scripts download BigCodeBench `bigcode/bigcodebench` split `v0.1.2` via Hugging Face `datasets`. They shuffle its 1,140 tasks with seed 123 into 798 retrieval-pool tasks and 342 evaluation tasks. The method-level audit uses a separate seed-123 split of each task's test methods. No dataset copy is committed here.

## Run

Start with a small, paid API run:

```bash
python code/run_heldout_eval.py --mode oneshot --limit 2 --save-artifacts
python code/run_heldout_eval.py --mode isr --limit 2 --save-artifacts
```

For the 342-task held-out comparison, run each method with the same model and configuration:

```bash
python code/run_heldout_eval.py --mode oneshot --limit 342 --split-seed 123 --save-artifacts
python code/run_heldout_eval.py --mode simple_retry --limit 342 --split-seed 123 --save-artifacts
python code/run_heldout_eval.py --mode code_refine --limit 342 --split-seed 123 --save-artifacts
python code/run_heldout_eval.py --mode isr --limit 342 --split-seed 123 --save-artifacts
```

For full-split experiments, use `python code/run_experiment_concurrent.py --mode isr --concurrency 4`; the supported modes are `zeroshot`, `oneshot`, `isr`, and `code_refine`. Outputs go to the ignored `code/results/` directory unless `--out` is specified. The code executes generated programs and benchmark tests locally, so run it in a disposable, isolated environment with the benchmark's required Python packages installed. API calls and dataset downloads are needed for actual runs.

The default retrieval setting includes up to 400 characters of each retrieved training canonical solution. Set `CANONICAL_PREFIX_CHARS=0` (or `--canonical-prefix-chars 0`) for the no-prefix control. Keep this setting, provider, model, seed, and test split identical across compared methods.

## Scope

The repository provides the experiment implementation and analysis scripts, not the original run artifacts. Reported paper percentages cannot be recalculated from this checkout alone. Full-split feedback evaluation exposes all validator methods during refinement; the held-out audit separates feedback and scoring methods within each task, but is not an official hidden test set. EvalPlus and cross-task reuse are separate boundary checks.
