# SelfCorrect Agent

**When does LLM self-correction actually help — and when does it make a correct answer worse?**

SelfCorrect Agent is an experimental research project that evaluates several forms of iterative LLM correction on **GSM8K** and **HumanEval**. The project compares unguided self-refinement with correction driven by external evidence such as calculators, unit tests, and previously verified examples.

The main finding is simple:

> **More revision is not automatically better. Self-correction becomes more reliable when the feedback signal is concrete, externally checkable, and strong enough to justify changing the original answer.**

## Research Question

Many agent workflows assume that asking an LLM to critique and rewrite its own output will improve accuracy. This project tests that assumption under controlled settings.

It asks:

1. Does unguided self-refinement improve reasoning accuracy?
2. Can external tools prevent harmful revisions?
3. Do reflection memories help across tasks?
4. Does increasing the number of correction rounds provide monotonic gains?
5. Can verified execution feedback act as a reliable candidate-selection signal?

## Experimental Setup

- **Model:** local Qwen3-8B Q4
- **Temperature:** `0`
- **Seed:** `42`
- **Math benchmark:** first 100 GSM8K examples
- **Code benchmark:** first 100 HumanEval examples
- **Math metric:** exact-match numerical accuracy
- **Code metric:** official `check(candidate)` unit-test pass rate

The evaluation pipeline records per-example outputs, correction traces, aggregate metrics, pairwise method comparisons, and failure cases.

## Methods

| Method | Feedback signal | Role in the study |
| --- | --- | --- |
| Direct | none | non-reasoning baseline |
| CoT | internal reasoning | reasoning baseline |
| Self-Refine original | model self-critique | tests unguided rewrite behavior |
| Self-Refine calculator | self-critique + calculator gate | only revises when arithmetic evidence supports the critique |
| Reflection | verified historical lessons | tests experience reuse |
| CRITIC | calculator | tool-based arithmetic verification |
| Self-Repair | unit-test output | repairs code from executable feedback |
| Correct Example Retrieval | previously verified correct examples | retrieval-based experience reuse without future-answer leakage |

## Results

### GSM8K

| Method | Correct | Accuracy |
| --- | ---: | ---: |
| Direct | 38 / 100 | 38.0% |
| CoT | 94 / 100 | 94.0% |
| Self-Refine, original, 1 round | 90 / 100 | 90.0% |
| Reflection, recent-memory baseline | 94 / 100 | 94.0% |
| CRITIC with calculator feedback | 90 / 100 | 90.0% |
| Reflection + local-hash retrieval | 89 / 100 | 89.0% |
| Reflection + verified-example retrieval | **96 / 100** | **96.0%** |

Key pairwise observations:

- CoT corrected 58 Direct failures but degraded 2 Direct successes.
- Unguided Self-Refine corrected **0** CoT failures and changed **4** previously correct CoT answers into wrong answers.
- Verified-example retrieval improved 2 CoT failures without degrading an already-correct CoT answer in this 100-example run.

### HumanEval

| Method | Passed | Pass rate |
| --- | ---: | ---: |
| Direct | 82 / 100 | 82.0% |
| CoT | 85 / 100 | 85.0% |
| Self-Repair, 1 round | 85 / 100 | 85.0% |
| Self-Repair, 2 rounds | 85 / 100 | 85.0% |
| CoT-Repair, 1 round | 85 / 100 | 85.0% |
| Best-of executed methods | **88 / 100** | **88.0%** |

Key observations:

- Unit-test-guided Self-Repair fixed 3 Direct failures and did not break a previously passing Direct solution in this run.
- A second repair round did not improve over one round.
- Different methods solve partially different subsets of HumanEval, so executable tests can also act as a candidate-selection signal.

## Core Interpretation

The experiments support three practical conclusions.

### 1. Self-critique without evidence is unstable

The model can produce a plausible critique of an answer that was already correct. If the system automatically accepts every critique, additional reasoning can reduce accuracy.

### 2. External feedback is more useful when it is task-aligned

A calculator can verify arithmetic, but it cannot determine whether the model misunderstood the problem statement. Unit tests are stronger for code because they directly check executable behavior.

### 3. Correction should be gated

A robust agent should not treat every self-generated criticism as sufficient evidence for revision. The correction policy should depend on a verifiable signal or a validated memory.

## Leakage Control

The project explicitly avoids giving the current example's reference answer to the solving agent.

- `solve()` receives the problem only.
- Reference answers are used after generation for evaluation.
- Self-Refine prompts do not contain the gold answer.
- CRITIC only verifies calculator expressions proposed by the model.
- Code repair only receives execution / unit-test feedback.
- Correct-example retrieval only uses examples that appeared earlier and were already externally verified.

This distinction is important because otherwise a "self-correction" experiment can accidentally become answer-conditioned revision.

## Research Artifacts

The repository retains the intermediate artifacts needed to inspect the experiments rather than only publishing final percentages.

```text
self-correct-agent/results/
├── baseline_records.jsonl
├── baseline_summary.json
├── code_records.jsonl
├── code_summary.json
├── evaluation_report.md
└── when_correction_helps.svg
```

![When correction helps](self-correct-agent/results/when_correction_helps.svg)

Additional material includes:

- [`self-correct-agent/failure_review.md`](self-correct-agent/failure_review.md) — selected GSM8K failure analysis;
- [`docs/paper_notes.md`](docs/paper_notes.md) — literature notes;
- [`docs/echo_repro_to_selfcorrect.md`](docs/echo_repro_to_selfcorrect.md) — connection between correction research and repository-level bug reproduction.

## Repository Structure

```text
agent_selfcorrect/
├── docs/
│   ├── paper_notes.md
│   ├── research_proposal.md
│   └── echo_repro_to_selfcorrect.md
└── self-correct-agent/
    ├── main.py
    ├── agents/
    │   ├── base.py
    │   ├── self_refine.py
    │   ├── reflection.py
    │   ├── reflection_memory.py
    │   └── critic.py
    ├── tools/
    ├── eval/
    ├── data/
    ├── results/
    ├── logs/
    └── requirements.txt
```

## Quick Start

```bash
cd self-correct-agent
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
```

The default local setup expects an OpenAI-compatible `llama-server` endpoint.

```bash
llama-server -m "$QWEN_GGUF_PATH" \
  --host "$LOCAL_LLM_HOST" \
  --port "$LOCAL_LLM_PORT" \
  --reasoning off
```

### GSM8K examples

```bash
.venv/bin/python main.py eval --provider local --method baseline --mode direct --max-tokens 512 --reset-results
.venv/bin/python main.py eval --provider local --method baseline --mode cot --max-tokens 512
.venv/bin/python main.py eval --provider local --method self_refine --self-refine-mode original --rounds 1 --max-tokens 512
.venv/bin/python main.py eval --provider local --method critic --max-tokens 512
```

### HumanEval examples

```bash
.venv/bin/python eval/run_code_eval.py --provider local --mode direct --limit 100 --max-tokens 768 --workers 1 --reset-results
.venv/bin/python eval/run_code_eval.py --provider local --mode self_repair --limit 100 --max-tokens 768 --repair-rounds 1 --workers 1
```

### Rebuild the report

```bash
.venv/bin/python eval/build_evaluation_report.py
```

## Related Research Directions

The experimental design is informed by work on ReAct, Self-Refine, Reflexion, CRITIC, experience learning, self-consistency, and chain-of-verification. The repository uses these ideas as comparison points rather than claiming to reproduce every paper exactly.

## Current Limitations

- The reported runs use only 100 examples from each benchmark.
- Results come from one local model configuration and should not be generalized to all LLMs.
- Calculator feedback is intentionally narrow and cannot validate semantic reasoning.
- HumanEval execution feedback is stronger than math feedback because code behavior can be directly tested.
- Additional model scales, repeated seeds, and equal-compute baselines would strengthen the conclusions.

## Research Status

This is an independent experimental project for studying **reliable LLM-agent correction and verification**. The emphasis is on controlled comparison, failure analysis, and evidence-gated revision rather than on presenting self-correction as universally beneficial.