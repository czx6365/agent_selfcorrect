"""Run direct and chain-of-thought GSM8K baselines on the fixed evaluation set."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agents.base import BaselineAgent
from agents.self_refine import SelfRefineAgent
from eval.llm_client import (
    AnthropicCompatibleClient,
    LLMConfigurationError,
    LocalLlamaServerClient,
    OpenAICompatibleClient,
)
from eval.metrics import accuracy, exact_match, extract_gsm8k_answer


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "dataset.jsonl"
DEFAULT_RESULTS_DIR = ROOT / "results"
DEFAULT_FAILURE_REVIEW = ROOT.parent / "failure_review.md"
RECORDS_FILENAME = "baseline_records.jsonl"
SUMMARY_FILENAME = "baseline_summary.json"


def load_dataset(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_existing_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_comparison(records: list[dict[str, Any]]) -> dict[str, int] | None:
    by_method: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        by_method.setdefault(record["method"], {})[record["id"]] = record
    direct = by_method.get("baseline_direct")
    cot = by_method.get("baseline_cot")
    if not direct or not cot or direct.keys() != cot.keys():
        return None

    comparison = {"kept_correct": 0, "fixed_by_cot": 0, "regressed_by_cot": 0, "wrong_both": 0}
    for question_id, direct_record in direct.items():
        cot_record = cot[question_id]
        key = {
            (True, True): "kept_correct",
            (False, True): "fixed_by_cot",
            (True, False): "regressed_by_cot",
            (False, False): "wrong_both",
        }[(bool(direct_record["correct"]), bool(cot_record["correct"]))]
        comparison[key] += 1
    return comparison


def compare_methods(records: list[dict[str, Any]], before: str, after: str) -> dict[str, int] | None:
    by_method = {method: {item["id"]: item for item in records if item["method"] == method} for method in (before, after)}
    if not by_method[before] or by_method[before].keys() != by_method[after].keys():
        return None
    result = {"kept_correct": 0, "fixed": 0, "regressed": 0, "wrong_both": 0}
    for question_id, before_record in by_method[before].items():
        after_record = by_method[after][question_id]
        key = {
            (True, True): "kept_correct", (False, True): "fixed",
            (True, False): "regressed", (False, False): "wrong_both",
        }[(bool(before_record["correct"]), bool(after_record["correct"]))]
        result[key] += 1
    return result


def self_refine_report(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    cot_to_refine = compare_methods(records, "baseline_cot", "self_refine")
    direct_to_cot = compare_methods(records, "baseline_direct", "baseline_cot")
    if cot_to_refine is None or direct_to_cot is None:
        return None
    direct = {item["id"]: item for item in records if item["method"] == "baseline_direct"}
    cot = {item["id"]: item for item in records if item["method"] == "baseline_cot"}
    refine = {item["id"]: item for item in records if item["method"] == "self_refine"}
    cot_fixes = [question_id for question_id in direct if not direct[question_id]["correct"] and cot[question_id]["correct"]]
    return {
        "cot_to_self_refine": cot_to_refine,
        "on_cot_fixes": {
            "total": len(cot_fixes),
            "kept_correct": sum(bool(refine[item_id]["correct"]) for item_id in cot_fixes),
            "regressed": sum(not refine[item_id]["correct"] for item_id in cot_fixes),
        },
    }


def write_failure_review(path: Path, records: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    failures = [record for record in records if not record["correct"]]
    lines = [
        "# Baseline Failure Review",
        "",
        f"- Method: `{summary['method']}`",
        f"- Evaluated: {summary['total']} GSM8K test examples (fixed seed: {summary['dataset_seed']})",
        f"- Accuracy: {summary['accuracy']:.1%} ({summary['correct']}/{summary['total']})",
        f"- Failure cases below: {min(len(failures), 5)} of {len(failures)}",
        "",
        "## Interpretation boundary",
        "",
        "A one-pass baseline only establishes that the final answer is wrong. It cannot prove whether the model lacked the reasoning capability or had a recoverable mistake that it failed to check. That distinction requires a later independent critique or tool-feedback trace; do not infer it from the gold answer during generation.",
        "",
        "## Sample Failures",
        "",
    ]
    if not failures:
        lines.append("No failures were recorded.")
    for record in failures[:5]:
        lines.extend(
            [
                f"### {record['id']}",
                "",
                f"Question: {record['question']}",
                "",
                f"- Expected: `{record['expected_answer']}`",
                f"- Extracted prediction: `{record['prediction'] or 'unparsed'}`",
                f"- Initial classification: `needs critique/verification`; baseline evidence alone cannot separate capability failure from missed checking.",
                "",
                "Model response:",
                "",
                "```text",
                record["raw_response"].strip(),
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    examples = load_dataset(args.dataset)
    if args.limit:
        examples = examples[: args.limit]
    if not examples:
        raise ValueError("No evaluation examples found.")

    if args.provider == "anthropic":
        client = AnthropicCompatibleClient(
            model=args.model,
            base_url=args.base_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            thinking=args.thinking,
        )
    elif args.provider == "local":
        client = LocalLlamaServerClient(
            model=args.model,
            base_url=args.base_url,
            temperature=args.temperature,
            seed=args.seed,
            max_tokens=args.max_tokens,
        )
    else:
        client = OpenAICompatibleClient(
            model=args.model,
            base_url=args.base_url,
            temperature=args.temperature,
            seed=args.seed,
            max_tokens=args.max_tokens,
        )
    method = "self_refine" if args.method == "self_refine" else f"baseline_{args.mode}"
    agent = SelfRefineAgent(client, max_rounds=args.rounds) if args.method == "self_refine" else BaselineAgent(client, mode=args.mode)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.results_dir / RECORDS_FILENAME
    summary_path = args.results_dir / SUMMARY_FILENAME
    if args.reset_results and summary_path.exists():
        summary_path.unlink()
    all_records = [] if args.reset_results else load_existing_records(records_path)
    if all_records and summary_path.exists():
        existing_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        existing_methods = existing_summary.get("methods", {}).values()
        requested_config = {
            "provider": args.provider,
            "llm_model": client.model,
            "temperature": args.temperature,
            "max_tokens": getattr(client, "max_tokens", None),
            "thinking": getattr(client, "thinking", None),
        }
        if any(
            any(existing.get(key) != value for key, value in requested_config.items())
            for existing in existing_methods
        ):
            raise ValueError(
                "Existing results use a different model configuration. "
                "Pass --reset-results to start a new comparable experiment."
            )
    other_records = [record for record in all_records if record.get("method") != method]
    records = [record for record in all_records if record.get("method") == method] if args.resume else []
    completed_ids = {record["id"] for record in records}
    if completed_ids:
        print(f"Resuming with {len(completed_ids)} completed examples.", flush=True)
    pending_examples = [example for example in examples if example["id"] not in completed_ids]
    if args.max_new is not None:
        pending_examples = pending_examples[: args.max_new]

    def evaluate_example(example: dict[str, Any]) -> dict[str, Any]:
        raw_response, trace = agent.solve(example["question"], question_id=example["id"])
        prediction = extract_gsm8k_answer(raw_response)
        correct = exact_match(prediction, example["answer"])
        trace.final_answer = prediction or ""
        trace.final_correct = correct
        return {
            "method": method,
            "id": example["id"],
            "question": example["question"],
            "expected_answer": example["answer"],
            "prediction": prediction,
            "correct": correct,
            "raw_response": raw_response,
            "trace": {"method": trace.method, "steps": trace.steps},
        }

    with records_path.open("w", encoding="utf-8") as records_handle:
        for record in other_records + records:
            records_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(evaluate_example, example) for example in pending_examples]
            for completed, future in enumerate(as_completed(futures), start=1):
                record = future.result()
                records.append(record)
                records_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                records_handle.flush()
                print(
                    f"[{len(completed_ids) + completed}/{len(examples)}] {record['id']}: "
                    f"{'correct' if record['correct'] else 'wrong'}",
                    flush=True,
                )

    summary: dict[str, Any] = {
        **accuracy(records),
        "method": method,
        "dataset": str(args.dataset),
        "dataset_seed": examples[0]["sampling_seed"],
        "llm_model": client.model,
        "provider": args.provider,
        "temperature": args.temperature,
        "llm_seed": args.seed,
        "max_tokens": getattr(client, "max_tokens", None),
        "thinking": getattr(client, "thinking", None),
        "completed_at": datetime.now(UTC).isoformat(),
        "complete": len(records) == len(examples),
        "target_total": len(examples),
    }
    combined_summary: dict[str, Any] = {}
    if summary_path.exists() and not args.reset_results:
        combined_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    methods = combined_summary.get("methods", {})
    methods[method] = summary
    combined_summary = {
        "dataset": str(args.dataset),
        "dataset_seed": examples[0]["sampling_seed"],
        "methods": methods,
        "comparison": build_comparison(other_records + records),
        "self_refine": self_refine_report(other_records + records),
        "updated_at": datetime.now(UTC).isoformat(),
    }
    summary_path.write_text(json.dumps(combined_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if summary["complete"]:
        write_failure_review(args.failure_review, records, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--method", choices=("baseline", "self_refine"), default="baseline")
    parser.add_argument("--provider", choices=("openai", "anthropic", "local"), default="local")
    parser.add_argument("--mode", choices=("direct", "cot"), default="cot")
    parser.add_argument("--rounds", type=int, default=1, help="Self-Refine critique/revision rounds.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new", type=int, default=None, help="At most this many new examples per invocation.")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent model requests; use conservatively.")
    parser.add_argument("--model", default=None, help="Overrides the provider's configured model.")
    parser.add_argument("--base-url", default=None, help="Overrides the provider's configured base URL.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--thinking", choices=("enabled", "disabled"), default="disabled")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--reset-results",
        action="store_true",
        help="Discard existing records before starting a different model configuration.",
    )
    parser.add_argument("--failure-review", type=Path, default=DEFAULT_FAILURE_REVIEW)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted run from its existing JSONL records.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        summary = run(args)
    except LLMConfigurationError as error:
        raise SystemExit(f"Evaluation was not started: {error}") from error
    state = "complete" if summary["complete"] else f"partial, target {summary['target_total']}"
    print(f"Accuracy ({summary['method']}, {state}): {summary['accuracy']:.1%} ({summary['correct']}/{summary['total']})")


if __name__ == "__main__":
    main()
