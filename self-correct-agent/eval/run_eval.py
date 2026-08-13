"""在固定 GSM8K 数据集上运行 Direct、CoT 和 Self-Refine 评测。"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agents.base import BaselineAgent
from agents.critic import CriticAgent
from agents.self_refine import SelfRefineAgent
from eval.llm_client import (
    AnthropicCompatibleClient,
    LLMConfigurationError,
    LocalLlamaServerClient,
    OpenAICompatibleClient,
)
from eval.metrics import accuracy, exact_match, extract_gsm8k_answer
from agents.reflection import CorrectExample, Reflection, ReflectionAgent

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = PROJECT_ROOT / "data" / "dataset.jsonl"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_FAILURE_REVIEW = PROJECT_ROOT / "failure_review.md"
DEFAULT_REFLECTION_LOG = PROJECT_ROOT / "logs" / "reflection_log.jsonl"
RECORDS_FILENAME = "baseline_records.jsonl"
SUMMARY_FILENAME = "baseline_summary.json"


def load_dataset(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL 数据集。"""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """将记录写入 JSONL 文件。"""
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_existing_records(path: Path) -> list[dict[str, Any]]:
    """读取已有评测记录，用于结果合并和断点续跑。"""
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_comparison(records: list[dict[str, Any]]) -> dict[str, int] | None:
    """逐题比较 Direct 和 CoT 的正确性变化。"""
    by_method: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        by_method.setdefault(record["method"], {})[record["id"]] = record

    direct = by_method.get("baseline_direct")
    cot = by_method.get("baseline_cot")
    # 只有两种方法覆盖完全相同的题目时才能比较。
    if not direct or not cot or direct.keys() != cot.keys():
        return None

    comparison = {
        "kept_correct": 0,
        "fixed_by_cot": 0,
        "regressed_by_cot": 0,
        "wrong_both": 0,
    }
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


def compare_methods(
    records: list[dict[str, Any]], before: str, after: str
) -> dict[str, int] | None:
    """比较任意两种方法：保持正确、修复、退化和均错误。"""
    by_method = {
        method: {
            item["id"]: item
            for item in records
            if item["method"] == method
        }
        for method in (before, after)
    }
    if not by_method[before] or by_method[before].keys() != by_method[after].keys():
        return None

    result = {"kept_correct": 0, "fixed": 0, "regressed": 0, "wrong_both": 0}
    for question_id, before_record in by_method[before].items():
        after_record = by_method[after][question_id]
        key = {
            (True, True): "kept_correct",
            (False, True): "fixed",
            (True, False): "regressed",
            (False, False): "wrong_both",
        }[(bool(before_record["correct"]), bool(after_record["correct"]))]
        result[key] += 1
    return result


def self_refine_report(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """分析 Self-Refine 相比 CoT 的修复与退化情况。"""
    cot_to_refine = compare_methods(records, "baseline_cot", "self_refine")
    variants = [
        method
        for method in ("self_refine", "self_refine_calculator", "self_refine_gate")
        if any(record["method"] == method for record in records)
    ]
    by_round = {
        method: compare_methods(records, "baseline_cot", method)
        for method in variants
    }
    direct_to_cot = compare_methods(records, "baseline_direct", "baseline_cot")
    if not variants or direct_to_cot is None:
        return None

    direct = {
        item["id"]: item
        for item in records
        if item["method"] == "baseline_direct"
    }
    cot = {
        item["id"]: item
        for item in records
        if item["method"] == "baseline_cot"
    }
    refine = {
        item["id"]: item
        for item in records
        if item["method"] == "self_refine"
    }

    # 检查 Self-Refine 是否破坏了 CoT 已经修复的题目。
    cot_fixes = [
        question_id
        for question_id in direct
        if not direct[question_id]["correct"] and cot[question_id]["correct"]
    ]
    report: dict[str, Any] = {
        "cot_to_self_refine": cot_to_refine,
        "by_round": by_round,
    }
    if refine:
        report["on_cot_fixes"] = {
            "total": len(cot_fixes),
            "kept_correct": sum(
                bool(refine[item_id]["correct"]) for item_id in cot_fixes
            ),
            "regressed": sum(
                not refine[item_id]["correct"] for item_id in cot_fixes
            ),
        }
    return report


def reflection_report(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """比较按时间与按语义检索的 Reflection，且保留旧实验的兼容统计。"""
    recent = compare_methods(records, "baseline_cot", "reflection_recent")
    embedding = compare_methods(records, "baseline_cot", "reflection_embedding")
    embedding_correct = compare_methods(
        records, "baseline_cot", "reflection_embedding_correct_embedding"
    )
    if recent is not None or embedding is not None or embedding_correct is not None:
        return {
            "vs_cot": {
                "recent": recent,
                "embedding": embedding,
                "embedding_correct": embedding_correct,
            },
            "embedding_vs_recent": compare_methods(
                records, "reflection_recent", "reflection_embedding"
            ),
            "embedding_correct_vs_embedding": compare_methods(
                records,
                "reflection_embedding",
                "reflection_embedding_correct_embedding",
            ),
        }
    return compare_methods(records, "baseline_cot", "reflection")


def write_failure_review(
    path: Path, records: list[dict[str, Any]], summary: dict[str, Any]
) -> None:
    """生成最多包含五个失败样本的 Markdown 报告。"""
    failures = [record for record in records if not record["correct"]]
    lines = [
        "# Baseline Failure Review",
        "",
        f"- Method: `{summary['method']}`",
        f"- Evaluated: {summary['total']} GSM8K test examples "
        f"(fixed seed: {summary['dataset_seed']})",
        f"- Accuracy: {summary['accuracy']:.1%} "
        f"({summary['correct']}/{summary['total']})",
        f"- Failure cases below: {min(len(failures), 5)} of {len(failures)}",
        "",
        "## Interpretation boundary",
        "",
        "A one-pass baseline only establishes that the final answer is wrong. "
        "It cannot prove whether the model lacked the reasoning capability or "
        "had a recoverable mistake that it failed to check. That distinction "
        "requires a later independent critique or tool-feedback trace; do not "
        "infer it from the gold answer during generation.",
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
                "- Initial classification: `needs critique/verification`; "
                "baseline evidence alone cannot separate capability failure "
                "from missed checking.",
                "",
                "Model response:",
                "",
                "```text",
                "\n".join(
                    line.rstrip()
                    for line in record["raw_response"].strip().splitlines()
                ),
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    """执行评测、保存记录并生成汇总结果。"""
    if args.method == "self_refine" and args.rounds != 1:
        raise ValueError("Self-Refine comparison supports only --rounds 1.")
    if args.method == "reflection" and args.workers != 1:
        raise ValueError("Reflection 必须使用 --workers 1，保证 memory 写入顺序可复现。")
    if args.method == "reflection" and args.memory_top_k < 1:
        raise ValueError("Reflection 的 --memory-top-k 必须至少为 1。")
    if args.method == "reflection" and args.correct_top_k < 1:
        raise ValueError("Reflection 的 --correct-top-k 必须至少为 1。")
    examples = load_dataset(args.dataset)
    if args.limit:
        examples = examples[: args.limit]
    if not examples:
        raise ValueError("No evaluation examples found.")

    # provider 只决定调用协议，模型配置会写入 summary。
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

    if args.method == "baseline":
        method = f"baseline_{args.mode}"
        agent = BaselineAgent(client, mode=args.mode)
    elif args.method == "self_refine":
        method = {
            "original": "self_refine",
            "calculator": "self_refine_calculator",
            "decision_gate": "self_refine_gate",
        }[args.self_refine_mode]
        agent = SelfRefineAgent(
            client,
            max_rounds=args.rounds,
            mode=args.self_refine_mode,
        )
    elif args.method == "critic":
        # CRITIC 只依赖本题的计算器反馈，因此可安全并发运行。
        method = "critic"
        agent = CriticAgent(client)
    else:
        method = f"reflection_{args.memory_retrieval}"
        if args.correct_examples != "none":
            method = f"{method}_correct_{args.correct_examples}"
        agent = ReflectionAgent(
            client,
            memory_limit=args.memory_top_k,
            retrieval=args.memory_retrieval,
            correct_examples=args.correct_examples,
            correct_top_k=args.correct_top_k,
            embedding_model=args.embedding_model,
        )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.results_dir / RECORDS_FILENAME
    summary_path = args.results_dir / SUMMARY_FILENAME

    if args.reset_results and summary_path.exists():
        summary_path.unlink()
    all_records = [] if args.reset_results else load_existing_records(records_path)

    # 防止不同模型配置混入同一组实验结果。
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

    # 保留其他方法结果；--resume 时恢复当前方法结果。
    other_records = [
        record for record in all_records if record.get("method") != method
    ]
    records = (
        [
            record
            for record in all_records
            if record.get("method") == method
        ]
        if args.resume
        else []
    )
    completed_ids = {record["id"] for record in records}
    if args.method == "reflection" and records:
        # --resume 会创建新 agent，必须从已有 trace 恢复记忆才与一次性运行等价。
        agent.restore_memory(
            [
                Reflection(
                    question_id=record["id"],
                    lesson=step["lesson_written"],
                    question=record["question"],
                )
                for record in records
                for step in record["trace"]["steps"]
                if "lesson_written" in step
            ]
        )
        agent.restore_correct_examples(
            [
                CorrectExample(
                    question_id=record["id"],
                    question=record["question"],
                    solution=record["raw_response"],
                    answer=record["prediction"] or "",
                )
                for record in records
                if record.get("correct")
            ]
        )
    if completed_ids:
        print(
            f"Resuming with {len(completed_ids)} completed examples.",
            flush=True,
        )

    pending_examples = [
        example for example in examples if example["id"] not in completed_ids
    ]
    if args.max_new is not None:
        pending_examples = pending_examples[: args.max_new]

    reflection_log = None
    if args.method == "reflection":
        args.reflection_log.parent.mkdir(parents=True, exist_ok=True)
        reflection_log = args.reflection_log.open("a", encoding="utf-8")

    def evaluate_example(example: dict[str, Any]) -> dict[str, Any]:
        """评测一道题，标准答案不会传入 Agent。"""
        raw_response, trace = agent.solve(
            example["question"], question_id=example["id"]
        )
        prediction = extract_gsm8k_answer(raw_response)
        correct = exact_match(prediction, example["answer"])
        trace.final_answer = prediction or ""
        trace.final_correct = correct
        if args.method == "reflection":
            if correct and args.correct_examples != "none":
                # 只有当前题已被外部判对后，才允许进入后续题的正确样例库。
                correct_example = agent.remember_correct(
                    example["question"],
                    raw_response,
                    prediction or "",
                    question_id=example["id"],
                )
                trace.steps.append(
                    {
                        "round": 1,
                        "feedback": "correct",
                        "feedback_source": "external_evaluator",
                        "correct_example_written": correct_example.question_id,
                    }
                )
            elif not correct:
                # 评测器只传递“错误”这一信号，不把参考答案泄漏给 agent。
                reflection = agent.reflect(
                    example["question"], raw_response, question_id=example["id"]
                )
                trace.steps.append(
                    {
                        "round": 1,
                        "feedback": "incorrect",
                        "feedback_source": "external_evaluator",
                        "lesson_written": reflection.lesson,
                    }
                )
                if reflection_log:
                    reflection_log.write(
                        json.dumps(
                            {
                                "question_id": example["id"],
                                "lesson": reflection.lesson,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    reflection_log.flush()
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
        # 先回写旧记录，确保中断后可以继续运行。
        for record in other_records + records:
            records_handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(evaluate_example, example)
                for example in pending_examples
            ]
            for completed, future in enumerate(as_completed(futures), start=1):
                record = future.result()
                records.append(record)
                records_handle.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )
                records_handle.flush()
                print(
                    f"[{len(completed_ids) + completed}/{len(examples)}] "
                    f"{record['id']}: "
                    f"{'correct' if record['correct'] else 'wrong'}",
                    flush=True,
                )

    if reflection_log:
        reflection_log.close()

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
        "feedback_source": {
            "baseline": "none",
            "self_refine": "self",
            "reflection": "external_evaluator_memory",
            "critic": "calculator",
        }[args.method],
    }
    if args.method == "self_refine":
        summary["rounds"] = args.rounds
        summary["self_refine_mode"] = args.self_refine_mode
        summary["feedback_source"] = {
            "original": "self",
            "calculator": "self_with_calculator_gate",
            "decision_gate": "self_with_decision_gate",
        }[args.self_refine_mode]
    if args.method == "reflection":
        summary.update(
            {
                "memory_retrieval": args.memory_retrieval,
                "memory_top_k": args.memory_top_k,
                "correct_examples": args.correct_examples,
                "correct_top_k": (
                    args.correct_top_k
                    if args.correct_examples != "none"
                    else None
                ),
                "embedding_model": (
                    args.embedding_model
                    if (
                        args.memory_retrieval == "embedding"
                        or args.correct_examples == "embedding"
                    )
                    else None
                ),
            }
        )

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
        "reflection": reflection_report(other_records + records),
        "updated_at": datetime.now(UTC).isoformat(),
    }
    summary_path.write_text(
        json.dumps(combined_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # 固定保留 CoT baseline 的失败案例，避免后续方法覆盖第 1 周产物。
    if summary["complete"] and args.method == "baseline" and args.mode == "cot":
        write_failure_review(args.failure_review, records, summary)
    return summary


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--method",
        choices=("baseline", "self_refine", "reflection", "critic"),
        default="baseline",
    )
    parser.add_argument(
        "--provider",
        choices=("openai", "anthropic", "local"),
        default="local",
    )
    parser.add_argument("--mode", choices=("direct", "cot"), default="cot")
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Self-Refine comparison supports only 1 round.",
    )
    parser.add_argument(
        "--self-refine-mode",
        choices=("original", "calculator", "decision_gate"),
        default="original",
        help=(
            "Compare original, calculator-gated, and decision-gated "
            "Self-Refine."
        ),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--max-new",
        type=int,
        default=None,
        help="At most this many new examples per invocation.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent model requests; use conservatively.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Overrides the provider's configured model.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Overrides the provider's configured base URL.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument(
        "--thinking",
        choices=("enabled", "disabled"),
        default="disabled",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
    )
    parser.add_argument(
        "--reset-results",
        action="store_true",
        help="Discard existing records before starting a different configuration.",
    )
    parser.add_argument(
        "--failure-review",
        type=Path,
        default=DEFAULT_FAILURE_REVIEW,
    )
    parser.add_argument(
        "--reflection-log",
        type=Path,
        default=DEFAULT_REFLECTION_LOG,
        help="Append generated lessons here when using --method reflection.",
    )
    parser.add_argument(
        "--memory-retrieval",
        choices=("recent", "embedding"),
        default="recent",
        help="Reflection lesson retrieval strategy.",
    )
    parser.add_argument(
        "--memory-top-k",
        type=int,
        default=3,
        help="Number of Reflection lessons injected into each solve prompt.",
    )
    parser.add_argument(
        "--correct-examples",
        choices=("none", "embedding"),
        default="none",
        help=(
            "Retrieve earlier externally verified correct solutions as "
            "few-shot examples for Reflection."
        ),
    )
    parser.add_argument(
        "--correct-top-k",
        type=int,
        default=3,
        help="Number of verified correct examples injected into each prompt.",
    )
    parser.add_argument(
        "--embedding-model",
        default="local-hash",
        help=(
            "Embedding model for reflection retrieval. Use local-hash for "
            "offline retrieval, or a Sentence-Transformers model/path."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted run from existing JSONL records.",
    )
    return parser.parse_args()


def main() -> None:
    """程序入口。"""
    args = parse_args()
    try:
        summary = run(args)
    except (LLMConfigurationError, ValueError) as error:
        raise SystemExit(f"Evaluation was not started: {error}") from error

    state = (
        "complete"
        if summary["complete"]
        else f"partial, target {summary['target_total']}"
    )
    print(
        f"Accuracy ({summary['method']}, {state}): "
        f"{summary['accuracy']:.1%} "
        f"({summary['correct']}/{summary['total']})"
    )


if __name__ == "__main__":
    main()
