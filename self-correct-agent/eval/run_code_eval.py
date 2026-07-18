"""Run HumanEval code-generation experiments with unit-test feedback."""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.llm_client import (
    AnthropicCompatibleClient,
    LLMConfigurationError,
    LocalLlamaServerClient,
    OpenAICompatibleClient,
)
from tools.code_runner import run_unit_tests


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "datasets" / "humaneval" / "test.parquet"
DEFAULT_RESULTS_DIR = ROOT / "results"
RECORDS_FILENAME = "code_records.jsonl"
SUMMARY_FILENAME = "code_summary.json"


DIRECT_PROMPT = """Complete the Python function below.

Return only valid Python code for the complete function. Do not include Markdown
fences, explanations, tests, or examples.

{prompt}
"""


COT_PROMPT = """Complete the Python function below.

First reason briefly in comments inside the code if useful, then return valid
Python code for the complete function. Do not include Markdown fences,
explanations outside code, tests, or examples.

{prompt}
"""


REPAIR_PROMPT = """The Python function below failed its unit tests.

Original problem:
{prompt}

Previous code:
```python
{code}
```

Unit-test feedback:
```text
{feedback}
```

Return only corrected valid Python code for the complete function. Do not include
Markdown fences, explanations, tests, or examples.
"""


def load_humaneval(path: Path) -> list[dict[str, Any]]:
    """Load official HumanEval parquet rows while preserving source fields."""
    try:
        import pandas as pd
    except ImportError as error:
        raise RuntimeError("HumanEval loading needs pandas and pyarrow.") from error
    frame = pd.read_parquet(path)
    records: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        records.append(
            {
                "id": str(row["task_id"]),
                "prompt": str(row["prompt"]),
                "test": str(row["test"]),
                "entry_point": str(row["entry_point"]),
            }
        )
    return records


def make_client(args: argparse.Namespace):
    """Create the same provider clients used by the GSM8K evaluator."""
    if args.provider == "anthropic":
        return AnthropicCompatibleClient(
            model=args.model,
            base_url=args.base_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            thinking=args.thinking,
        )
    if args.provider == "local":
        return LocalLlamaServerClient(
            model=args.model,
            base_url=args.base_url,
            temperature=args.temperature,
            seed=args.seed,
            max_tokens=args.max_tokens,
        )
    return OpenAICompatibleClient(
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        seed=args.seed,
        max_tokens=args.max_tokens,
    )


def extract_code(response: str, prompt: str, entry_point: str) -> str:
    """Extract a complete function from chatty model output."""
    text = response.strip("\n")
    fenced = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.S | re.I)
    if fenced:
        text = fenced.group(1).strip()

    # Some models return the prompt plus a body. Keep the first target function.
    function_match = re.search(
        rf"(?ms)^def\s+{re.escape(entry_point)}\s*\(.*?(?=^\S|\Z)",
        text,
    )
    if function_match:
        return function_match.group(0).rstrip() + "\n"

    # HumanEval prompts end after the function docstring; a body-only completion
    # can be appended to the official prompt.
    body_text = text.strip("\n")
    body = (
        body_text
        if all(not line.strip() or line.startswith((" ", "\t")) for line in body_text.splitlines())
        else textwrap.indent(body_text.strip(), "    ")
    )
    return prompt.rstrip() + "\n" + body + "\n"


def solve_once(
    client: Any,
    example: dict[str, Any],
    *,
    mode: str,
    max_repair_rounds: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Generate code, run tests, and optionally repair with external feedback."""
    template = COT_PROMPT if mode in {"cot", "cot_repair"} else DIRECT_PROMPT
    response = client.complete(template.format(prompt=example["prompt"]))
    code = extract_code(response, example["prompt"], example["entry_point"])
    verdict = run_unit_tests(
        prompt=example["prompt"],
        code=code,
        test=example["test"],
        entry_point=example["entry_point"],
        timeout_seconds=timeout_seconds,
    )
    steps = [
        {
            "round": 0,
            "mode": mode,
            "response": response,
            "code": code,
            "verdict": verdict,
        }
    ]

    repair_round = 0
    while (
        mode in {"self_repair", "cot_repair"}
        and not verdict["passed"]
        and repair_round < max_repair_rounds
    ):
        repair_round += 1
        response = client.complete(
            REPAIR_PROMPT.format(
                prompt=example["prompt"],
                code=code,
                feedback=verdict["feedback"] or verdict["status"],
            )
        )
        code = extract_code(response, example["prompt"], example["entry_point"])
        verdict = run_unit_tests(
            prompt=example["prompt"],
            code=code,
            test=example["test"],
            entry_point=example["entry_point"],
            timeout_seconds=timeout_seconds,
        )
        steps.append(
            {
                "round": repair_round,
                "mode": "repair",
                "response": response,
                "code": code,
                "verdict": verdict,
            }
        )

    method = f"code_{mode}"
    if mode in {"self_repair", "cot_repair"} and max_repair_rounds != 1:
        method = f"{method}_r{max_repair_rounds}"
    return {
        "method": method,
        "id": example["id"],
        "entry_point": example["entry_point"],
        "passed": verdict["passed"],
        "status": verdict["status"],
        "feedback": verdict["feedback"],
        "code": code,
        "trace": {"method": method, "steps": steps},
    }


def load_existing(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    passed = sum(bool(record["passed"]) for record in records)
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": passed / total if total else 0.0,
    }


def compare_methods(
    records: list[dict[str, Any]], before: str, after: str
) -> dict[str, Any] | None:
    """Compare two code methods on shared HumanEval task ids."""
    before_records = {
        record["id"]: record for record in records if record.get("method") == before
    }
    after_records = {
        record["id"]: record for record in records if record.get("method") == after
    }
    if not before_records or before_records.keys() != after_records.keys():
        return None
    result = {"kept_pass": 0, "fixed": 0, "regressed": 0, "failed_both": 0}
    for task_id, before_record in before_records.items():
        before_passed = bool(before_record["passed"])
        after_passed = bool(after_records[task_id]["passed"])
        key = {
            (True, True): "kept_pass",
            (False, True): "fixed",
            (True, False): "regressed",
            (False, False): "failed_both",
        }[(before_passed, after_passed)]
        result[key] += 1
    return result


def best_of_available(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Report pass rate if unit tests select any passing candidate per task."""
    by_method: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        by_method.setdefault(record["method"], {})[record["id"]] = record
    if len(by_method) < 2:
        return None
    common_ids = set.intersection(
        *(set(items.keys()) for items in by_method.values())
    )
    if not common_ids:
        return None
    passed = sum(
        any(method_records[task_id]["passed"] for method_records in by_method.values())
        for task_id in common_ids
    )
    return {
        "methods": sorted(by_method),
        "total": len(common_ids),
        "passed": passed,
        "failed": len(common_ids) - passed,
        "pass_rate": passed / len(common_ids),
    }


def build_comparison(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the code-method comparison table stored in code_summary.json."""
    return {
        "direct_to_cot": compare_methods(records, "code_direct", "code_cot"),
        "direct_to_self_repair": compare_methods(
            records, "code_direct", "code_self_repair"
        ),
        "direct_to_cot_repair": compare_methods(
            records, "code_direct", "code_cot_repair"
        ),
        "cot_to_self_repair": compare_methods(
            records, "code_cot", "code_self_repair"
        ),
        "cot_to_cot_repair": compare_methods(records, "code_cot", "code_cot_repair"),
        "self_repair_to_cot_repair": compare_methods(
            records, "code_self_repair", "code_cot_repair"
        ),
        "best_of_available": best_of_available(records),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.mode in {"self_repair", "cot_repair"} and args.workers != 1:
        raise ValueError("repair modes should use --workers 1 for readable traces.")
    examples = load_humaneval(args.source)
    if args.limit:
        examples = examples[: args.limit]
    if not examples:
        raise ValueError("No HumanEval examples found.")

    client = make_client(args)
    method = f"code_{args.mode}"
    if args.mode in {"self_repair", "cot_repair"} and args.repair_rounds != 1:
        method = f"{method}_r{args.repair_rounds}"
    args.results_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.results_dir / RECORDS_FILENAME
    summary_path = args.results_dir / SUMMARY_FILENAME

    all_records = [] if args.reset_results else load_existing(records_path)
    other_records = [
        record for record in all_records if record.get("method") != method
    ]
    records = (
        [record for record in all_records if record.get("method") == method]
        if args.resume
        else []
    )
    completed_ids = {record["id"] for record in records}
    pending = [example for example in examples if example["id"] not in completed_ids]

    def evaluate(example: dict[str, Any]) -> dict[str, Any]:
        try:
            return solve_once(
                client,
                example,
                mode=args.mode,
                max_repair_rounds=args.repair_rounds,
                timeout_seconds=args.timeout_seconds,
            )
        except Exception as error:
            return {
                "method": method,
                "id": example["id"],
                "entry_point": example["entry_point"],
                "passed": False,
                "status": "error",
                "feedback": f"{type(error).__name__}: {error}",
                "code": "",
                "trace": {"method": method, "steps": []},
            }

    with records_path.open("w", encoding="utf-8") as handle:
        for record in other_records + records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(evaluate, example) for example in pending]
            for completed, future in enumerate(as_completed(futures), start=1):
                record = future.result()
                records.append(record)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                print(
                    f"[{len(completed_ids) + completed}/{len(examples)}] "
                    f"{record['id']}: {'pass' if record['passed'] else record['status']}",
                    flush=True,
                )

    summary = {
        **summarize(records),
        "method": method,
        "dataset": str(args.source),
        "provider": args.provider,
        "llm_model": client.model,
        "temperature": args.temperature,
        "llm_seed": args.seed,
        "max_tokens": getattr(client, "max_tokens", None),
        "repair_rounds": (
            args.repair_rounds
            if args.mode in {"self_repair", "cot_repair"}
            else 0
        ),
        "feedback_source": (
            "unit_tests" if args.mode in {"self_repair", "cot_repair"} else "none"
        ),
        "timeout_seconds": args.timeout_seconds,
        "complete": len(records) == len(examples),
        "target_total": len(examples),
        "completed_at": datetime.now(UTC).isoformat(),
    }
    combined = {} if args.reset_results or not summary_path.exists() else json.loads(
        summary_path.read_text(encoding="utf-8")
    )
    methods = combined.get("methods", {})
    methods[method] = summary
    combined = {
        "dataset": str(args.source),
        "methods": methods,
        "comparison": build_comparison(other_records + records),
        "updated_at": datetime.now(UTC).isoformat(),
    }
    summary_path.write_text(
        json.dumps(combined, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--mode",
        choices=("direct", "cot", "self_repair", "cot_repair"),
        default="direct",
    )
    parser.add_argument(
        "--provider",
        choices=("openai", "anthropic", "local"),
        default="local",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--thinking", choices=("enabled", "disabled"), default="disabled")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repair-rounds", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=3.0)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--reset-results", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        summary = run(args)
    except (LLMConfigurationError, ValueError) as error:
        raise SystemExit(f"Code evaluation was not started: {error}") from error
    state = "complete" if summary["complete"] else f"partial, target {summary['target_total']}"
    print(
        f"Pass rate ({summary['method']}, {state}): "
        f"{summary['pass_rate']:.1%} ({summary['passed']}/{summary['total']})",
        flush=True,
    )


if __name__ == "__main__":
    main()
