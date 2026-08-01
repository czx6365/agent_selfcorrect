"""Export normalized solve traces from evaluation result files."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MATH_RECORDS = PROJECT_ROOT / "results" / "baseline_records.jsonl"
DEFAULT_CODE_RECORDS = PROJECT_ROOT / "results" / "code_records.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "logs" / "solve_trace.jsonl"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_math(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_type": "math",
        "id": record["id"],
        "method": record["method"],
        "question": record["question"],
        "prediction": record.get("prediction"),
        "correct": bool(record.get("correct")),
        "feedback_sources": sorted(
            {
                step.get("feedback_source", "unknown")
                for step in record.get("trace", {}).get("steps", [])
            }
        ),
        "trace": record.get("trace", {}),
    }


def normalize_code(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_type": "code",
        "id": record["id"],
        "method": record["method"],
        "entry_point": record.get("entry_point"),
        "passed": bool(record.get("passed")),
        "status": record.get("status"),
        "feedback_sources": [
            "unit_tests" if record["method"].startswith(("code_self_repair", "code_cot_repair")) else "none"
        ],
        "trace": record.get("trace", {}),
    }


def export_logs(
    *,
    math_records: Path,
    code_records: Path,
    output: Path,
    min_method_records: int = 100,
    include_invalid_r3: bool = False,
) -> int:
    records: list[dict[str, Any]] = []
    raw_math_records = load_jsonl(math_records)
    raw_code_records = load_jsonl(code_records)
    complete_math_methods = {
        method
        for method, count in Counter(record["method"] for record in raw_math_records).items()
        if count >= min_method_records
    }
    complete_code_methods = {
        method
        for method, count in Counter(record["method"] for record in raw_code_records).items()
        if count >= min_method_records
    }
    records.extend(
        normalize_math(record)
        for record in raw_math_records
        if record["method"] in complete_math_methods
    )
    for record in raw_code_records:
        if record["method"] not in complete_code_methods:
            continue
        if not include_invalid_r3 and record.get("method") == "code_self_repair_r3":
            # This method is omitted until rerun with an available LLM endpoint.
            continue
        records.append(normalize_code(record))

    records.sort(key=lambda item: (item["task_type"], item["method"], item["id"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--math-records", type=Path, default=DEFAULT_MATH_RECORDS)
    parser.add_argument("--code-records", type=Path, default=DEFAULT_CODE_RECORDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--include-invalid-r3",
        action="store_true",
        help="Include the failed code_self_repair_r3 run captured during LLM outage.",
    )
    parser.add_argument(
        "--min-method-records",
        type=int,
        default=100,
        help="Only export methods with at least this many records.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = export_logs(
        math_records=args.math_records,
        code_records=args.code_records,
        output=args.output,
        min_method_records=args.min_method_records,
        include_invalid_r3=args.include_invalid_r3,
    )
    print(f"Wrote {count} trace records to {args.output}")


if __name__ == "__main__":
    main()
