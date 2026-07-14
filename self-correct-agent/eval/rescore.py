"""Re-parse existing model outputs after changing the deterministic scorer."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from eval.metrics import accuracy, exact_match, extract_gsm8k_answer
from eval.run_eval import write_failure_review, write_jsonl


def rescore(records_path: Path, summary_path: Path, failure_review: Path | None) -> None:
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line]
    for record in records:
        record["prediction"] = extract_gsm8k_answer(record["raw_response"])
        record["correct"] = exact_match(record["prediction"], record["expected_answer"])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(accuracy(records))
    summary["rescored_at"] = datetime.now(UTC).isoformat()
    write_jsonl(records_path, records)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if failure_review:
        write_failure_review(failure_review, records, summary)
    print(f"Rescored {records_path.name}: {summary['accuracy']:.1%} ({summary['correct']}/{summary['total']})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--failure-review", type=Path)
    args = parser.parse_args()
    rescore(args.records, args.summary, args.failure_review)


if __name__ == "__main__":
    main()
