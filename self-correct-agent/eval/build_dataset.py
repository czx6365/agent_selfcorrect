"""Create the fixed 100-question GSM8K subset used by all weekly experiments."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "datasets" / "gsm8k" / "test.parquet"
DEFAULT_OUTPUT = ROOT / "dataset.jsonl"


def build_dataset(source: Path, output: Path, sample_size: int, seed: int) -> None:
    try:
        import pandas as pd
    except ImportError as error:
        raise RuntimeError(
            "Dataset creation needs pandas and pyarrow. Install requirements in a virtual environment."
        ) from error

    frame = pd.read_parquet(source)
    if sample_size > len(frame):
        raise ValueError(f"Requested {sample_size} examples, but source only has {len(frame)}")

    source_indices = random.Random(seed).sample(range(len(frame)), sample_size)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for dataset_index, source_index in enumerate(source_indices):
            row = frame.iloc[source_index]
            answer_raw = str(row["answer"])
            final_answer = answer_raw.rsplit("####", maxsplit=1)[-1].strip().replace(",", "")
            record = {
                "id": f"gsm8k_test_{dataset_index:03d}",
                "source_index": int(source_index),
                "question": str(row["question"]),
                "answer": final_answer,
                "answer_raw": answer_raw,
                "dataset": "gsm8k",
                "split": "test",
                "sampling_seed": seed,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_dataset(args.source, args.output, args.sample_size, args.seed)
    print(f"Wrote {args.sample_size} examples to {args.output}")


if __name__ == "__main__":
    main()
