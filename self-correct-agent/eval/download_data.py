from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from pathlib import Path


GSM8K_FILES = {
    "train": "https://huggingface.co/datasets/openai/gsm8k/resolve/main/main/train-00000-of-00001.parquet",
    "test": "https://huggingface.co/datasets/openai/gsm8k/resolve/main/main/test-00000-of-00001.parquet",
}

HUMANEVAL_FILES = {
    "test": "https://huggingface.co/datasets/openai/openai_humaneval/resolve/main/openai_humaneval/test-00000-of-00001.parquet",
}


def download(url: str, target: Path) -> None:
    # 统一保留原始 parquet，避免数据转换时丢失官方字段。
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, target.open("wb") as f:
        shutil.copyfileobj(response, f)


def write_manifest(root: Path, dataset: str, files: dict[str, str]) -> None:
    # manifest 记录来源 URL 和 split，便于实验复现与数据审计。
    manifest = {
        "dataset": dataset,
        "files": [
            {"split": split, "url": url, "path": f"{dataset}/{split}.parquet"}
            for split, url in files.items()
        ],
    }
    (root / f"{dataset}_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent / "datasets",
    )
    parser.add_argument(
        "--dataset",
        choices=["gsm8k", "humaneval", "all"],
        default="all",
    )
    args = parser.parse_args()

    root = args.root
    root.mkdir(parents=True, exist_ok=True)

    if args.dataset in {"gsm8k", "all"}:
        dataset_root = root / "gsm8k"
        for split, url in GSM8K_FILES.items():
            download(url, dataset_root / f"{split}.parquet")
        write_manifest(root, "gsm8k", GSM8K_FILES)

    if args.dataset in {"humaneval", "all"}:
        dataset_root = root / "humaneval"
        for split, url in HUMANEVAL_FILES.items():
            download(url, dataset_root / f"{split}.parquet")
        write_manifest(root, "humaneval", HUMANEVAL_FILES)


if __name__ == "__main__":
    main()
