"""Answer parsing and aggregate metrics for GSM8K evaluation."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable


NUMBER_PATTERN = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
ANSWER_TOKEN = r"([-+]?\d[\d,]*(?:\.\d+)?(?:\s*/\s*[-+]?\d[\d,]*(?:\.\d+)?)?)"
FINAL_PATTERN = re.compile(rf"(?:####|FINAL\s*:?)\s*{ANSWER_TOKEN}", re.IGNORECASE)


def _normalize_number_token(token: str) -> str | None:
    token = token.replace(",", "").replace(" ", "")
    if "/" not in token:
        return token
    numerator, denominator = token.split("/", maxsplit=1)
    try:
        denominator_value = Decimal(denominator)
        if denominator_value == 0:
            return None
        return str(Decimal(numerator) / denominator_value)
    except InvalidOperation:
        return None


def extract_gsm8k_answer(text: str) -> str | None:
    """Extract the final number from a GSM8K answer or a model completion."""
    match = FINAL_PATTERN.search(text)
    if match:
        return _normalize_number_token(match.group(1))

    numbers = NUMBER_PATTERN.findall(text)
    return numbers[-1].replace(",", "") if numbers else None


def exact_match(prediction: str | None, answer: str | None) -> bool:
    """Numerically compare parsed answers, accepting harmless decimal formatting."""
    if prediction is None or answer is None:
        return False
    try:
        return Decimal(prediction.replace(",", "")) == Decimal(answer.replace(",", ""))
    except (InvalidOperation, AttributeError):
        return prediction.strip() == answer.strip()


def accuracy(records: Iterable[dict[str, Any]]) -> dict[str, int | float]:
    records = list(records)
    correct = sum(bool(record["correct"]) for record in records)
    total = len(records)
    return {
        "total": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": correct / total if total else 0.0,
    }
