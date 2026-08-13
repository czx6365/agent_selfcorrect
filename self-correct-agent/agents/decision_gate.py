"""Decision gate for deciding whether a Self-Refine revision is worth using."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

from eval.metrics import extract_gsm8k_answer

from .base import CompletionClient
from .self_refine import extract_arithmetic_evidence


CONFIDENCE_PROMPT = """Compare the draft and revised math solutions below.
Do not use a reference answer. Judge only whether the revision is better
supported by the problem statement and critique.

Return JSON only:
{{
  "draft_confidence": <integer 0-100>,
  "revised_confidence": <integer 0-100>,
  "revision_risk": "low" | "medium" | "high",
  "reason": "<one short reason>"
}}

Problem:
{question}

Draft solution:
{draft}

Critique:
{critique}

Revised solution:
{revised}
"""

VAGUE_PATTERNS = (
    "may be",
    "might be",
    "could be",
    "not sure",
    "check carefully",
    "review the arithmetic",
    "needs to be checked",
    "possibly",
)

SPECIFIC_MARKERS = (
    "CHECK:",
    "CLAIMED:",
    "REASON:",
    "because",
    "instead of",
    "should be",
    "the draft",
)


@dataclass
class GateDecision:
    """A transparent accept/reject decision for one candidate revision."""

    decision: Literal["accept", "reject"]
    score: int
    reasons: list[str]
    features: dict[str, Any]


def _parse_decimal(value: str | None) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(value.replace(",", ""))
    except (InvalidOperation, AttributeError):
        return None


def _answers_changed(draft: str, revised: str) -> bool | None:
    draft_answer = _parse_decimal(extract_gsm8k_answer(draft))
    revised_answer = _parse_decimal(extract_gsm8k_answer(revised))
    if draft_answer is None or revised_answer is None:
        return None
    return draft_answer != revised_answer


def _critique_is_specific(critique: str) -> bool:
    normalized = critique.strip().lower()
    if not normalized or normalized == "no_change":
        return False
    has_marker = any(marker.lower() in normalized for marker in SPECIFIC_MARKERS)
    has_arithmetic = bool(re.search(r"\d+\s*[-+*/]\s*\d+", normalized))
    has_vague_language = any(pattern in normalized for pattern in VAGUE_PATTERNS)
    return (has_marker or has_arithmetic) and not has_vague_language


def _extract_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _confidence_judgment(
    question: str,
    draft: str,
    critique: str,
    revised: str,
    client: CompletionClient | None,
) -> dict[str, Any]:
    if client is None:
        return {"status": "skipped"}
    response = client.complete(
        CONFIDENCE_PROMPT.format(
            question=question,
            draft=draft,
            critique=critique,
            revised=revised,
        )
    )
    parsed = _extract_json_object(response)
    if not parsed:
        return {"status": "invalid", "raw_response": response}
    return {
        "status": "ok",
        "draft_confidence": parsed.get("draft_confidence"),
        "revised_confidence": parsed.get("revised_confidence"),
        "revision_risk": parsed.get("revision_risk"),
        "reason": parsed.get("reason"),
    }


def decide_revision(
    *,
    question: str,
    draft: str,
    critique: str,
    revised: str,
    client: CompletionClient | None = None,
    use_confidence: bool = True,
) -> GateDecision:
    """Accept a revision only when evidence outweighs regression risk."""

    score = 0
    reasons: list[str] = []
    critique_text = critique.strip()

    if critique_text == "NO_CHANGE":
        return GateDecision(
            decision="reject",
            score=-99,
            reasons=["critique requested no change"],
            features={"critique": "no_change"},
        )

    evidence = extract_arithmetic_evidence(critique)
    answers_changed = _answers_changed(draft, revised)
    critique_specific = _critique_is_specific(critique)
    confidence = (
        _confidence_judgment(question, draft, critique, revised, client)
        if use_confidence
        else {"status": "disabled"}
    )

    if evidence["status"] == "verified_mismatch":
        score += 4
        reasons.append("calculator verified an arithmetic mismatch")
    elif evidence["status"] not in {"no_change", "missing_evidence"}:
        score -= 1
        reasons.append(f"arithmetic evidence was not usable: {evidence['status']}")

    if critique_specific:
        score += 1
        reasons.append("critique names a specific issue")
    else:
        score -= 2
        reasons.append("critique is vague or underspecified")

    if answers_changed is True and evidence["status"] != "verified_mismatch":
        score -= 2
        reasons.append("final answer changed without verified evidence")
    elif answers_changed is False:
        score += 1
        reasons.append("revision does not change the extracted final answer")

    if confidence.get("status") == "ok":
        try:
            draft_confidence = int(confidence.get("draft_confidence"))
            revised_confidence = int(confidence.get("revised_confidence"))
        except (TypeError, ValueError):
            draft_confidence = revised_confidence = None

        if draft_confidence is not None and revised_confidence is not None:
            if revised_confidence - draft_confidence >= 20:
                score += 1
                reasons.append("judge confidence favors the revision")
            elif draft_confidence - revised_confidence >= 20:
                score -= 1
                reasons.append("judge confidence favors the draft")

        if confidence.get("revision_risk") == "high":
            score -= 2
            reasons.append("judge marked the revision as high risk")

    decision: Literal["accept", "reject"] = "accept" if score >= 2 else "reject"
    return GateDecision(
        decision=decision,
        score=score,
        reasons=reasons,
        features={
            "evidence": evidence,
            "answers_changed": answers_changed,
            "critique_specific": critique_specific,
            "confidence": confidence,
        },
    )
