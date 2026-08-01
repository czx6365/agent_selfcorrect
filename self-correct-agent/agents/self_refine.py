from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any

from tools.calculator import calculate

from .base import BaseAgent, CHAIN_OF_THOUGHT_PROMPT, CompletionClient, SolveTrace


ORIGINAL_CRITIQUE_PROMPT = """Review the proposed solution to the math problem below.
Check every interpretation and arithmetic operation. State one concrete error if one exists.
If the solution is sound, respond with exactly `NO_CHANGE`.
Do not use any external answer and do not give a replacement final answer.

Problem:
{question}

Proposed solution:
{draft}
"""

ORIGINAL_REVISION_PROMPT = """Solve the math problem again using the draft and critique below.
Keep the draft unchanged when the critique says `NO_CHANGE`; otherwise repair only the identified issue.
Show the arithmetic needed to solve it. End with the exact line `FINAL: <number>`.

Problem:
{question}

Draft:
{draft}

Critique:
{critique}
"""

CALCULATOR_CRITIQUE_PROMPT = """Review the proposed solution to the math problem below.
Only request a revision when you can point to a concrete arithmetic operation
from the proposed solution that can be verified with a calculator.

If there is no calculator-checkable arithmetic error, respond with exactly:
NO_CHANGE

If there is a calculator-checkable arithmetic error, respond in exactly this
format:
REVISE
CHECK: <one arithmetic expression using only numbers, +, -, *, /, and parentheses>
CLAIMED: <the numeric result claimed by the proposed solution>
REASON: <brief explanation of the arithmetic mismatch>

Do not critique ambiguous wording, style, completeness, or interpretation unless
the critique is backed by the calculator-checkable arithmetic mismatch above.
Do not use any external answer and do not give a replacement final answer.

Problem:
{question}

Proposed solution:
{draft}
"""

CALCULATOR_REVISION_PROMPT = """Revise the proposed solution using only the verified
calculator evidence below.

The calculator proves only that the checked arithmetic operation was wrong. Do
not reinterpret the problem, do not invent missing facts, and do not change the
answer for any reason other than repairing this arithmetic mismatch.

Show the arithmetic needed to solve it. End with the exact line `FINAL: <number>`.

Problem:
{question}

Draft:
{draft}

Verified evidence:
{critique}
"""

CHECK_PATTERN = re.compile(r"^\s*CHECK\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
CLAIMED_PATTERN = re.compile(
    r"^\s*CLAIMED\s*:\s*([-+]?\$?[\d,]+(?:\.\d+)?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _normalize_expression(expression: str) -> str:
    return expression.strip().strip("`$").replace("×", "*").replace("÷", "/")


def extract_arithmetic_evidence(critique: str) -> dict[str, Any]:
    """Return calculator evidence only when the critique proves an arithmetic error."""
    if not critique.strip().upper().startswith("REVISE"):
        return {"status": "no_change"}

    check_match = CHECK_PATTERN.search(critique)
    claimed_match = CLAIMED_PATTERN.search(critique)
    if not check_match or not claimed_match:
        return {"status": "missing_evidence"}

    expression = _normalize_expression(check_match.group(1))
    try:
        claimed = Decimal(claimed_match.group(1).replace("$", "").replace(",", ""))
        result = Decimal(calculate(expression))
    except (ArithmeticError, InvalidOperation, SyntaxError, ValueError) as error:
        return {
            "status": "invalid_evidence",
            "expression": expression,
            "error": str(error),
        }

    if result == claimed:
        return {
            "status": "no_mismatch",
            "expression": expression,
            "claimed": str(claimed),
            "calculator_result": str(result),
        }
    return {
        "status": "verified_mismatch",
        "expression": expression,
        "claimed": str(claimed),
        "calculator_result": str(result),
    }


class SelfRefineAgent(BaseAgent):
    def __init__(
        self,
        client: CompletionClient,
        max_rounds: int = 1,
        mode: str = "original",
    ) -> None:
        if max_rounds != 1:
            raise ValueError("Self-Refine comparison currently supports exactly one round.")
        if mode not in {"original", "calculator"}:
            raise ValueError("mode must be 'original' or 'calculator'")
        self.client = client
        self.max_rounds = max_rounds
        self.mode = mode

    def solve(self, question: str, *, question_id: str = "unknown"):
        # 初稿固定采用 CoT，Self-Refine 的比较对象是高质量初稿而非 Direct。
        draft = self.client.complete(CHAIN_OF_THOUGHT_PROMPT.format(question=question))
        method = "self_refine" if self.mode == "original" else "self_refine_calculator"
        trace = SolveTrace(
            question_id=question_id,
            method=method,
            steps=[{"round": 0, "response": draft, "feedback": "", "feedback_source": "none"}],
            final_answer=draft,
        )
        if self.mode == "calculator":
            revised, step = self._calculator_gated_round(question, draft)
        else:
            revised, step = self._original_round(question, draft)

        trace.steps.append(step)
        trace.final_answer = revised
        return revised, trace

    def _original_round(self, question: str, draft: str) -> tuple[str, dict[str, Any]]:
        # 旧版 Self-Refine：只要模型自评声称有问题，就把批评交给改写阶段。
        critique = self.client.complete(
            ORIGINAL_CRITIQUE_PROMPT.format(question=question, draft=draft)
        )
        revised = draft
        if critique.strip() == "NO_CHANGE":
            feedback_source = "self"
        else:
            revised = self.client.complete(
                ORIGINAL_REVISION_PROMPT.format(
                    question=question,
                    draft=draft,
                    critique=critique,
                )
            )
            feedback_source = "self"
        return revised, {
            "round": 1,
            "response": revised,
            "feedback": critique,
            "feedback_source": feedback_source,
            "mode": "original",
        }

    def _calculator_gated_round(self, question: str, draft: str) -> tuple[str, dict[str, Any]]:
        # 新版门控：只有计算器证明了算术不一致才允许改写。
        critique = self.client.complete(
            CALCULATOR_CRITIQUE_PROMPT.format(question=question, draft=draft)
        )
        evidence = extract_arithmetic_evidence(critique)
        if evidence["status"] != "verified_mismatch":
            return draft, {
                "round": 1,
                "response": draft,
                "feedback": critique,
                "feedback_source": "self_rejected",
                "mode": "calculator",
                "evidence": evidence,
            }

        verified_feedback = (
            f"Calculator evaluated {evidence['expression']} as "
            f"{evidence['calculator_result']}, but the draft claimed "
            f"{evidence['claimed']}.\n\nOriginal critique:\n{critique}"
        )
        revised = self.client.complete(
            CALCULATOR_REVISION_PROMPT.format(
                question=question,
                draft=draft,
                critique=verified_feedback,
            )
        )
        return revised, {
            "round": 1,
            "response": revised,
            "feedback": critique,
            "feedback_source": "calculator",
            "mode": "calculator",
            "evidence": evidence,
        }
