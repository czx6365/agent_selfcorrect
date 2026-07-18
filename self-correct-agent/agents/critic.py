"""Tool-assisted critic agent for arithmetic consistency checks."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any

from tools.calculator import calculate

from .base import BaseAgent, CompletionClient, SolveTrace


SOLVE_PROMPT = """Solve the grade-school math problem carefully.

Show the arithmetic. End with these two exact lines:
VERIFY: <one arithmetic expression using only numbers, +, -, *, /, and parentheses>
FINAL: <number>

VERIFY must evaluate to the same numeric value as FINAL. Do not put an equals
sign, units, variables, or explanation on the VERIFY line.

Problem:
{question}
"""

CRITIQUE_PROMPT = """A calculator checked an arithmetic expression from a proposed solution.

Calculator expression: {expression}
Calculator result: {tool_result}
Proposed FINAL value: {candidate_answer}

State one concise, concrete inconsistency between the calculator result and the
proposed final value. Do not use a reference answer or solve a different
problem. Do not output a replacement final answer.

Problem:
{question}

Proposed solution:
{draft}
"""

REVISION_PROMPT = """Revise the proposed solution using the calculator feedback below.
The calculator proves only that VERIFY and FINAL disagree; its numeric result
is not a reference answer. Re-derive the intended arithmetic from the problem,
then correct either VERIFY, FINAL, or both as needed. Do not invent missing
facts and do not use a reference answer.

End with exactly these two lines:
VERIFY: <one arithmetic expression>
FINAL: <number>

Problem:
{question}

Proposed solution:
{draft}

Calculator feedback:
{critique}
"""

FINAL_PATTERN = re.compile(
    r"^\s*(?:\*\*)?FINAL\s*:\s*([-+]?\$?[\d,]+(?:\.\d+)?)",
    re.IGNORECASE | re.MULTILINE,
)


def extract_verify_expression(response: str) -> str | None:
    """提取模型声明的可验证算式；不猜测正文中的任意公式。"""
    for line in response.splitlines():
        # 接受模型常见的 Markdown 包装，如 **VERIFY:** 或 ### VERIFY:。
        normalized = line.strip().lstrip("#>- ").replace("**", "").strip()
        match = re.fullmatch(r"VERIFY\s*:\s*(.+?)\s*", normalized, re.IGNORECASE)
        if not match:
            continue
        expression = match.group(1).strip().strip("`$")
        # 兼容常见的乘除符号，但不接受等号或自然语言。
        expression = expression.replace("×", "*").replace("÷", "/")
        return expression if len(expression) <= 200 else None
    return None


def extract_final_number(response: str) -> Decimal | None:
    """读取 FINAL 数字，仅用于与工具结果做数值一致性比较。"""
    match = FINAL_PATTERN.search(response)
    if not match:
        return None
    try:
        return Decimal(match.group(1).replace("$", "").replace(",", ""))
    except InvalidOperation:
        return None


def run_calculator(expression: str | None) -> dict[str, Any]:
    """把工具失败转成结构化反馈，保证单题评测不会被格式错误中断。"""
    if not expression:
        return {"status": "no_expression", "expression": None, "result": None}
    try:
        result = calculate(expression)
    except (ArithmeticError, SyntaxError, ValueError, InvalidOperation) as error:
        return {
            "status": "tool_error",
            "expression": expression,
            "result": None,
            "error": str(error),
        }
    return {"status": "ok", "expression": expression, "result": result}


class CriticAgent(BaseAgent):
    """用计算器确认算术不一致后才调用语言模型改写答案。"""

    def __init__(self, client: CompletionClient) -> None:
        self.client = client

    def solve(self, question: str, *, question_id: str = "unknown") -> tuple[str, SolveTrace]:
        # 初稿主动提供 VERIFY，使工具输入来自模型而非评测标准答案。
        draft = self.client.complete(SOLVE_PROMPT.format(question=question))
        expression = extract_verify_expression(draft)
        tool = run_calculator(expression)
        candidate_answer = extract_final_number(draft)

        tool_value = (
            Decimal(tool["result"])
            if tool["status"] == "ok" and tool["result"] is not None
            else None
        )
        matches_tool = (
            tool_value == candidate_answer
            if tool_value is not None and candidate_answer is not None
            else None
        )

        final = draft
        critique = "NO_CHANGE"
        if matches_tool is False:
            # 只有可证明的工具矛盾才触发改写，避免模型无依据地重写正确答案。
            critique = self.client.complete(
                CRITIQUE_PROMPT.format(
                    question=question,
                    draft=draft,
                    expression=tool["expression"],
                    tool_result=tool["result"],
                    candidate_answer=candidate_answer,
                )
            ).strip()
            if critique == "NO_CHANGE":
                critique = (
                    f"The calculator evaluated {tool['expression']} as "
                    f"{tool['result']}, which does not match FINAL: {candidate_answer}."
                )
            final = self.client.complete(
                REVISION_PROMPT.format(
                    question=question,
                    draft=draft,
                    critique=critique,
                )
            )

        trace = SolveTrace(
            question_id=question_id,
            method="critic",
            steps=[
                {
                    "round": 0,
                    "response": draft,
                    "feedback": "",
                    "feedback_source": "none",
                },
                {
                    "round": 1,
                    "response": final,
                    "feedback": critique,
                    "feedback_source": "calculator",
                    "tool": tool,
                    "final_matches_tool": matches_tool,
                },
            ],
            final_answer=final,
        )
        return final, trace
