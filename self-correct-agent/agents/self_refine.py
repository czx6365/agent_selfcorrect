from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any

from tools.calculator import calculate

from .base import BaseAgent, CHAIN_OF_THOUGHT_PROMPT, CompletionClient, SolveTrace


# 原始 Self-Refine 批评提示词：
# 允许模型检查题意理解、计算过程等任意问题。
ORIGINAL_CRITIQUE_PROMPT = """Review the proposed solution to the math problem below.
Check every interpretation and arithmetic operation. State one concrete error if one exists.
If the solution is sound, respond with exactly `NO_CHANGE`.
Do not use any external answer and do not give a replacement final answer.

Problem:
{question}

Proposed solution:
{draft}
"""

# 原始 Self-Refine 修改提示词：
# 当模型认为初稿存在问题时，根据自我批评重新生成答案。
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

# 计算器门控模式的批评提示词：
# 模型只能指出能够通过计算器验证的算术错误，不能随意修改题意理解。
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

# 计算器门控模式的修改提示词：
# 只允许根据已经验证的算术错误修改答案，禁止重新解释题意。
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


# 从模型批评中提取 CHECK 字段中的算术表达式。
# 例如：CHECK: (10 + 5) * 2
CHECK_PATTERN = re.compile(
    r"^\s*CHECK\s*:\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# 从模型批评中提取 CLAIMED 字段中的结果。
# 支持正负数、美元符号、千位分隔符和小数。
CLAIMED_PATTERN = re.compile(
    r"^\s*CLAIMED\s*:\s*([-+]?\$?[\d,]+(?:\.\d+)?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _normalize_expression(expression: str) -> str:
    """规范化模型输出的算术表达式，使其能够交给计算器执行。"""
    return (
        expression
        .strip()
        .strip("`$")
        .replace("×", "*")
        .replace("÷", "/")
    )


def extract_arithmetic_evidence(critique: str) -> dict[str, Any]:
    """
    从模型批评中提取并验证算术证据。

    只有当以下条件全部满足时，才返回 verified_mismatch：
    1. 批评以 REVISE 开头；
    2. 同时包含 CHECK 和 CLAIMED 字段；
    3. CHECK 表达式能够被计算器正确执行；
    4. 计算器结果与模型提取出的 CLAIMED 结果不一致。
    """

    # 模型没有要求修改，直接视为无需变更。
    if not critique.strip().upper().startswith("REVISE"):
        return {"status": "no_change"}

    # 提取待验证的算术表达式和初稿声称的结果。
    check_match = CHECK_PATTERN.search(critique)
    claimed_match = CLAIMED_PATTERN.search(critique)

    # 模型虽然输出了 REVISE，但没有按指定格式提供完整证据。
    if not check_match or not claimed_match:
        return {"status": "missing_evidence"}

    expression = _normalize_expression(check_match.group(1))

    try:
        # Decimal 避免使用 float 时出现不必要的精度误差。
        claimed = Decimal(
            claimed_match.group(1)
            .replace("$", "")
            .replace(",", "")
        )

        # 使用外部计算器工具验证算术表达式。
        result = Decimal(calculate(expression))

    except (ArithmeticError, InvalidOperation, SyntaxError, ValueError) as error:
        # 表达式非法、计算失败或结果无法转换为 Decimal。
        return {
            "status": "invalid_evidence",
            "expression": expression,
            "error": str(error),
        }

    # 计算器结果与初稿声称结果一致，说明模型批评并未证明错误。
    if result == claimed:
        return {
            "status": "no_mismatch",
            "expression": expression,
            "claimed": str(claimed),
            "calculator_result": str(result),
        }

    # 只有计算器结果和初稿结果确实不一致，才允许进入修改阶段。
    return {
        "status": "verified_mismatch",
        "expression": expression,
        "claimed": str(claimed),
        "calculator_result": str(result),
    }


class SelfRefineAgent(BaseAgent):
    """支持原始自我反思和计算器门控反思的 Self-Refine Agent。"""

    def __init__(
        self,
        client: CompletionClient,
        max_rounds: int = 1,
        mode: str = "original",
    ) -> None:
        """
        初始化 Self-Refine Agent。

        Args:
            client: 大语言模型调用客户端。
            max_rounds: 自我修改轮数，目前实验固定为一轮。
            mode:
                original：传统 Self-Refine，直接相信模型自我批评；
                calculator：只有计算器验证存在算术错误时才允许修改。
        """
        if max_rounds != 1:
            raise ValueError(
                "Self-Refine comparison currently supports exactly one round."
            )

        if mode not in {"original", "calculator", "decision_gate"}:
            raise ValueError("mode must be 'original', 'calculator', or 'decision_gate'")

        self.client = client
        self.max_rounds = max_rounds
        self.mode = mode

    def solve(
        self,
        question: str,
        *,
        question_id: str = "unknown",
    ) -> tuple[str, SolveTrace]:
        """
        求解数学问题，并执行一轮 Self-Refine。

        流程：
        1. 使用 CoT 提示词生成高质量初稿；
        2. 根据 mode 进入原始自评或计算器门控自评；
        3. 保存完整执行轨迹；
        4. 返回最终答案和轨迹。
        """

        # 初稿固定使用 Chain-of-Thought。
        # 实验比较的是“高质量初稿经过不同反馈机制后的变化”，
        # 而不是比较 Direct Prompt 和 CoT Prompt。
        draft = self.client.complete(
            CHAIN_OF_THOUGHT_PROMPT.format(question=question)
        )

        # 根据模式设置实验方法名称，便于后续统计和对比。
        method = {
            "original": "self_refine",
            "calculator": "self_refine_calculator",
            "decision_gate": "self_refine_gate",
        }[self.mode]

        # 初始化执行轨迹，round=0 表示初稿阶段。
        trace = SolveTrace(
            question_id=question_id,
            method=method,
            steps=[
                {
                    "round": 0,
                    "response": draft,
                    "feedback": "",
                    "feedback_source": "none",
                }
            ],
            final_answer=draft,
        )

        # 根据配置选择不同的 Self-Refine 策略。
        if self.mode == "calculator":
            revised, step = self._calculator_gated_round(question, draft)
        elif self.mode == "decision_gate":
            revised, step = self._decision_gate_round(question, draft)
        else:
            revised, step = self._original_round(question, draft)

        # 保存反思阶段轨迹，并更新最终答案。
        trace.steps.append(step)
        trace.final_answer = revised

        return revised, trace

    def _original_round(
        self,
        question: str,
        draft: str,
    ) -> tuple[str, dict[str, Any]]:
        """
        执行传统 Self-Refine。

        模型先对初稿进行自我批评：
        - 输出 NO_CHANGE：保持初稿；
        - 输出其他批评：直接将批评交给模型执行修改。

        该模式没有外部验证，因此可能出现错误批评导致答案退化。
        """

        # 让模型检查初稿中的题意理解和计算过程。
        critique = self.client.complete(
            ORIGINAL_CRITIQUE_PROMPT.format(
                question=question,
                draft=draft,
            )
        )

        # 默认保持原始答案不变。
        revised = draft

        if critique.strip() == "NO_CHANGE":
            # 模型认为初稿没有问题，不调用修改模型。
            feedback_source = "self"
        else:
            # 旧版策略直接相信模型的自我批评并执行修改。
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

    def _calculator_gated_round(
        self,
        question: str,
        draft: str,
    ) -> tuple[str, dict[str, Any]]:
        """
        执行计算器门控的 Self-Refine。

        模型可以提出算术错误，但必须提供：
        - CHECK：具体算术表达式；
        - CLAIMED：初稿声称的结果。

        系统使用计算器验证后，只有确定存在数值不一致时才允许修改。
        """

        # 让模型寻找能够通过计算器验证的具体算术错误。
        critique = self.client.complete(
            CALCULATOR_CRITIQUE_PROMPT.format(
                question=question,
                draft=draft,
            )
        )

        # 使用确定性的计算器验证模型提供的证据。
        evidence = extract_arithmetic_evidence(critique)

        # 没有发现经过验证的算术错误时，拒绝模型的修改请求。
        if evidence["status"] != "verified_mismatch":
            return draft, {
                "round": 1,
                "response": draft,
                "feedback": critique,
                "feedback_source": "self_rejected",
                "mode": "calculator",
                "evidence": evidence,
            }

        # 将计算器验证结果转化为可信反馈。
        # 修改模型只能根据这条经过验证的证据修复答案。
        verified_feedback = (
            f"Calculator evaluated {evidence['expression']} as "
            f"{evidence['calculator_result']}, but the draft claimed "
            f"{evidence['claimed']}.\n\n"
            f"Original critique:\n{critique}"
        )

        # 使用验证后的算术证据生成修订答案。
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

    def _decision_gate_round(
        self,
        question: str,
        draft: str,
    ) -> tuple[str, dict[str, Any]]:
        """
        执行带采纳门控的 Self-Refine。

        模型仍然先自我批评并生成候选修订，但最终是否采纳由
        decision gate 根据证据强度、答案变化和置信度风险决定。
        """

        # 局部导入避免 decision_gate 复用本文件证据提取函数时形成循环导入。
        from dataclasses import asdict

        from .decision_gate import decide_revision

        critique = self.client.complete(
            ORIGINAL_CRITIQUE_PROMPT.format(
                question=question,
                draft=draft,
            )
        )

        if critique.strip() == "NO_CHANGE":
            return draft, {
                "round": 1,
                "response": draft,
                "feedback": critique,
                "feedback_source": "decision_gate_rejected",
                "mode": "decision_gate",
                "gate": {
                    "decision": "reject",
                    "score": -99,
                    "reasons": ["critique requested no change"],
                    "features": {"critique": "no_change"},
                },
            }

        candidate_revision = self.client.complete(
            ORIGINAL_REVISION_PROMPT.format(
                question=question,
                draft=draft,
                critique=critique,
            )
        )

        gate = decide_revision(
            question=question,
            draft=draft,
            critique=critique,
            revised=candidate_revision,
            client=self.client,
        )
        final = candidate_revision if gate.decision == "accept" else draft
        feedback_source = (
            "decision_gate_accepted"
            if gate.decision == "accept"
            else "decision_gate_rejected"
        )

        return final, {
            "round": 1,
            "response": final,
            "candidate_revision": candidate_revision,
            "feedback": critique,
            "feedback_source": feedback_source,
            "mode": "decision_gate",
            "gate": asdict(gate),
        }
