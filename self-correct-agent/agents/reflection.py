
"""Reflection agent: external verdicts become reusable lessons and retry context."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .base import BaseAgent, CompletionClient, SolveTrace


SOLVE_WITH_MEMORY_PROMPT = """Solve the grade-school math problem carefully.

Useful lessons from earlier failed attempts:
{lessons}

Show the arithmetic and end with exactly `FINAL: <number>`.

Problem:
{question}
"""

RETRY_WITHOUT_REFLECTION_PROMPT = """The previous answer to this grade-school math problem was marked incorrect
by an external evaluator. Solve it again independently and check the arithmetic.

Show the arithmetic and end with exactly `FINAL: <number>`.

Problem:
{question}
"""

SPECIFIC_REFLECTION_PROMPT = """An attempted solution to a grade-school math problem was marked incorrect
by an external evaluator.

Return exactly one short, general, actionable verification rule for similar
future problems. The rule must name a check such as a relationship, base value,
or arithmetic operation, but must not mention this particular problem.

Output rules:
- Output one plain sentence only, with no title, list, Markdown, or explanation.
- Do not repeat or quote the problem or attempted solution.
- Do not include names, numbers, quantities, or a final answer.
- Do not calculate, state, or guess the reference answer.
- Do not claim the problem is incomplete or ambiguous.

Problem:
{question}

Incorrect attempt:
{attempt}
"""

VAGUE_REFLECTION_PROMPT = """An attempted solution to a grade-school math problem was marked incorrect
by an external evaluator.

Return exactly one short, generic verification rule with no diagnostic detail.
It must be broadly applicable to any arithmetic word problem.

Output rules:
- Output one plain sentence only, with no title, list, Markdown, or explanation.
- Do not mention the problem, attempted solution, names, numbers, quantities, or answers.

Problem:
{question}

Incorrect attempt:
{attempt}
"""


@dataclass
class Reflection:
    """一条由外部判错触发、可被后续题目读取的失败教训。"""

    question_id: str
    lesson: str


class ReflectionAgent(BaseAgent):
    """保存最近失败教训，并支持带教训的同题重试。"""

    def __init__(
        self,
        client: CompletionClient,
        *,
        memory_limit: int = 3,
        lesson_mode: str = "specific",
    ) -> None:
        if lesson_mode not in {"specific", "vague"}:
            raise ValueError("lesson_mode must be 'specific' or 'vague'")
        self.client = client
        self.memory_limit = memory_limit
        self.lesson_mode = lesson_mode
        self.memory: list[Reflection] = []

    def _recent_lessons(self) -> str:
        # 第一个版本按时间取最近记忆；后续可替换为关键词或向量检索。
        recent = self.memory[-self.memory_limit:]
        if not recent:
            return "No earlier lessons are available."
        return "\n".join(f"- {item.lesson}" for item in recent)

    def restore_memory(self, reflections: list[Reflection]) -> None:
        """断点续跑时恢复已完成题目产生的教训，保持执行顺序。"""
        self.memory = list(reflections)

    def solve(
        self,
        question: str,
        *,
        question_id: str = "unknown",
        attempt_number: int = 1,
        retry_without_reflection: bool = False,
    ) -> tuple[str, SolveTrace]:
        """解题一次；无反思对照组重试时只收到“判错”信号。"""
        lessons = self._recent_lessons()
        prompt = (
            RETRY_WITHOUT_REFLECTION_PROMPT.format(question=question)
            if retry_without_reflection
            else SOLVE_WITH_MEMORY_PROMPT.format(question=question, lessons=lessons)
        )
        response = self.client.complete(prompt)
        trace = SolveTrace(
            question_id=question_id,
            method="reflection",
            steps=[
                {
                    "attempt": attempt_number,
                    "response": response,
                    "feedback": "",
                    "feedback_source": "none",
                    "lessons_used": lessons,
                }
            ],
            final_answer=response,
        )
        return response, trace

    @staticmethod
    def _lesson_is_clean(lesson: str, question: str, attempt: str) -> bool:
        """拒绝多句、格式化、数字和题目专有词，防止记忆污染。"""
        words = re.findall(r"[A-Za-z]+", lesson)
        if not 6 <= len(words) <= 24:
            return False
        if re.search(r"\d|[*#`]|\n", lesson):
            return False
        if len(re.findall(r"[.!?]", lesson)) != 1:
            return False
        # 句首以外的大写词通常是题目中的人名或缩写。
        if re.search(r"(?<!^)\b[A-Z][a-z]+\b", lesson):
            return False
        source_words = set(re.findall(r"[a-z]{4,}", f"{question} {attempt}".lower()))
        lesson_words = set(word.lower() for word in words)
        # 与题干/草稿重合的内容词说明模型在复述具体情境，而不是写可迁移规则。
        allowed = {
            "before", "finalizing", "verify", "check", "each", "stated",
            "relationship", "relationships", "recompute", "arithmetic", "from",
            "defined", "base", "carefully", "review", "reasoning", "and", "the",
        }
        return not ((lesson_words - allowed) & source_words)

    def _fallback_lesson(self) -> str:
        if self.lesson_mode == "vague":
            return "Before finalizing, carefully review the reasoning and arithmetic."
        return "Before finalizing, verify each stated relationship and recompute the arithmetic from its defined base."

    def reflect(
        self,
        question: str,
        attempt: str,
        *,
        question_id: str,
    ) -> Reflection:
        """仅在外部评测判错后生成 lesson，且不会得到 gold answer。"""
        template = (
            SPECIFIC_REFLECTION_PROMPT
            if self.lesson_mode == "specific"
            else VAGUE_REFLECTION_PROMPT
        )
        generated = " ".join(
            self.client.complete(template.format(question=question, attempt=attempt)).split()
        )
        lesson = (
            generated
            if self._lesson_is_clean(generated, question, attempt)
            else self._fallback_lesson()
        )
        reflection = Reflection(question_id=question_id, lesson=lesson)
        self.memory.append(reflection)
        return reflection
