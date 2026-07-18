
"""Reflection agent: external verdicts become reusable lessons and retry context."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from .base import BaseAgent, CompletionClient, SolveTrace
from .reflection_memory import (
    HashingTextEncoder,
    SentenceTransformerEncoder,
    TextEncoder,
    cosine_similarity,
)


SOLVE_WITH_MEMORY_PROMPT = """Solve the grade-school math problem carefully.

Useful lessons from earlier failed attempts:
{lessons}

Similar earlier problems that were externally verified correct:
{correct_examples}

Use the earlier problems only as solution-pattern references. Do not copy their
numbers unless the current problem gives the same numbers.

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
    question: str = ""


@dataclass
class CorrectExample:
    """一条此前已被外部判分确认正确的解题样例。"""

    question_id: str
    question: str
    solution: str
    answer: str


class ReflectionAgent(BaseAgent):
    """保存最近失败教训，并支持带教训的同题重试。"""

    def __init__(
        self,
        client: CompletionClient,
        *,
        memory_limit: int = 3,
        lesson_mode: str = "specific",
        retrieval: Literal["recent", "embedding"] = "recent",
        correct_examples: Literal["none", "embedding"] = "none",
        correct_top_k: int = 3,
        embedding_model: str = "local-hash",
        encoder: TextEncoder | None = None,
    ) -> None:
        if lesson_mode not in {"specific", "vague"}:
            raise ValueError("lesson_mode must be 'specific' or 'vague'")
        self.client = client
        self.memory_limit = memory_limit
        self.lesson_mode = lesson_mode
        if retrieval not in {"recent", "embedding"}:
            raise ValueError("retrieval must be 'recent' or 'embedding'")
        if correct_examples not in {"none", "embedding"}:
            raise ValueError("correct_examples must be 'none' or 'embedding'")
        self.retrieval = retrieval
        self.correct_examples = correct_examples
        self.correct_top_k = correct_top_k
        self.embedding_model = embedding_model
        self._encoder = encoder
        if (
            (retrieval == "embedding" or correct_examples == "embedding")
            and encoder is None
        ):
            self._encoder = (
                HashingTextEncoder()
                if embedding_model == "local-hash"
                else SentenceTransformerEncoder(embedding_model)
            )
        self.memory: list[Reflection] = []
        self._memory_vectors: list[list[float]] = []
        self.correct_memory: list[CorrectExample] = []
        self._correct_vectors: list[list[float]] = []

    def _index_reflection(self, reflection: Reflection) -> None:
        """为 lesson 建索引；只会在外部评测已经判错之后调用。"""
        if self.retrieval != "embedding":
            return
        if self._encoder is None:
            raise RuntimeError("Embedding encoder was not initialized.")
        document = f"Problem: {reflection.question}\nLesson: {reflection.lesson}"
        self._memory_vectors.append(self._encoder.encode(document))

    def _retrieve_lessons(self, question: str) -> str:
        """从历史错题中取 lesson；绝不访问尚未评测的题目。"""
        if not self.memory:
            return "No earlier lessons are available."
        if self.retrieval == "recent":
            selected = self.memory[-self.memory_limit:]
        else:
            if self._encoder is None:
                raise RuntimeError("Embedding encoder was not initialized.")
            query_vector = self._encoder.encode(question)
            ranked_indices = sorted(
                range(len(self.memory)),
                key=lambda index: (cosine_similarity(query_vector, self._memory_vectors[index]), index),
                reverse=True,
            )
            selected = [self.memory[index] for index in ranked_indices[: self.memory_limit]]
        return "\n".join(f"- {item.lesson}" for item in selected)

    def _index_correct_example(self, example: CorrectExample) -> None:
        """为已判对样例建索引；只索引当前题之前已经完成的样例。"""
        if self.correct_examples == "none":
            return
        if self._encoder is None:
            raise RuntimeError("Embedding encoder was not initialized.")
        self._correct_vectors.append(self._encoder.encode(example.question))

    @staticmethod
    def _compact_solution(solution: str, answer: str, max_chars: int = 900) -> str:
        """压缩正确样例，避免 top-k few-shot 把 prompt 塞得太满。"""
        text = "\n".join(line.rstrip() for line in solution.strip().splitlines())
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + " ..."
        return f"{text}\nVerified final answer: {answer}"

    def _retrieve_correct_examples(self, question: str) -> str:
        """检索最相似的历史正确样例，作为 few-shot 依据。"""
        if self.correct_examples == "none":
            return "No earlier correct examples are available."
        if not self.correct_memory:
            return "No earlier correct examples are available."
        if self._encoder is None:
            raise RuntimeError("Embedding encoder was not initialized.")
        query_vector = self._encoder.encode(question)
        ranked_indices = sorted(
            range(len(self.correct_memory)),
            key=lambda index: (
                cosine_similarity(query_vector, self._correct_vectors[index]),
                index,
            ),
            reverse=True,
        )
        selected = [
            self.correct_memory[index]
            for index in ranked_indices[: self.correct_top_k]
        ]
        blocks = []
        for position, example in enumerate(selected, start=1):
            blocks.append(
                "\n".join(
                    [
                        f"Example {position}:",
                        f"Problem: {example.question}",
                        "Verified solution:",
                        self._compact_solution(example.solution, example.answer),
                    ]
                )
            )
        return "\n\n".join(blocks)

    def restore_memory(self, reflections: list[Reflection]) -> None:
        """断点续跑时恢复已完成题目产生的教训，保持执行顺序。"""
        self.memory = []
        self._memory_vectors = []
        for reflection in reflections:
            self.memory.append(reflection)
            self._index_reflection(reflection)

    def restore_correct_examples(self, examples: list[CorrectExample]) -> None:
        """断点续跑时恢复已经判对的历史样例，保持执行顺序。"""
        self.correct_memory = []
        self._correct_vectors = []
        for example in examples:
            self.correct_memory.append(example)
            self._index_correct_example(example)

    def solve(
        self,
        question: str,
        *,
        question_id: str = "unknown",
        attempt_number: int = 1,
        retry_without_reflection: bool = False,
    ) -> tuple[str, SolveTrace]:
        """解题一次；无反思对照组重试时只收到“判错”信号。"""
        lessons = self._retrieve_lessons(question)
        correct_examples = self._retrieve_correct_examples(question)
        prompt = (
            RETRY_WITHOUT_REFLECTION_PROMPT.format(question=question)
            if retry_without_reflection
            else SOLVE_WITH_MEMORY_PROMPT.format(
                question=question,
                lessons=lessons,
                correct_examples=correct_examples,
            )
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
                    "correct_examples_used": correct_examples,
                }
            ],
            final_answer=response,
        )
        return response, trace

    @staticmethod
    def _normalize_lesson(text: str) -> str:
        """提取模型输出中的第一条可执行规则，去掉常见格式噪声。"""
        lesson = " ".join(text.split())
        lesson = re.sub(r"^(lesson|rule|reflection)\s*:\s*", "", lesson, flags=re.I)
        lesson = lesson.strip(" -*#`")
        match = re.search(r"^(.+?[.!?])(?:\s|$)", lesson)
        if match:
            lesson = match.group(1)
        if lesson and lesson[-1] not in ".!?":
            lesson += "."
        return lesson

    @staticmethod
    def _lesson_is_clean(lesson: str) -> bool:
        """拒绝多句、格式化、数字和明显题目专有词，防止记忆污染。"""
        words = re.findall(r"[A-Za-z]+", lesson)
        if not 6 <= len(words) <= 24:
            return False
        if re.search(r"\d|[*#`]|\n", lesson):
            return False
        if len(re.findall(r"[.!?]", lesson)) != 1:
            return False
        if re.search(r"\b(vs|etc)\.$", lesson, flags=re.I):
            return False
        # 句首以外的大写词通常是题目中的人名或缩写。
        if re.search(r"(?<!^)\b[A-Z][a-z]+\b", lesson):
            return False
        return True

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
        generated = self._normalize_lesson(
            self.client.complete(template.format(question=question, attempt=attempt))
        )
        lesson = generated if self._lesson_is_clean(generated) else self._fallback_lesson()
        reflection = Reflection(question_id=question_id, lesson=lesson, question=question)
        self.memory.append(reflection)
        self._index_reflection(reflection)
        return reflection

    def remember_correct(
        self,
        question: str,
        solution: str,
        answer: str,
        *,
        question_id: str,
    ) -> CorrectExample:
        """仅在外部评测判对后，把模型自己的正确解法加入样例库。"""
        example = CorrectExample(
            question_id=question_id,
            question=question,
            solution=solution,
            answer=answer,
        )
        self.correct_memory.append(example)
        self._index_correct_example(example)
        return example
