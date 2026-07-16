from __future__ import annotations

from .base import BaseAgent, CHAIN_OF_THOUGHT_PROMPT, CompletionClient, SolveTrace


CRITIQUE_PROMPT = """Review the proposed solution to the math problem below.
Check every interpretation and arithmetic operation. State one concrete error if one exists.
If the solution is sound, respond with exactly `NO_CHANGE`.
Do not use any external answer and do not give a replacement final answer.

Problem:
{question}

Proposed solution:
{draft}
"""

REVISION_PROMPT = """Solve the math problem again using the draft and critique below.
Keep the draft unchanged when the critique says `NO_CHANGE`; otherwise repair only the identified issue.
Show the arithmetic needed to solve it. End with the exact line `FINAL: <number>`.

Problem:
{question}

Draft:
{draft}

Critique:
{critique}
"""


class SelfRefineAgent(BaseAgent):
    def __init__(self, client: CompletionClient, max_rounds: int = 1) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds must be at least 1")
        self.client = client
        self.max_rounds = max_rounds

    def solve(self, question: str, *, question_id: str = "unknown"):
        # 初稿固定采用 CoT，Self-Refine 的比较对象是高质量初稿而非 Direct。
        draft = self.client.complete(CHAIN_OF_THOUGHT_PROMPT.format(question=question))
        trace = SolveTrace(
            question_id=question_id,
            method="self_refine",
            steps=[{"round": 0, "response": draft, "feedback": "", "feedback_source": "none"}],
            final_answer=draft,
        )
        for round_number in range(1, self.max_rounds + 1):
            # 批评提示中不含标准答案，防止用 gold answer 伪造自我纠错能力。
            critique = self.client.complete(CRITIQUE_PROMPT.format(question=question, draft=draft))
            if critique.strip() == "NO_CHANGE":
                # 未定位具体问题时保持初稿，避免“为了修改而修改”。
                revised = draft
            else:
                # 只将模型自己提出的具体批评交给改写阶段。
                revised = self.client.complete(
                    REVISION_PROMPT.format(question=question, draft=draft, critique=critique)
            )
            # 每一轮都保留初稿、批评和改写，供事后分析回退原因。
            trace.steps.append(
                {
                    "round": round_number,
                    "response": revised,
                    "feedback": critique,
                    "feedback_source": "self",
                }
            )
            draft = revised
        trace.final_answer = draft
        return draft, trace
