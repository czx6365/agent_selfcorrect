from dataclasses import dataclass
from typing import Any, Dict, Protocol, Tuple


@dataclass
class SolveTrace:
    question_id: str
    method: str
    steps: list[Dict[str, Any]]
    final_answer: str
    final_correct: bool | None = None


class BaseAgent:
    def solve(self, question: str, *, question_id: str = "unknown") -> Tuple[str, SolveTrace]:
        raise NotImplementedError


class CompletionClient(Protocol):
    def complete(self, prompt: str) -> str:
        """Return one deterministic completion for a prompt."""


DIRECT_PROMPT = """Solve the following grade-school math word problem.
Return only the final numeric answer. Do not include explanation or units.

Problem:
{question}
"""

CHAIN_OF_THOUGHT_PROMPT = """Solve the following grade-school math word problem carefully.
Show the arithmetic needed to solve it. End with the exact line `FINAL: <number>`.
Do not round unless the problem explicitly asks you to.

Problem:
{question}
"""


class BaselineAgent(BaseAgent):
    """A single-pass GSM8K solver with either direct or step-by-step prompting."""

    def __init__(self, client: CompletionClient, mode: str = "cot") -> None:
        if mode not in {"direct", "cot"}:
            raise ValueError("mode must be 'direct' or 'cot'")
        self.client = client
        self.mode = mode

    def solve(self, question: str, *, question_id: str = "unknown") -> Tuple[str, SolveTrace]:
        prompt = (
            DIRECT_PROMPT.format(question=question)
            if self.mode == "direct"
            else CHAIN_OF_THOUGHT_PROMPT.format(question=question)
        )
        response = self.client.complete(prompt)
        trace = SolveTrace(
            question_id=question_id,
            method=f"baseline_{self.mode}",
            steps=[
                {
                    "round": 1,
                    "prompt_mode": self.mode,
                    "response": response,
                    "feedback": "",
                    "feedback_source": "none",
                }
            ],
            final_answer=response,
        )
        return response, trace
