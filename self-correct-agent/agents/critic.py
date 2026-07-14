from .base import BaseAgent, SolveTrace


class CriticAgent(BaseAgent):
    def solve(self, question: str):
        trace = SolveTrace(
            question_id="unknown",
            method="critic",
            steps=[],
            final_answer="",
        )
        raise NotImplementedError("Implement tool-interactive critique here.")

