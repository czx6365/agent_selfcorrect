from .base import BaseAgent, SolveTrace


class ReflexionAgent(BaseAgent):
    def solve(self, question: str):
        trace = SolveTrace(
            question_id="unknown",
            method="reflexion",
            steps=[],
            final_answer="",
        )
        raise NotImplementedError("Implement reflexion memory here.")

