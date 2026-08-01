"""Project CLI entry point for single solves and evaluations."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from agents.base import BaselineAgent
from agents.critic import CriticAgent
from agents.reflection import ReflectionAgent
from agents.self_refine import SelfRefineAgent
from eval import run_eval
from eval.llm_client import (
    AnthropicCompatibleClient,
    LLMConfigurationError,
    LocalLlamaServerClient,
    OpenAICompatibleClient,
)
from eval.metrics import extract_gsm8k_answer


ROOT = Path(__file__).resolve().parent
DEFAULT_SOLVE_LOG = ROOT / "logs" / "solve_trace.jsonl"


def build_client(args: argparse.Namespace):
    """Create the configured LLM client used by both solve and eval commands."""
    if args.provider == "anthropic":
        return AnthropicCompatibleClient(
            model=args.model,
            base_url=args.base_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            thinking=args.thinking,
        )
    if args.provider == "local":
        return LocalLlamaServerClient(
            model=args.model,
            base_url=args.base_url,
            temperature=args.temperature,
            seed=args.seed,
            max_tokens=args.max_tokens,
        )
    return OpenAICompatibleClient(
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        seed=args.seed,
        max_tokens=args.max_tokens,
    )


def build_agent(args: argparse.Namespace, client):
    """Instantiate one math-solving agent with the same method names as eval."""
    if args.method == "baseline":
        return BaselineAgent(client, mode=args.mode)
    if args.method == "self_refine":
        return SelfRefineAgent(
            client,
            max_rounds=args.rounds,
            mode=args.self_refine_mode,
        )
    if args.method == "critic":
        return CriticAgent(client)
    return ReflectionAgent(
        client,
        memory_limit=args.memory_top_k,
        retrieval=args.memory_retrieval,
        correct_examples=args.correct_examples,
        correct_top_k=args.correct_top_k,
        embedding_model=args.embedding_model,
    )


def solve_main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="main.py solve",
        description="Solve one GSM8K-style math question and save its trace.",
    )
    parser.add_argument("question", help="Question text to solve.")
    parser.add_argument(
        "--method",
        choices=("baseline", "self_refine", "reflection", "critic"),
        default="baseline",
    )
    parser.add_argument("--mode", choices=("direct", "cot"), default="cot")
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Self-Refine comparison supports only 1 round.",
    )
    parser.add_argument(
        "--self-refine-mode",
        choices=("original", "calculator"),
        default="original",
        help="Original Self-Refine r1 or calculator-gated Self-Refine.",
    )
    parser.add_argument(
        "--provider",
        choices=("openai", "anthropic", "local"),
        default="local",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--thinking", choices=("enabled", "disabled"), default="disabled")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--question-id", default="manual")
    parser.add_argument("--log", type=Path, default=DEFAULT_SOLVE_LOG)
    parser.add_argument("--json", action="store_true", help="Print the full JSON record.")
    parser.add_argument(
        "--memory-retrieval",
        choices=("recent", "embedding"),
        default="recent",
    )
    parser.add_argument("--memory-top-k", type=int, default=3)
    parser.add_argument(
        "--correct-examples",
        choices=("none", "embedding"),
        default="none",
    )
    parser.add_argument("--correct-top-k", type=int, default=3)
    parser.add_argument("--embedding-model", default="local-hash")
    args = parser.parse_args(argv)

    client = build_client(args)
    agent = build_agent(args, client)
    raw_response, trace = agent.solve(args.question, question_id=args.question_id)
    prediction = extract_gsm8k_answer(raw_response)
    trace.final_answer = prediction or ""
    method = trace.method
    if args.method == "reflection":
        method = f"reflection_{args.memory_retrieval}"
        if args.correct_examples != "none":
            method = f"{method}_correct_{args.correct_examples}"

    record = {
        "id": args.question_id,
        "method": method,
        "question": args.question,
        "prediction": prediction,
        "raw_response": raw_response,
        "trace": asdict(trace),
        "provider": args.provider,
        "llm_model": getattr(client, "model", None),
        "temperature": args.temperature,
        "llm_seed": args.seed,
        "max_tokens": getattr(client, "max_tokens", None),
        "created_at": datetime.now(UTC).isoformat(),
    }
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    if args.json:
        print(json.dumps(record, ensure_ascii=False, indent=2))
        return
    print(prediction if prediction is not None else raw_response.strip())


def main() -> None:
    """Dispatch project commands while preserving the old eval shorthand."""
    if len(sys.argv) > 1 and sys.argv[1] == "solve":
        try:
            solve_main(sys.argv[2:])
        except (LLMConfigurationError, ValueError) as error:
            raise SystemExit(f"Solve was not started: {error}") from error
        return

    # Preserve both `python main.py eval ...` and the earlier direct eval form.
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        del sys.argv[1]
    run_eval.main()


if __name__ == "__main__":
    main()
