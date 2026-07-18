"""在隔离的临时进程中运行 HumanEval 单元测试。"""

from __future__ import annotations

import ast
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def _prompt_prefix(prompt: str) -> str:
    """保留函数定义前的 import，避免候选代码缺少题目依赖。"""
    lines: list[str] = []
    for line in prompt.splitlines():
        if line.startswith("def "):
            break
        lines.append(line)
    return "\n".join(lines).strip()


def _simple_candidate_asserts(test: str) -> list[dict[str, str]]:
    """提取可安全重放的 ``candidate(args) == expected`` 断言。"""
    try:
        tree = ast.parse(test)
    except SyntaxError:
        return []
    assertions: list[dict[str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assert):
            continue
        compare = node.test
        if (
            isinstance(compare, ast.Compare)
            and len(compare.ops) == 1
            and isinstance(compare.ops[0], ast.Eq)
            and len(compare.comparators) == 1
            and isinstance(compare.left, ast.Call)
            and isinstance(compare.left.func, ast.Name)
            and compare.left.func.id == "candidate"
        ):
            assertions.append(
                {
                    "call": ast.unparse(compare.left),
                    "expected": ast.unparse(compare.comparators[0]),
                }
            )
    return assertions


def _collect_assert_diagnostics(
    *,
    prompt: str,
    code: str,
    test: str,
    entry_point: str,
    timeout_seconds: float,
) -> str:
    """重放简单断言，为修复模型提供实际值与期望值。"""
    assertions = _simple_candidate_asserts(test)
    if not assertions:
        return ""
    probe_blocks: list[str] = []
    for index, assertion in enumerate(assertions[:8]):
        label = f"assert_{index}"
        probe_blocks.append(
            "\n".join(
                [
                    "try:",
                    f"    got = {assertion['call']}",
                    f"    expected = {assertion['expected']}",
                    "    ok = got == expected",
                    "except Exception as error:",
                    "    got = type(error).__name__ + ': ' + str(error)",
                    f"    expected = {assertion['expected']}",
                    "    ok = False",
                    (
                        f"print({label!r} + ': got=' + repr(got) "
                        "+ ' expected=' + repr(expected) + ' ok=' + repr(ok))"
                    ),
                ]
            )
        )
    program = "\n".join(
        [
            "from __future__ import annotations",
            _prompt_prefix(prompt),
            code,
            f"candidate = {entry_point}",
            "\n".join(probe_blocks),
            "",
        ]
    )
    with tempfile.TemporaryDirectory(prefix="humaneval_diag_") as temp_dir:
        script = Path(temp_dir) / "diagnose.py"
        script.write_text(program, encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, str(script)],
                cwd=temp_dir,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ""
    return completed.stdout.strip()


def run_unit_tests(
    *,
    prompt: str,
    code: str,
    test: str,
    entry_point: str,
    timeout_seconds: float = 3.0,
) -> dict[str, Any]:
    """运行官方 ``check(candidate)``，返回可供 Agent 使用的工具反馈。"""
    program = "\n".join(
        [
            "from __future__ import annotations",
            _prompt_prefix(prompt),
            code,
            test,
            f"check({entry_point})",
            "",
        ]
    )
    with tempfile.TemporaryDirectory(prefix="humaneval_") as temp_dir:
        script = Path(temp_dir) / "candidate.py"
        script.write_text(program, encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, str(script)],
                cwd=temp_dir,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            return {
                "passed": False,
                "status": "timeout",
                "feedback": f"Timed out after {timeout_seconds} seconds.",
                "stdout": error.stdout or "",
                "stderr": error.stderr or "",
            }

    feedback = "\n".join(
        part for part in (completed.stdout, completed.stderr) if part
    ).strip()
    if completed.returncode != 0:
        diagnostics = _collect_assert_diagnostics(
            prompt=prompt,
            code=code,
            test=test,
            entry_point=entry_point,
            timeout_seconds=timeout_seconds,
        )
        if diagnostics:
            feedback = f"{feedback}\n\nAssertion diagnostics:\n{diagnostics}".strip()
    return {
        "passed": completed.returncode == 0,
        "status": "passed" if completed.returncode == 0 else "failed",
        "feedback": feedback[-4000:],
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
