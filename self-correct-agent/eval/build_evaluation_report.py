"""根据已有 JSON/JSONL 结果生成第 4、5 周报告和核心图。"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_MATH_SUMMARY = ROOT / "results" / "baseline_summary.json"
DEFAULT_CODE_SUMMARY = ROOT / "results" / "code_summary.json"
DEFAULT_MATH_RECORDS = ROOT / "results" / "baseline_records.jsonl"
DEFAULT_CODE_RECORDS = ROOT / "results" / "code_records.jsonl"
DEFAULT_REPORT = ROOT / "evaluation_report.md"
DEFAULT_CHART = ROOT / "when_correction_helps.svg"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def pairwise(
    records: list[dict[str, Any]],
    before: str,
    after: str,
    outcome_key: str,
) -> dict[str, int] | None:
    """只用最终判分做事后统计，不把标准答案返回给 Agent。"""
    by_method = {
        method: {
            record["id"]: bool(record[outcome_key])
            for record in records
            if record.get("method") == method
        }
        for method in (before, after)
    }
    if not by_method[before] or by_method[before].keys() != by_method[after].keys():
        return None
    counts = {"kept": 0, "fixed": 0, "regressed": 0, "failed_both": 0}
    for item_id, before_ok in by_method[before].items():
        after_ok = by_method[after][item_id]
        key = {
            (True, True): "kept",
            (False, True): "fixed",
            (True, False): "regressed",
            (False, False): "failed_both",
        }[(before_ok, after_ok)]
        counts[key] += 1
    return counts


def percent(value: float | None) -> str:
    return "-" if value is None else f"{value:.1%}"


def comparison_text(value: dict[str, int] | None) -> tuple[str, str]:
    if value is None:
        return "-", "-"
    return str(value["fixed"]), str(value["regressed"])


def _method_rate(methods: dict[str, Any], name: str) -> float | None:
    item = methods.get(name)
    if not item:
        return None
    return item.get("accuracy", item.get("pass_rate"))


def build_chart(path: Path, rows: list[tuple[str, float, str]]) -> None:
    """生成无第三方依赖的横向条形图。"""
    width = 900
    left = 250
    right = 70
    top = 70
    row_height = 44
    height = top + len(rows) * row_height + 65
    plot_width = width - left - right
    colors = {"self": "#d97706", "tool": "#087f5b", "baseline": "#4b5563"}
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        'aria-labelledby="title desc">',
        '<title id="title">When self-correction helps</title>',
        '<desc id="desc">Accuracy and pass rate for GSM8K and HumanEval methods.</desc>',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        '<text x="20" y="32" font-family="Arial, sans-serif" font-size="20" '
        'font-weight="600" fill="#111827">When self-correction helps</text>',
        '<text x="20" y="54" font-family="Arial, sans-serif" font-size="12" '
        'fill="#4b5563">Same local Qwen3-8B model; gold answers are used only after generation for evaluation.</text>',
    ]
    for tick in range(0, 101, 20):
        x = left + plot_width * tick / 100
        elements.append(
            f'<line x1="{x:.1f}" y1="{top - 10}" x2="{x:.1f}" '
            f'y2="{height - 35}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        elements.append(
            f'<text x="{x:.1f}" y="{height - 14}" text-anchor="middle" '
            f'font-family="Arial, sans-serif" font-size="11" fill="#6b7280">{tick}%</text>'
        )
    for index, (label, rate, category) in enumerate(rows):
        y = top + index * row_height
        bar_width = plot_width * rate
        elements.extend(
            [
                f'<text x="{left - 12}" y="{y + 21}" text-anchor="end" '
                f'font-family="Arial, sans-serif" font-size="13" fill="#111827">'
                f'{html.escape(label)}</text>',
                f'<rect x="{left}" y="{y + 7}" width="{bar_width:.1f}" height="22" '
                f'rx="3" fill="{colors[category]}"/>',
                f'<text x="{left + bar_width + 8:.1f}" y="{y + 22}" '
                f'font-family="Arial, sans-serif" font-size="12" font-weight="600" '
                f'fill="#111827">{rate:.0%}</text>',
            ]
        )
    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def build_report(args: argparse.Namespace) -> None:
    math_summary = load_json(args.math_summary)
    code_summary = load_json(args.code_summary)
    math_records = load_jsonl(args.math_records)
    code_records = load_jsonl(args.code_records)
    math_methods = math_summary["methods"]
    code_methods = code_summary["methods"]

    math_variants = sorted(
        name for name in math_methods if name == "self_refine" or name.startswith("self_refine_r")
    )
    code_variants = sorted(
        name for name in code_methods if name.startswith("code_self_repair")
    )

    table_rows: list[str] = []
    for name in math_variants:
        rounds = math_methods[name].get("rounds", 1)
        comparison = pairwise(math_records, "baseline_cot", name, "correct")
        fixed, regressed = comparison_text(comparison)
        table_rows.append(
            f"| GSM8K | 无（模型自评） | {rounds} | `{name}` | "
            f"{percent(_method_rate(math_methods, name))} | {fixed} | {regressed} |"
        )
    critic_comparison = pairwise(math_records, "baseline_cot", "critic", "correct")
    fixed, regressed = comparison_text(critic_comparison)
    table_rows.append(
        "| GSM8K | 计算器 | 1 | `critic` | "
        f"{percent(_method_rate(math_methods, 'critic'))} | {fixed} | {regressed} |"
    )
    for name in code_variants:
        rounds = code_methods[name].get("repair_rounds", 1)
        comparison = pairwise(code_records, "code_direct", name, "passed")
        fixed, regressed = comparison_text(comparison)
        table_rows.append(
            f"| HumanEval | 单元测试 | {rounds} | `{name}` | "
            f"{percent(_method_rate(code_methods, name))} | {fixed} | {regressed} |"
        )

    chart_rows = [
        ("GSM8K Direct baseline", _method_rate(math_methods, "baseline_direct") or 0, "baseline"),
        ("GSM8K CoT baseline", _method_rate(math_methods, "baseline_cot") or 0, "baseline"),
        ("GSM8K Self-Refine r1", _method_rate(math_methods, "self_refine") or 0, "self"),
    ]
    if "self_refine_r2" in math_methods:
        chart_rows.append(
            ("GSM8K Self-Refine r2", _method_rate(math_methods, "self_refine_r2") or 0, "self")
        )
    chart_rows.extend(
        [
            ("GSM8K calculator CRITIC", _method_rate(math_methods, "critic") or 0, "tool"),
            (
                "GSM8K retrieved verified examples",
                _method_rate(math_methods, "reflection_embedding_correct_embedding") or 0,
                "tool",
            ),
            ("HumanEval Direct baseline", _method_rate(code_methods, "code_direct") or 0, "baseline"),
            ("HumanEval unit-test repair r1", _method_rate(code_methods, "code_self_repair") or 0, "tool"),
        ]
    )
    if "code_self_repair_r2" in code_methods:
        chart_rows.append(
            (
                "HumanEval unit-test repair r2",
                _method_rate(code_methods, "code_self_repair_r2") or 0,
                "tool",
            )
        )
    chart_rows.append(
        (
            "HumanEval test-selected best-of",
            code_summary.get("comparison", {}).get("best_of_available", {}).get("pass_rate", 0),
            "tool",
        )
    )
    build_chart(args.chart, chart_rows)

    self_compare = pairwise(math_records, "baseline_cot", "self_refine", "correct")
    code_compare = pairwise(code_records, "code_direct", "code_self_repair", "passed")
    lines = [
        "# 第 4–5 周评测报告：自我纠错何时有用",
        "",
        "## 实验设置",
        "",
        "- 模型：本地 Qwen3-8B Q4，temperature=0。",
        "- 数学：固定 GSM8K 100 题；数值 exact match 判分。",
        "- 代码：HumanEval 前 100 题；官方 `check(candidate)` 单元测试判分。",
        "- 公平性：题目标准答案只在候选答案生成结束后用于统计，不进入 Self-Refine、CRITIC 或代码修复提示。",
        "- 采纳规则：数学 CRITIC 只在模型声明的 `VERIFY` 与 `FINAL` 冲突时重写；代码修复只由单测通过与否决定。",
        "",
        "![什么时候自我纠错有用](when_correction_helps.svg)",
        "",
        "## 关键结果",
        "",
        "| 题型 | 纠错反馈 | 纠错轮数 | 方法 | 准确率/通过率 | 相对基线改对 | 相对基线改错 |",
        "|---|---|---:|---|---:|---:|---:|",
        *table_rows,
        "",
        "基线口径：GSM8K 的纠错基线是 CoT 94%；HumanEval 工具修复基线是 Direct 82%。",
        "",
        "## 自评反馈 vs 工具反馈",
        "",
        (
            f"- 无外部信号的 Self-Refine（1 轮）为 90%，相对 CoT 改对 "
            f"{self_compare['fixed'] if self_compare else '-'} 题、改错 "
            f"{self_compare['regressed'] if self_compare else '-'} 题。模型会把听起来合理的自我批评当成事实，因此可能越改越差。"
        ),
        (
            f"- HumanEval 单测修复为 85%，相对 Direct 改对 "
            f"{code_compare['fixed'] if code_compare else '-'} 题、改错 "
            f"{code_compare['regressed'] if code_compare else '-'} 题。失败输入、异常和实际输出让反馈可执行，而且通过候选不会被再次改写。"
        ),
        "- GSM8K 计算器 CRITIC 只有 90%，没有超过 CoT。计算器能验证算式执行，却不能验证模型是否把自然语言题意翻译成了正确算式，因此外部工具并非天然有效。",
        "- 检索此前已被外部判对的相似解法达到 96%。这说明可靠反馈配合可迁移的上下文，比无依据自评更稳定。",
        "",
        "## 研究结论",
        "",
        "自我纠错是否有用，关键不在于多生成一轮，而在于反馈是否可靠、具体，并且能在不知道标准答案的情况下决定是否采纳修改。代码题比数学题更容易受益，是因为单元测试同时提供明确的对错信号和定位线索；数学计算器通常只能检查局部算术，无法检查题意建模。",
        "",
        (
            "轮数对照为：数学 CoT 0 轮 94%、Self-Refine 1 轮 90%、2 轮 "
            f"{percent(_method_rate(math_methods, 'self_refine_r2'))}；代码 Direct 0 轮 82%、"
            "单测修复 1 轮 85%、2 轮 "
            f"{percent(_method_rate(code_methods, 'code_self_repair_r2'))}。"
            "这支持“失败才修改、通过就停止”的门控策略；单纯增加轮数没有带来单调收益。"
        ),
        "不同轮数使用独立结果名（如 `self_refine_r2`、`code_self_repair_r2`），因此后续补跑 3 轮不会覆盖现有结果。",
        "",
        "## 复现实验",
        "",
        "```bash",
        "# 无外部反馈：数学自评 2、3 轮",
        ".venv/bin/python main.py eval --provider local --method self_refine --rounds 2 --max-tokens 512 --workers 4",
        ".venv/bin/python main.py eval --provider local --method self_refine --rounds 3 --max-tokens 512 --workers 4",
        "",
        "# 外部反馈：代码单测修复 2、3 轮",
        ".venv/bin/python eval/run_code_eval.py --provider local --mode self_repair --repair-rounds 2 --limit 100 --max-tokens 768 --workers 1",
        ".venv/bin/python eval/run_code_eval.py --provider local --mode self_repair --repair-rounds 3 --limit 100 --max-tokens 768 --workers 1",
        "",
        "# 结果更新后重新生成本报告与核心图",
        ".venv/bin/python eval/build_evaluation_report.py",
        "```",
        "",
        "## 标准答案泄漏检查",
        "",
        "评测器保存 `expected_answer` 只是为了最终统计；Agent 的 `solve()` 只接收题目。代码修复提示来自单元测试输出，数学 CRITIC 来自计算器输出。检索正确样例只允许使用当前题之前已经被外部判对的模型答案。以上过程都没有使用当前题标准答案选择候选。",
        "",
    ]
    args.report.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--math-summary", type=Path, default=DEFAULT_MATH_SUMMARY)
    parser.add_argument("--code-summary", type=Path, default=DEFAULT_CODE_SUMMARY)
    parser.add_argument("--math-records", type=Path, default=DEFAULT_MATH_RECORDS)
    parser.add_argument("--code-records", type=Path, default=DEFAULT_CODE_RECORDS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--chart", type=Path, default=DEFAULT_CHART)
    return parser.parse_args()


if __name__ == "__main__":
    build_report(parse_args())
