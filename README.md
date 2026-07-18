# SelfCorrect Agent

本仓库研究大语言模型在数学题和代码题上的自我纠错能力，重点比较模型自评、反思记忆、计算器反馈和单元测试反馈。

## 核心结论

固定使用本地 Qwen3-8B Q4、temperature=0，并在两个 100 题测试集上评测：

| 数据集 | 方法 | 结果 |
|---|---|---:|
| GSM8K | Direct | 38 / 100 |
| GSM8K | CoT | 94 / 100 |
| GSM8K | Self-Refine（1 / 2 轮） | 90 / 92 |
| GSM8K | CRITIC（计算器反馈） | 90 / 100 |
| GSM8K | Reflection + 已验证样例检索 | **96 / 100** |
| HumanEval | Direct | 82 / 100 |
| HumanEval | Self-Repair（单测反馈，1 / 2 轮） | 85 / 85 |
| HumanEval | 单测选择 Best-of | **88 / 100** |

实验表明：没有外部对错信号时，多轮自评不保证提升；单元测试反馈更具体可靠，但增加修复轮数也不会自动带来单调收益。

## 仓库结构

```text
agent_selfcorrect/
  self-correct-agent/   # Agent、工具、评测入口与实验结果
  docs/                 # 项目说明、论文笔记与论文 PDF
  README.md
```

主要文件：

- [`self-correct-agent/README.md`](self-correct-agent/README.md)：环境配置和完整运行命令。
- [`self-correct-agent/eval/evaluation_report.md`](self-correct-agent/eval/evaluation_report.md)：第 4–5 周评测报告。
- [`self-correct-agent/eval/when_correction_helps.svg`](self-correct-agent/eval/when_correction_helps.svg)：核心实验图。
- [`self-correct-agent/eval/results/`](self-correct-agent/eval/results/)：数学和代码的逐题记录与汇总。
- [`docs/paper_notes.md`](docs/paper_notes.md)：论文阅读笔记。
- [`docs/project_brief.md`](docs/project_brief.md)：项目任务说明。
- [`docs/echo_repro_to_selfcorrect.md`](docs/echo_repro_to_selfcorrect.md)：研究方向说明。

## 快速开始

```bash
cd self-correct-agent
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
```

启动本地 `llama-server` 后运行：

```bash
.venv/bin/python main.py eval --provider local --method baseline --mode cot --workers 1
.venv/bin/python eval/run_code_eval.py --provider local --mode self_repair --limit 100 --workers 1
```

`.env`、虚拟环境、模型文件和请求缓存不会提交到 Git。
