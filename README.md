# SelfCorrect Agent

一个研究大语言模型自我纠错能力的实验项目。

本项目关注的问题不是“让模型多想几遍会不会更好”，而是更具体地回答：

> 自我纠错在什么条件下真的有用？什么时候反而会把正确答案改错？

项目已经实现数学题和代码题两条实验线，比较 Direct、CoT、Self-Refine、Reflection、CRITIC、单元测试修复和已验证样例检索等方法，并保存了逐题记录、汇总结果和评测报告。

## 当前状态

已完成第 1-5 周核心实验：

- 固定数据集：GSM8K 100 题、HumanEval 前 100 题。
- 固定模型设置：本地 Qwen3-8B Q4，`temperature=0`，`seed=42`。
- 实现统一 Agent 接口和多种纠错方法。
- 实现数学 exact match 判分和 HumanEval 单元测试判分。
- 实现安全计算器、代码执行反馈、反思记忆和正确样例检索。
- 生成完整结果文件、失败案例分析、评测报告和核心图。
- 整理论文阅读笔记和本地 PDF。

## 核心结论

实验结果支持一个比较明确的判断：

**自我纠错是否有用，关键不在于多生成一轮，而在于反馈是否可靠、具体，并且能在不知道标准答案的情况下决定是否采纳修改。**

主要发现：

- 无外部反馈的 Self-Refine 不稳定。GSM8K 上 CoT 是 94%，Self-Refine 1 轮降到 90%，说明模型会把原本正确的答案改错。
- 代码题更容易从纠错中受益。HumanEval 上单元测试反馈把 Direct 从 82% 提升到 85%，且没有把已通过样例改坏。
- 数学计算器反馈有局限。计算器只能验证算术表达式，不能判断模型是否正确理解题意，因此 CRITIC 没有超过 CoT。
- 已验证正确样例检索最有效。Reflection + Sentence-Transformers 检索历史正确样例达到 96%，是当前 GSM8K 最好结果。
- 增加轮数不保证单调收益。数学 Self-Refine 2 轮仍低于 CoT；代码 Self-Repair 2 轮没有超过 1 轮。

## 实验结果

### GSM8K 数学题

固定 100 题，数值 exact match 判分。

| 方法 | 正确数 | 准确率 |
|---|---:|---:|
| Direct | 38 / 100 | 38.0% |
| CoT | 94 / 100 | 94.0% |
| Self-Refine 1 轮 | 90 / 100 | 90.0% |
| Self-Refine 2 轮 | 92 / 100 | 92.0% |
| Reflection 旧版 recent | 94 / 100 | 94.0% |
| CRITIC 计算器反馈 | 90 / 100 | 90.0% |
| Reflection + local-hash embedding | 89 / 100 | 89.0% |
| Reflection + 已验证正确样例检索 | **96 / 100** | **96.0%** |

关键对比：

- CoT 相对 Direct：改对 58 题，改错 2 题。
- Self-Refine 1 轮相对 CoT：改对 0 题，改错 4 题。
- Self-Refine 2 轮相对 CoT：改对 2 题，改错 4 题。
- 已验证正确样例检索相对 CoT：改对 2 题，改错 0 题。

### HumanEval 代码题

固定前 100 题，使用官方 `check(candidate)` 单元测试判分。

| 方法 | 通过数 | 通过率 |
|---|---:|---:|
| Direct | 82 / 100 | 82.0% |
| CoT | 85 / 100 | 85.0% |
| Self-Repair 1 轮 | 85 / 100 | 85.0% |
| Self-Repair 2 轮 | 85 / 100 | 85.0% |
| CoT-Repair 1 轮 | 85 / 100 | 85.0% |
| Best-of 已跑方法 | **88 / 100** | **88.0%** |

关键对比：

- CoT 相对 Direct：修复 4 题，退化 1 题。
- Self-Repair 相对 Direct：修复 3 题，退化 0 题。
- Best-of 说明不同方法能解出的题并不完全重合，单测可作为候选选择信号。

## 方法说明

项目实现了以下 Agent / 评测方法：

| 方法 | 反馈来源 | 说明 |
|---|---|---|
| Direct | 无 | 直接输出最终答案或代码 |
| CoT | 无 | 显式逐步推理后输出答案 |
| Self-Refine | 模型自评 | 初稿 -> 自我批评 -> 改写 |
| Reflection | 历史失败教训 | 外部判错后写 lesson，后续题目检索使用 |
| CRITIC | 计算器 | 模型给出 `VERIFY` 表达式，由安全计算器检查 |
| Self-Repair | 单元测试 | 代码失败后用 traceback、断言诊断修复 |
| Correct Example Retrieval | 已验证正确样例 | 只检索当前题之前已经被外部判分确认正确的样例 |

实验中特别注意避免标准答案泄漏：

- Agent 的 `solve()` 只接收题目，不接收当前题标准答案。
- 标准答案只在候选生成结束后用于最终统计。
- Self-Refine 的 critique prompt 不包含参考答案。
- CRITIC 只使用模型自己声明的 `VERIFY` 表达式和计算器结果。
- 代码修复只使用单元测试输出。
- 正确样例检索只使用当前题之前已经判对的模型答案，不提前读取未来题目。

## 仓库结构

```text
agent_selfcorrect/
  README.md
  docs/
    project_brief.md                 # 项目任务说明
    paper_notes.md                   # 论文阅读笔记
    echo_repro_to_selfcorrect.md     # 从 ECHO-Repro 到 SelfCorrect 的研究衔接
    papers/                          # 本地论文 PDF
  self-correct-agent/
    main.py                          # GSM8K CLI 入口
    agents/                          # baseline / self_refine / reflection / critic
    tools/                           # calculator / code_runner
    eval/
      dataset.jsonl                  # GSM8K 100 题子集
      run_eval.py                    # 数学评测
      run_code_eval.py               # HumanEval 代码评测
      build_evaluation_report.py     # 生成报告和图
      evaluation_report.md           # 第 4-5 周实验报告
      when_correction_helps.svg      # 核心实验图
      results/                       # records 与 summary
    logs/                            # solve trace 与 reflection log
    failure_review.md                # GSM8K baseline 失败案例
    requirements.txt
```

## 快速开始

```bash
cd self-correct-agent
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
```

启动本地 `llama-server`。默认评测使用 OpenAI-compatible 的本地接口：

```bash
llama-server -m "$QWEN_GGUF_PATH" \
  --host "$LOCAL_LLM_HOST" \
  --port "$LOCAL_LLM_PORT" \
  --reasoning off
```

运行 GSM8K 数学评测：

```bash
.venv/bin/python main.py eval --provider local --method baseline --mode direct --max-tokens 512 --reset-results
.venv/bin/python main.py eval --provider local --method baseline --mode cot --max-tokens 512
.venv/bin/python main.py eval --provider local --method self_refine --rounds 1 --max-tokens 512
.venv/bin/python main.py eval --provider local --method self_refine --rounds 2 --max-tokens 512
.venv/bin/python main.py eval --provider local --method critic --max-tokens 512
```

运行 HumanEval 代码评测：

```bash
.venv/bin/python eval/run_code_eval.py --provider local --mode direct --limit 100 --max-tokens 768 --workers 1 --reset-results
.venv/bin/python eval/run_code_eval.py --provider local --mode cot --limit 100 --max-tokens 768 --workers 1
.venv/bin/python eval/run_code_eval.py --provider local --mode self_repair --limit 100 --max-tokens 768 --repair-rounds 1 --workers 1
.venv/bin/python eval/run_code_eval.py --provider local --mode self_repair --limit 100 --max-tokens 768 --repair-rounds 2 --workers 1
```

重新生成报告和核心图：

```bash
.venv/bin/python eval/build_evaluation_report.py
```

## 主要产物

- `self-correct-agent/eval/results/baseline_records.jsonl`：GSM8K 逐题记录。
- `self-correct-agent/eval/results/baseline_summary.json`：GSM8K 汇总结果和 pairwise comparison。
- `self-correct-agent/eval/results/code_records.jsonl`：HumanEval 逐题记录。
- `self-correct-agent/eval/results/code_summary.json`：HumanEval 汇总结果和 Best-of 统计。
- `self-correct-agent/eval/evaluation_report.md`：第 4-5 周核心评测报告。
- `self-correct-agent/eval/when_correction_helps.svg`：核心实验图。
- `self-correct-agent/failure_review.md`：CoT baseline 失败样例。
- `docs/paper_notes.md`：论文阅读笔记。

## 论文与背景

项目参考并对照了以下方向：

- ReAct：推理和工具行动交替。
- Self-Refine：模型自评和迭代改写。
- Reflexion：失败后写自然语言反思记忆。
- CRITIC：工具交互式验证和纠错。
- Large Language Models Cannot Self-Correct Reasoning Yet：无外部反馈时自纠错不稳定的反方证据。
- ExpeL、Self-Consistency、Chain-of-Verification：经验学习、多路径推理和验证链。

详细笔记见 `docs/paper_notes.md`。

## 下一步可扩展方向

- 补跑 3 轮或更多轮数，画出更完整的收益递减曲线。
- 加入等成本 self-consistency baseline，比较“多采样投票”和“多轮自我修改”。
- 对 GSM8K 错误类型做人工标注，区分建模错误、算术错误、抽取错误。
- 把 HumanEval 的单元测试反馈扩展到真实仓库 Issue / CI 修复任务。
- 将 Reflection 记忆升级为可投票、可编辑、可去噪的经验库。
