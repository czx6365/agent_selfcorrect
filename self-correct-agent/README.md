# self-correct-agent

SelfCorrect 的第 1–5 周实验实现。

## 目录

```text
self-correct-agent/
  main.py
  agents/
    base.py
    self_refine.py
    reflection.py
    critic.py
  tools/
    calculator.py
    code_runner.py
  data/
    dataset.jsonl
    datasets/
      gsm8k/
      humaneval/
  eval/
    download_data.py
    build_dataset.py
    llm_client.py
    metrics.py
    run_eval.py
    run_code_eval.py
    build_evaluation_report.py
    export_logs.py
  results/
    baseline_records.jsonl
    baseline_summary.json
    code_records.jsonl
    code_summary.json
    evaluation_report.md
    when_correction_helps.svg
  logs/
  ../docs/
  requirements.txt
```

## 数据集

- GSM8K: 小学数学应用题，天然适合做数值判分。
- HumanEval: 编程题，有单元测试，可以直接做外部反馈。

## 第 1 周：GSM8K baseline

本周固定使用 GSM8K 测试集的 100 题随机子集（`seed=42`），避免后续方法在不同题目上比较。

```bash
# 首次创建虚拟环境并生成数据集
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python eval/build_dataset.py

# 启动本地 llama.cpp 服务（默认 provider）
llama-server -m "$QWEN_GGUF_PATH" --host "$LOCAL_LLM_HOST" --port "$LOCAL_LLM_PORT" --reasoning off

# 配置云端 OpenAI-compatible 模型；不会把密钥写入文件
export OPENAI_API_KEY="..."
export OPENAI_MODEL="..."
# 可选：DeepSeek 等兼容服务
export OPENAI_BASE_URL="https://api.openai.com/v1"

# 或者使用 Anthropic-compatible 服务（本仓库会自动加载 .env）
# ANTHROPIC_BASE_URL=...
# ANTHROPIC_AUTH_TOKEN=...
# ANTHROPIC_MODEL=...

# 两个单轮 baseline：直接作答、逐步推理
.venv/bin/python main.py eval --provider local --method baseline --mode direct
.venv/bin/python main.py eval --provider local --method baseline --mode cot

# Anthropic-compatible endpoint
.venv/bin/python main.py eval --provider anthropic --method baseline --mode cot
```

### 单题求解入口

```bash
.venv/bin/python main.py solve "小明有3个苹果，又买了2袋每袋4个，一共有几个苹果？" --method baseline --mode cot
.venv/bin/python main.py solve "小明有3个苹果，又买了2袋每袋4个，一共有几个苹果？" --method self_refine --rounds 1 --json
.venv/bin/python main.py solve "小明有3个苹果，又买了2袋每袋4个，一共有几个苹果？" --method critic
```

`solve` 会打印抽取出的最终数字，并把完整 trace 追加写入 `logs/solve_trace.jsonl`。`--json` 可直接查看本次求解的完整记录。

每次运行只会更新这些结果文件：

- `results/baseline_records.jsonl`：三种方法的逐题预测、原始回答与 trace。
- `results/baseline_summary.json`：可复现实验配置与方法对比统计。
- `failure_review.md`：最多五个实际失败案例；明确标注单轮结果无法区分“能力不够”和“缺少检查”。

### 本地评测

切换模型时，第一次运行加上 `--reset-results`，防止不同模型的记录混在一起：

```bash
.venv/bin/python main.py eval --provider local --method baseline --mode direct --max-tokens 512 --reset-results
.venv/bin/python main.py eval --provider local --method baseline --mode cot --max-tokens 512
```

若评测被中断，使用同一配置继续，不要再次重置：

```bash
.venv/bin/python main.py eval --provider local --method baseline --mode direct --max-tokens 512 --resume --workers 4
```

固定 GSM8K 测试子集为 100 题，采样种子为 `42`。

## 第 2 周：Self-Refine

```bash
.venv/bin/python main.py eval --provider local --method self_refine --self-refine-mode original --rounds 1 --max-tokens 512 --workers 4
.venv/bin/python main.py eval --provider local --method self_refine --self-refine-mode calculator --rounds 1 --max-tokens 512 --workers 4
```

Self-Refine 先复用 CoT 初稿，再做不含参考答案的自评。当前保留两种一轮模式：`--self-refine-mode original` 是旧版 r1，只要模型自评指出错误就改写，结果写为 `self_refine`；`--self-refine-mode calculator` 只有在自评给出 calculator 可验证的算术证据，且计算器证明该算式与声称结果不一致时才允许改写，结果写为 `self_refine_calculator`。`baseline_summary.json` 的 `self_refine` 字段会报告 CoT 到各 Self-Refine 模式的改对/改错。

## 第 3 周：Reflection

```bash
.venv/bin/python main.py eval --provider local --method reflection --max-tokens 512 --workers 1
```

Reflection 只在外部评测判错后写入一条不含标准答案的具体教训，下一题带入最近三条教训。由于记忆写入顺序属于实验条件，必须使用 `--workers 1`。日志写入 `logs/reflection_log.jsonl`。

### Memory Retrieval

Reflection 可按时间读取最近 lesson，或用 embedding 从已完成错题中检索语义相近的 lesson。embedding 索引只包含当前题之前已经判错的记录；`--resume` 会从已完成记录重建索引，不会读取未来题目。

```bash
# 时间检索对照组
.venv/bin/python main.py eval --provider local --method reflection --memory-retrieval recent --memory-top-k 3 --max-tokens 512 --workers 1

# embedding 检索实验组；默认 local-hash，不需要联网下载 HuggingFace 模型
.venv/bin/python main.py eval --provider local --method reflection --memory-retrieval embedding --memory-top-k 3 --max-tokens 512 --workers 1

# 可选：先安装 sentence-transformers；如果本地已缓存模型，再指定更强的语义模型
.venv/bin/python -m pip install sentence-transformers
.venv/bin/python main.py eval --provider local --method reflection --memory-retrieval embedding --embedding-model sentence-transformers/all-MiniLM-L6-v2 --memory-top-k 3 --max-tokens 512 --workers 1
```

两组结果会以 `reflection_recent` 与 `reflection_embedding` 写入同一份评测记录；汇总会比较两者相对 CoT 的改对/改错，以及 embedding 相对 recent 的变化。

### Correct Example Retrieval

还可以让 Reflection 维护一个“已判对样例库”：每道题解完后，只有外部判分为正确，才把这道题和模型自己的正确解法写入样例库；后续题用 embedding 检索 top-3 相似正确样例作为 few-shot 依据。这个实验不能提前从当前 100 题里挑正确题，否则会泄漏未来评测信息。

```bash
# 离线流程测试：用 local-hash 检索已判对样例
.venv/bin/python main.py eval --provider local --method reflection --memory-retrieval embedding --correct-examples embedding --correct-top-k 3 --memory-top-k 3 --max-tokens 512 --workers 1

# 真正 Sentence-Transformers embedding：需要先安装并能下载/缓存模型
.venv/bin/python -m pip install sentence-transformers
.venv/bin/python main.py eval --provider local --method reflection --memory-retrieval embedding --correct-examples embedding --correct-top-k 3 --embedding-model sentence-transformers/all-MiniLM-L6-v2 --memory-top-k 3 --max-tokens 512 --workers 1
```

这组结果会写成 `reflection_embedding_correct_embedding`，可直接和 `baseline_cot`、`reflection_embedding` 比较。

### 已完成结果

固定 GSM8K 100 题、Qwen3-8B Q4、本地 llama-server、`temperature=0`、`max_tokens=512`：

| 方法 | 正确数 | 准确率 |
|---|---:|---:|
| Direct | 38 / 100 | 38.0% |
| CoT | 94 / 100 | 94.0% |
| 旧版 Self-Refine（1 轮） | 90 / 100 | 90.0% |
| Reflection（旧版 recent） | 94 / 100 | 94.0% |
| Critic（计算器 VERIFY） | 90 / 100 | 90.0% |
| Reflection + local-hash embedding | 89 / 100 | 89.0% |
| Reflection + Sentence-Transformers top-3 正确样例 | 96 / 100 | 96.0% |

CoT 比 Direct 改对 58 题、改错 2 题。Self-Refine 没有修复任何 CoT 错题，反而将 4 个 CoT 正确答案改错；在 CoT 修复的 58 题中，保住 56 题、改坏 2 题。

Sentence-Transformers 正确样例检索使用 `sentence-transformers/all-MiniLM-L6-v2`，每题只检索此前已经被外部判分确认正确的 top-3 相似样例，不提前读取未来题。它相对 CoT 保住 94 题、修复 2 题、退化 0 题，是当前 GSM8K 最强结果。

## HumanEval 代码题

代码题评测入口：

```bash
.venv/bin/python eval/run_code_eval.py --provider local --mode direct --limit 100 --max-tokens 768 --workers 1 --reset-results
.venv/bin/python eval/run_code_eval.py --provider local --mode cot --limit 100 --max-tokens 768 --workers 1
.venv/bin/python eval/run_code_eval.py --provider local --mode self_repair --limit 100 --max-tokens 768 --repair-rounds 1 --workers 1
.venv/bin/python eval/run_code_eval.py --provider local --mode cot_repair --limit 100 --max-tokens 768 --repair-rounds 1 --workers 1
```

HumanEval 前 100 题、Qwen3-8B Q4、本地 llama-server、`temperature=0`、`max_tokens=768`：

| 方法 | 通过数 | 通过率 |
|---|---:|---:|
| Direct | 82 / 100 | 82.0% |
| CoT | 85 / 100 | 85.0% |
| Self-Repair（Direct 初稿 + 单测反馈修 1 轮） | 85 / 100 | 85.0% |
| Self-Repair（Direct 初稿 + 单测反馈修 2 轮） | 85 / 100 | 85.0% |
| CoT-Repair（CoT 初稿 + 单测反馈修 1 轮） | 85 / 100 | 85.0% |
| Best-of 已跑方法（单测选择任一通过候选） | 88 / 100 | 88.0% |

CoT 相对 Direct 修复 4 题、退化 1 题；Self-Repair 相对 Direct 修复 3 题、退化 0 题。CoT-Repair 目前没有超过 CoT，说明“已有错误 CoT 思路 + 一轮 traceback 修复”不一定能跳出原错误算法。`results/code_summary.json` 会保存逐方法统计和 pairwise comparison。

## 第 4–5 周：工具反馈与关键实验

CRITIC 的确定性工具集中在 `tools/`：数学题用安全 AST 计算器，代码题在临时子进程中运行 HumanEval 官方单测并返回 traceback 与断言诊断。完整的“自评反馈 vs 工具反馈”对照、标准答案泄漏检查和核心图见 [`results/evaluation_report.md`](results/evaluation_report.md)。

数学 Self-Refine 现在保留旧版 r1 和 calculator 门控两种一轮模式；代码修复的不同轮数仍保存为独立方法名（如 `code_self_repair_r2`）。跑完新实验后执行：

```bash
.venv/bin/python eval/build_evaluation_report.py
```

即可从两份 summary/records 重新生成报告与 `results/when_correction_helps.svg`。

正式 GSM8K baseline 显式关闭 DeepSeek 的隐藏思考（`ANTHROPIC_THINKING=disabled`），因此温度 `0` 能生效，且“直接作答 vs 可见逐步推理”只比较提示方式。OpenAI-compatible 请求使用种子 `42`；请求会以内容哈希缓存到 `.cache/llm/`，同一配置重复运行不会重复扣费。

如果某次实验中断，先恢复本地 LLM 服务，再用相同参数加 `--resume` 续跑。例如：

```bash
.venv/bin/python main.py eval --provider local --method self_refine --self-refine-mode calculator --rounds 1 --max-tokens 512 --workers 4 --resume
```

若代码实验结果里出现运行级 `error`，通常是 LLM 端点不可用；恢复服务后重新运行同一方法即可覆盖该方法的旧记录。

## 日志与最终材料

评测的完整逐题轨迹保存在 `results/*_records.jsonl`。交付前可统一导出到最终日志：

```bash
.venv/bin/python eval/export_logs.py
```

补充材料：

- `../docs/paper_notes.md`：论文阅读笔记。
- `../docs/research_proposal.md`：1 页科研提案。
- `../docs/demo_script.md`：3 分钟演示稿。
