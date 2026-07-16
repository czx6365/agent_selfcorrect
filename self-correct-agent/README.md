# self-correct-agent

SelfCorrect 的第 1、2 周实验实现。

## 目录

```text
self-correct-agent/
  main.py
  agents/
    base.py
    self_refine.py
  eval/
    download_data.py
    build_dataset.py
    llm_client.py
    metrics.py
    run_eval.py
    datasets/
      gsm8k/
      humaneval/
    results/
  logs/
  docs/
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

每次运行只会更新两份结果文件：

- `eval/results/baseline_records.jsonl`：三种方法的逐题预测、原始回答与 trace。
- `eval/results/baseline_summary.json`：可复现实验配置与方法对比统计。
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
.venv/bin/python main.py eval --provider local --method self_refine --rounds 1 --max-tokens 512 --workers 4
```

Self-Refine 先复用 CoT 初稿，再做不含参考答案的自评和改写。`baseline_summary.json` 的 `self_refine` 字段会报告 CoT 到 Self-Refine 的改对/改错，并单列 58 个 CoT 修复题被保留或改错的数量。

## 第 3 周：Reflection

```bash
.venv/bin/python main.py eval --provider local --method reflection --max-tokens 512 --workers 1
```

Reflection 只在外部评测判错后写入一条不含标准答案的具体教训，下一题带入最近三条教训。由于记忆写入顺序属于实验条件，必须使用 `--workers 1`。日志写入 `logs/reflection_log.jsonl`。

### 已完成结果

固定 GSM8K 100 题、Qwen3-8B Q4、本地 llama-server、`temperature=0`、`max_tokens=512`：

| 方法 | 正确数 | 准确率 |
|---|---:|---:|
| Direct | 38 / 100 | 38.0% |
| CoT | 94 / 100 | 94.0% |
| Self-Refine（1 轮） | 90 / 100 | 90.0% |

CoT 比 Direct 改对 58 题、改错 2 题。Self-Refine 没有修复任何 CoT 错题，反而将 4 个 CoT 正确答案改错；在 CoT 修复的 58 题中，保住 56 题、改坏 2 题。

正式 GSM8K baseline 显式关闭 DeepSeek 的隐藏思考（`ANTHROPIC_THINKING=disabled`），因此温度 `0` 能生效，且“直接作答 vs 可见逐步推理”只比较提示方式。OpenAI-compatible 请求使用种子 `42`；请求会以内容哈希缓存到 `eval/cache/`，同一配置重复运行不会重复扣费。
