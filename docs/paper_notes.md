# SelfCorrect 论文阅读笔记

本文件对应项目中的自纠错解题智能体方向，重点关注一个问题：自我纠错什么时候真的有用，什么时候会把答案改坏。

## 已下载论文

PDF 已下载到 `docs/papers/`，抽取文本在 `docs/paper_text/`。

| 类型 | 论文 | arXiv | 本地 PDF |
|---|---|---|---|
| 必读 | ReAct: Synergizing Reasoning and Acting in Language Models | https://arxiv.org/abs/2210.03629 | `docs/papers/ReAct_推理行动协同_2022.pdf` |
| 必读 | Self-Refine: Iterative Refinement with Self-Feedback | https://arxiv.org/abs/2303.17651 | `docs/papers/SelfRefine_自反馈迭代优化_2023.pdf` |
| 必读 | Reflexion: Language Agents with Verbal Reinforcement Learning | https://arxiv.org/abs/2303.11366 | `docs/papers/Reflexion_语言反思强化_2023.pdf` |
| 关键 | CRITIC: LLMs Can Self-Correct with Tool-Interactive Critiquing | https://arxiv.org/abs/2305.11738 | `docs/papers/CRITIC_工具交互纠错_2024.pdf` |
| 关键 | Large Language Models Cannot Self-Correct Reasoning Yet | https://arxiv.org/abs/2310.01798 | `docs/papers/无反馈无法自纠错_2024.pdf` |
| 选读 | ExpeL: LLM Agents Are Experiential Learners | https://arxiv.org/abs/2308.10144 | `docs/papers/ExpeL_经验学习智能体_2024.pdf` |
| 选读 | Self-Consistency Improves Chain of Thought Reasoning | https://arxiv.org/abs/2203.11171 | `docs/papers/自一致性思维链_2022.pdf` |
| 选读 | Chain-of-Verification Reduces Hallucination | https://arxiv.org/abs/2309.11495 | `docs/papers/验证链减少幻觉_2023.pdf` |

## 1. ReAct: Synergizing Reasoning and Acting in Language Models

### 1. 它解决什么问题？

ReAct 解决的是推理和行动割裂的问题。普通 CoT 只在模型内部推理，容易幻觉和错误传播；只行动的 Agent 又缺少高层计划和状态跟踪。ReAct 把推理、动作、环境观察交替组织起来，让模型一边想、一边查、一边根据观察更新计划。

### 2. 它的自我纠错 / 反馈机制是什么？

核心轨迹是：

```text
Thought -> Action -> Observation -> Thought -> ...
```

模型通过外部环境返回的 Observation 修正下一步 Thought 和 Action。它不是典型的“写一段反思再改答案”，而是把反馈嵌入解题过程本身。对 SelfCorrect 项目来说，ReAct 是 `critic` 或工具型 Agent 的基础结构。

### 3. 它如何评估？有没有用到标准答案？

论文在 HotpotQA、FEVER、ALFWorld、WebShop 等任务上评测。QA 和事实验证任务中，模型可以查询 Wikipedia API；交互式任务中，环境返回状态反馈。最终评测仍用任务标准答案或成功率，但推理过程中主要依赖工具或环境观察，而不是直接拿标准答案指导修改。

### 4. 它有什么局限？

ReAct 的效果依赖工具接口、动作空间和观察质量。如果工具返回的信息不充分，或者动作格式容易出错，Agent 仍会失败。它还会消耗更多上下文和调用次数。

### 5. 对我的实现有什么启发？

`critic` 版本可以采用 ReAct 式 trace：

```json
{"thought": "...", "action": "calculator(...)", "observation": "..."}
```

这比只记录最终答案更利于分析失败原因。数学题可以把计算器结果作为 Observation；代码题可以把单元测试报错作为 Observation。实验上应区分“纯语言反思”和“有环境观察的修正”。

## 2. Self-Refine: Iterative Refinement with Self-Feedback

### 1. 它解决什么问题？

Self-Refine 研究的是：LLM 第一次输出不一定最好，能不能让同一个模型先给自己反馈，再根据反馈改写，从而在测试时提升质量。

### 2. 它的自我纠错 / 反馈机制是什么？

算法分三步：

```text
INIT: 生成初始答案
FEEDBACK: 对初始答案生成具体、可执行的反馈
REFINE: 根据反馈改写答案
```

这个循环可以重复多轮，直到达到停止条件或最大轮数。它强调反馈要 actionable 和 specific，也就是指出具体问题并给出可操作修改方向。

### 3. 它如何评估？有没有用到标准答案？

论文在 7 类任务上评估，包括对话、代码优化、数学推理、情感改写、摘要等，用人工偏好或自动指标比较初始输出和迭代输出。部分任务有明确指标，部分任务依赖人工评价。需要注意，某些任务的停止条件或评价方式可能不等同于真实部署时可用的反馈。

### 4. 它有什么局限？

最大风险是反馈本身会错。论文的错误分析也指出，失败常来自反馈定位错误、建议方向错误或反馈过于泛泛。对于推理题，模型自己未必能判断答案是否正确，可能出现“越反思越错”。

### 5. 对我的实现有什么启发？

项目中的 `self_refine.py` 应该严格记录每轮：

```json
{"round": 1, "answer": "...", "feedback": "...", "feedback_source": "self"}
```

评测时不能只看最终准确率，还要统计：

- wrong -> correct：自我纠错真正救回的题
- correct -> wrong：反思把对题改错的题
- wrong -> wrong：无效反思
- correct -> correct：保持正确

这能直接回答 Self-Refine 在当前任务域里是否真的有用。

## 3. Reflexion: Language Agents with Verbal Reinforcement Learning

### 1. 它解决什么问题？

Reflexion 解决的是 Agent 如何从失败经历中学习，而不是只在一次解题内部修改。传统 RL 需要大量交互和参数更新，Reflexion 则把失败反馈转成自然语言记忆，下次尝试时放进上下文。

### 2. 它的自我纠错 / 反馈机制是什么？

流程大致是：

```text
执行任务 -> 得到反馈信号 -> 写反思 -> 存入 episodic memory -> 下次尝试带上反思
```

反馈可以来自环境奖励、启发式规则、模型自评或工具反馈。反思文本相当于“语义梯度”，不更新模型参数，只更新提示上下文。

### 3. 它如何评估？有没有用到标准答案？

论文在 ALFWorld、HotpotQA、HumanEval 等任务上评估，报告了多轮尝试后的成功率提升。HumanEval 中使用测试结果作为反馈，HotpotQA 中也涉及是否答对的信号。这里要特别注意：如果用标准答案或 oracle correctness 决定是否继续反思，在科研评测中必须单独标注，不能伪装成无外部反馈的自我纠错。

### 4. 它有什么局限？

Reflexion 的效果依赖反馈信号质量和反思质量。错误反思会污染记忆；记忆太多会挤占上下文；跨题迁移时，某条经验可能只适用于特定题型。

### 5. 对我的实现有什么启发？

`reflection.py` 应该单独维护：

```text
logs/reflection_log.jsonl
```

每条反思最好包含错误类型、失败原因和下一次行动建议，而不是“我要更仔细”这种空话。可以对比三种记忆：

- 无记忆
- 泛泛反思
- 带错误类型标签的具体反思

## 4. CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing

### 1. 它解决什么问题？

CRITIC 关注的是：LLM 单靠自己检查不可靠，但如果像人一样调用外部工具核查，就可能稳定改进输出。它把自我纠错从“模型空想”改成“工具验证后修正”。

### 2. 它的自我纠错 / 反馈机制是什么？

CRITIC 是 verify-then-correct：

```text
初始输出 -> 调用工具验证 -> 得到 critique -> 根据 critique 修改 -> 可重复
```

工具可以是搜索引擎、Wikipedia、代码解释器、计算器、毒性检测器等。关键不是“多改几遍”，而是每次修改都有外部证据或执行结果支撑。

### 3. 它如何评估？有没有用到标准答案？

论文在开放问答、数学程序合成、毒性降低等任务上评测。反馈来自工具交互，不是直接把标准答案喂给模型。论文还强调，只依赖模型自我验证时，提升可能很小甚至退化。

### 4. 它有什么局限？

工具反馈也可能不完整或被误读。搜索结果可能噪声大，代码执行只能验证测试覆盖到的行为，计算器只能检查计算不能检查建模是否正确。

### 5. 对我的实现有什么启发？

项目中的 `critic.py` 应该优先做两个低成本工具：

- `calculator.py`：检查数学中间计算和最终数值
- `code_exec.py`：运行候选代码和单元测试

日志中必须标注 `feedback_source: "tool"`，这样才能和 `feedback_source: "self"` 做对照。

## 5. Large Language Models Cannot Self-Correct Reasoning Yet

### 1. 它解决什么问题？

这篇是项目最重要的反方论文。它问的是：如果没有外部反馈，LLM 真的能纠正自己的推理错误吗？结论偏否定：在 reasoning 任务上，intrinsic self-correction 通常不能稳定提升，甚至会降低性能。

### 2. 它的自我纠错 / 反馈机制是什么？

论文区分两类反馈：

- intrinsic self-correction：模型只根据自身能力检查和修改
- external/oracle feedback：外部告诉模型答案是否正确，或提供工具、标签、证据

它指出很多已有工作里的“自我纠错提升”，其实依赖 oracle label 来决定何时停止或是否继续修改。

### 3. 它如何评估？有没有用到标准答案？

论文用 GSM8K、CommonSenseQA、HotpotQA 等推理任务测试 GPT-3.5、GPT-4、GPT-4-Turbo、Llama-2 等模型。它专门比较有 oracle label 和无 oracle label 的情况，并强调公平比较要控制推理成本，例如和 self-consistency 使用相同数量的模型输出。

### 4. 它有什么局限？

论文主要讨论推理任务中的内生自纠错，不等于否定所有反馈型方法。它并不否认工具反馈、检索证据、单元测试等外部信号的价值。

### 5. 对我的实现有什么启发？

这是项目实验设计的底线：

- 不能用标准答案决定是否采纳修改
- 不能用标准答案决定停止在哪一轮
- 必须单独报告 oracle-feedback 实验
- 多轮自纠错要和等成本 baseline 比较，例如 self-consistency

核心实验应包括：

```text
baseline
self_refine，无外部反馈
self_refine + oracle，仅作为上界
critic，工具反馈
self_consistency，等成本采样 baseline
```

## 6. ExpeL: LLM Agents Are Experiential Learners

### 1. 它解决什么问题？

ExpeL 研究的是 Agent 如何跨任务积累经验。Reflexion 更像在同一任务或同一题上失败后重试，ExpeL 更强调从一批训练任务中收集成功和失败经验，抽象成可复用知识，再用于新任务。

### 2. 它的自我纠错 / 反馈机制是什么？

ExpeL 分三步：

```text
收集成功/失败轨迹 -> 抽取经验规则和 insight -> 测试时检索经验辅助决策
```

它会对经验进行 ADD、UPVOTE、DOWNVOTE、EDIT 等操作，保留更稳定的策略和失败模式。

### 3. 它如何评估？有没有用到标准答案？

论文在多个交互式决策任务上评估经验积累后的性能提升，也测试迁移能力。经验来自任务尝试和反馈，不需要更新模型参数。

### 4. 它有什么局限？

经验抽象可能过拟合训练任务。经验库变大后还需要检索和去噪，否则上下文会变长且混乱。

### 5. 对我的实现有什么启发？

这个项目的高级版可以把 `reflection_log.jsonl` 从“单题反思”升级成“经验库”：

```json
{"error_type": "calculation", "lesson": "百分比题要先确定每个人的基数", "votes": 3}
```

评测时可以分成训练题和测试题，观察经验是否能迁移，而不是只在同一道题上重试。

## 7. Self-Consistency Improves Chain of Thought Reasoning

### 1. 它解决什么问题？

Self-Consistency 解决的是 CoT 贪心解码容易走到单一路径错误的问题。复杂推理题往往有多条解法，如果多个独立推理路径得到同一个答案，这个答案更可能正确。

### 2. 它的自我纠错 / 反馈机制是什么？

它不是反思式纠错，而是采样式纠错：

```text
采样多条 CoT 推理路径 -> 抽取每条最终答案 -> 多数投票
```

它不要求模型指出自己的错误，只利用答案一致性做聚合。

### 3. 它如何评估？有没有用到标准答案？

论文在 GSM8K、SVAMP、AQuA、StrategyQA、ARC-Challenge 等任务上评估，报告显著提升。推理过程中不使用标准答案，最终只用标准答案评测准确率。

### 4. 它有什么局限？

主要成本是调用次数增加。如果模型的多条路径系统性犯同一种错，多数投票也会错。它也不能解释具体哪里错了，只能给出更稳的答案选择。

### 5. 对我的实现有什么启发？

Self-Consistency 是多轮自纠错必须比较的等成本 baseline。比如 `self_refine` 调用 3 次模型，就应该和“采样 3 条答案后投票”比较，否则提升可能只是调用次数更多带来的。

## 8. Chain-of-Verification Reduces Hallucination

### 1. 它解决什么问题？

CoVe 解决的是事实型回答中的幻觉。模型先生成初稿，再主动规划核查问题，独立回答这些核查问题，最后生成修订版。

### 2. 它的自我纠错 / 反馈机制是什么？

流程是：

```text
baseline response
-> plan verification questions
-> answer verification questions independently
-> final verified response
```

关键点是独立回答 verification questions，避免模型在核查时被自己原来的错误答案带偏。

### 3. 它如何评估？有没有用到标准答案？

论文在基于 Wikidata 的列表题、闭卷 MultiSpanQA、长文本生成等任务上评估幻觉减少情况。验证过程主要依赖模型分解出的核查问题，不直接用标准答案指导修改。

### 4. 它有什么局限？

CoVe 更适合事实核查，不一定直接适合数学推理。核查问题如果规划得差，后续验证也会偏。闭卷验证仍然可能受到模型内部知识错误影响。

### 5. 对我的实现有什么启发？

对 SelfCorrect 来说，CoVe 可转化为“验证子问题”：

- 数学题：列出每个中间量应该检查什么
- 代码题：列出需要覆盖的边界测试
- 多跳问答：列出每个事实链节点

它可以作为 `critic` 的一种工具前置步骤：先让模型规划检查项，再交给计算器、单元测试或检索工具验证。

## 综合结论：对 SelfCorrect 项目的设计原则

1. 自我纠错必须区分反馈来源。`self`、`tool`、`oracle` 三类反馈要分开记录和报告。
2. 不能用标准答案决定是否停止或采纳修改。oracle 只能作为上界实验。
3. 每个方法都要记录完整 trace，不能只存最终答案。
4. 评估不仅看 final accuracy，还要看 correct -> wrong 和 wrong -> correct。
5. 多轮方法必须和等成本 baseline 比较，尤其是 self-consistency。
6. 反馈质量是核心变量。空泛反思、具体反思、工具反馈要做消融。
7. 数学题和代码题应该分开看。代码题有单元测试，通常更适合工具型自纠错；数学题更能暴露 intrinsic self-correction 的局限。
8. 最终报告要回答“何时有用”，而不是只证明“某种方法有时能提升”。
