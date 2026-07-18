# 从 ECHO-Repro 到执行反馈驱动的 SelfCorrect 软件工程智能体

我之前的研究聚焦于：

> **让大语言模型根据 Issue、日志和仓库上下文，生成能够真实执行、准确复现目标故障，并用于补丁验证的 Bug Reproduction Test 或 Reproduction Harness。**

这类工件需要满足 **Fail-to-Pass（F→P）**：在 buggy 版本上失败，在 fixed 版本上通过。

目前我已经实现了 ECHO-Repro 原型，包含问题结构化、上下文检索、复现脚本生成、真实执行、失败分类和迭代修正。实验表明该方向可行，但环境、依赖、执行命令和 oracle 仍是主要难点。

因此，我计划选择 **SelfCorrect**，并将其聚焦到编程与软件工程任务：

> **利用代码执行、测试结果和结构化失败反馈进行自纠错，并探索跨任务经验能否提升后续任务成功率。**

这相当于将 ECHO-Repro 中的“生成—执行—诊断—修正—验证”机制抽象为通用 SelfCorrect Agent，再逐步扩展到 Issue-to-BRT、CI Harness 和补丁验证。

------

# 一、此前的研究：ECHO-Repro

## 1.1 研究背景

大语言模型已经能够生成代码、测试和补丁，但在真实软件工程任务中，生成结果经常存在以下问题：

- 代码看起来合理，但由于 API、依赖、路径或测试框架错误而无法执行；
- 脚本虽然失败，但失败原因可能是环境、依赖或脚本本身，而不是目标 bug；
- 测试 oracle 不准确，无法验证 Issue 中描述的真实错误行为；
- 真实仓库还涉及版本、命令、环境变量和配置文件，单独生成测试函数通常不够；
- 补丁通过原有测试，也不一定真正解决了当前 Issue。

因此，我将研究重点从“生成测试函数”扩展为：

> **生成 Environment-aware Reproduction Harness，即同时包含测试逻辑、执行命令、环境适配和验证 oracle 的可执行复现工件。**

------

## 1.2 研究问题

ECHO-Repro 主要研究三个问题。

### 问题一：如何将自然语言 Issue 转换为明确的复现目标？

系统需要从非结构化 Issue 中提取：

- 当前行为和预期行为；
- 报错信息和复现提示；
- 环境版本；
- 关键词和可疑函数。

核心是明确“应该触发什么行为”和“什么结果才算复现成功”。

### 问题二：如何让生成结果适配真实仓库？

除了测试代码，还需要确定：

- 执行目录和执行命令；
- 测试框架和已有测试模板；
- 项目依赖和导入方式；
- 环境变量和配置文件。

### 问题三：如何判断是否真实复现？

有效复现需要满足：

```text
Buggy Version：失败，并表现出目标错误
Fixed Version：通过
```

即：

```text
Buggy Fail + Fixed Pass = Fail-to-Pass
```

只在 buggy 版本上失败并不够，因为失败可能来自环境、依赖、脚本或 oracle。

------

## 1.3 ECHO-Repro 的方法

整体流程为：

```text
Issue / CI Log / Repository
            ↓
BugSpec Extraction
            ↓
Source / Test / Environment Retrieval
            ↓
Concise Context Construction
            ↓
Reproduction Harness Generation
            ↓
Execution + Failure Classification
            ↓
Targeted Revision
            ↓
Buggy / Fixed Validation
```

### 1.3.1 BugSpec Extraction

系统利用 LLM 和规则回退机制，将原始 Issue 转换为结构化 `BugSpec`：

```text
title
current_behavior
expected_behavior
failure_signature
reproduction_hint
keywords
suspect_symbols
```

其作用是为后续检索、生成和验证提供统一任务描述。

### 1.3.2 三路上下文检索

系统分别检索：

| 上下文            | 作用                             |
| ----------------- | -------------------------------- |
| Source Code       | 理解相关 API 和错误路径          |
| Existing Tests    | 学习测试框架、fixture 和断言方式 |
| Environment Files | 获取依赖、命令和项目配置         |

检索范围包括源码、测试文件、依赖文件和 CI workflow。

### 1.3.3 Concise Context Construction

系统不会直接输入整个仓库，而是压缩为：

```text
Bug Summary
Current / Expected Behavior
Failure Signature
Relevant Source Code
Relevant Tests
Environment Configuration
```

这样可以降低 Token 消耗并减少无关信息干扰。

### 1.3.4 Harness Generation

模型生成完整复现工件，包括：

- `reproduce.py` 或测试文件；
- 输入构造和调用路径；
- 断言或错误签名；
- 执行命令和环境检查。

ECHO-Repro 当前不直接修改 bug 源码，而是生成一个可用于后续补丁验证的独立工件。

### 1.3.5 执行、验证与失败分类

系统在真实仓库中执行脚本，收集 exit code、stdout、stderr、traceback 和测试结果，并将失败分类为：

```text
dependency_error
environment_error
harness_error
oracle_error
trigger_miss
fake_reproduction
buggy_pass
fixed_fail
reproduced
```

不同类型对应不同修正策略。例如：

- dependency error：检查依赖和版本；
- harness error：修改语法、导入和调用方式；
- oracle error：修改断言；
- trigger miss：重新构造输入和触发路径。

因此，ECHO-Repro 的核心不是让模型主观反思，而是：

> **根据真实执行产生的结构化反馈进行针对性修正。**

------

## 1.4 初步实验结果

### 1.4.1 早期原型实验

在早期 20 条仓库级样本中：

- 成功 12 条；
- 失败 8 条。

主要失败包括：

| 失败类别          | 数量 |
| ----------------- | ---- |
| 未触发目标问题    | 3    |
| dependency_error  | 2    |
| environment_error | 1    |
| harness_error     | 1    |
| oracle_error      | 1    |

结果说明，LLM 能够为部分真实 Issue 生成有效复现脚本，但环境、依赖和 oracle 仍是主要瓶颈。

### 1.4.2 CI-Repair-Bench 扩展实验

随后比较了三种输入：

- S1：Issue Only；
- S3：Log Only；
- S5：Full Context。

校正后的 30×3 实验结果为：

| 设置            | 可执行 | F→P         |
| --------------- | ------ | ----------- |
| S1 Issue Only   | 30/30  | 7/30，23.3% |
| S3 Log Only     | 30/30  | 4/30，13.3% |
| S5 Full Context | 29/30  | 7/30，23.3% |

目前可以得到以下结论：

1. CI Log 单独使用时信息不足；
2. Full Context 在当前样本上没有总体优于 Issue Only；
3. 不同上下文的价值可能与具体失败类型有关；
4. Executor、环境构建和 oracle 的正确性会直接影响实验结论；
5. 当前工作的价值不仅是最终 F→P 数字，也包括建立了可重复执行和可定位失败原因的验证框架。

------

## 1.5 当前研究结论

我目前形成的核心认识是：

> **软件工程智能体的瓶颈不只是第一次生成能力，而是失败后能否识别原因并选择正确的修正动作。**

纯语言反思可能误解日志、重复错误策略、混淆环境与程序问题，甚至制造伪失败。因此，更可靠的自纠错需要：

```text
External Execution
        +
Typed Failure Diagnosis
        +
Controlled Revision
```

这也是我计划在 SelfCorrect 项目中进一步研究的核心机制。

------

# 二、我已经阅读过的相关论文

## 2.1 2023：LIBRO

### [Large Language Models are Few-shot Testers: Exploring LLM-based General Bug Reproduction](https://dl.acm.org/doi/10.1109/ICSE48619.2023.00194)

**会议：ICSE 2023**

LIBRO 根据 Bug Report 多次生成候选测试，并通过注入、执行、聚类和排序筛选 Bug Reproduction Test，证明了 LLM 从自然语言问题描述生成复现测试的可行性。

------

## 2.2 2024：AGENTLESS

### [AGENTLESS: Demystifying LLM-based Software Engineering Agents](https://arxiv.org/abs/2407.01489)

**正式发表：FSE 2025**

AGENTLESS 将软件修复拆分为：

```text
Localization → Repair → Patch Validation
```

它通过层次化定位、多候选补丁和复现测试完成补丁筛选，在 SWE-bench Lite 上达到 32.0%。

**启发：** Agent 不一定越自由越好，受控、分阶段的流程可能更加稳定。

------

## 2.3 2024：SpecRover

### [SpecRover: Code Intent Extraction via LLMs](https://dl.acm.org/doi/10.1109/ICSE55347.2025.00080)

**正式发表：ICSE 2025**

SpecRover 在定位代码的同时推断函数的预期行为，并由 Reviewer Agent 结合 Issue、测试和执行结果审查补丁。

**启发：** 自纠错不仅要知道“哪里错了”，还要理解程序“应该如何表现”。

------

## 2.4 2024：AEGIS

### [AEGIS: An Agent-based Framework for Bug Reproduction from Issue Descriptions](https://dl.acm.org/doi/10.1145/3696630.3728557)

**正式发表：FSE 2025 Industry Track**

AEGIS 通过简洁上下文构建和有限状态机，约束 Agent 的创建、执行、验证、修改和重启过程，解决子任务混乱、上下文过长和行为失控问题。

**启发：** 不同执行反馈应触发不同的修正状态和动作。

------

## 2.5 2025：Google BRT Agent

### [Agentic Bug Reproduction for Effective Automated Program Repair at Google](https://arxiv.org/abs/2502.01821)

该工作面向 Google 内部代码库生成 Bug Reproduction Test。BRT Agent 在 80 个工业 Bug 上达到 28% 的 plausible BRT 生成率，并使 APR 系统可修复的 Bug 数量提升约 30%。

**启发：** BRT 不只是测试，还可以作为补丁生成上下文、补丁筛选器和独立验证工件。

------

## 2.6 2025：Otter

### [Otter: Generating Tests from Issues to Validate SWE Patches](https://proceedings.mlr.press/v267/ahmed25b.html)

**会议：ICML 2025**

Otter 由 Localizer、Self-reflective Action Planner 和 Test Generator 构成，并结合规则检查修复生成结果。Otter 的 F→P 为 31.4%，Otter++ 达到 37.0%。

**启发：** Self-Reflection 需要结合规则检查和真实执行，而不能只依赖模型自评。

------

## 2.7 2026：SWE-CI

### [SWE-CI: Evaluating Agent Capabilities in Maintaining Codebases via Continuous Integration](https://arxiv.org/abs/2603.03823)

SWE-CI 通过连续多轮需求和代码修改评价 Agent 的长期维护能力，并使用 EvoScore 衡量软件演化过程中的正确性和可维护性。

**启发：** 自纠错不仅要关注当前任务成功，还要考虑回归错误和长期影响。

------

## 2.8 2026：iCoRe

### [iCoRe: An Iterative Correlation-Aware Retriever for Bug Reproduction Test Generation](https://dl.acm.org/doi/10.1145/3808193)

**会议：FSE 2026**

iCoRe 区分生产代码与测试代码检索，并结合函数调用关系和生成结果反馈迭代优化上下文，在两个 Benchmark 上达到 42.0% 和 52.8% 的 F→P。

**启发：** 生成失败后，不一定只需要修改答案，也可能需要重新检索上下文。

------

## 2.9 2026：CI-Repair-Bench

### [CI-Repair-Bench: A Repository-Aware Benchmark for Automated Patch Validation via CI Workflows](https://arxiv.org/abs/2604.27148)

CI-Repair-Bench 包含 103 个仓库中的 567 个真实 CI failure instances，覆盖日志、Workflow、失败提交、修复提交和 12 类 CI 错误。

**启发：** 软件工程自纠错不能只处理代码逻辑错误，还需要处理依赖、环境、配置和执行命令问题。



------

# 三、课程项目选择：SelfCorrect 软件工程智能体

## 一、核心 Insight

### 1. 自我纠错的关键不是“再想一次”，而是“获得可靠反馈”

纯语言反思容易出现两种问题：

- 错误答案没有被真正纠正；
- 原本正确的答案反而被改错。

因此，可靠的 SelfCorrect 应建立在：

```
真实执行反馈+结构化错误诊断+针对性修正
```

而不是简单让模型重复生成。

### 2. 不同失败原因需要不同修正策略

编程和软件工程任务中的失败并不都是代码逻辑错误，还可能来自：

- 语法和运行时错误；
- 单元测试失败；
- 依赖或环境错误；
- 执行命令错误；
- oracle 或验证规则错误。

因此，应先进行 **Failure Classification**，再决定修改代码、调整环境、重新检索上下文，还是修改验证条件。

### 3. 外部工具反馈可能是自我纠错有效的前提

项目可以重点比较：

```
直接生成vs.纯 LLM Self-Reflectionvs.原始执行日志反馈vs.结构化工具反馈
```

核心问题不是模型能否改答案，而是：

> **哪一种反馈能够真正提高 Wrong→Correct，同时减少 Correct→Wrong。**

这与老师研究计划中“自我纠错什么时候有用、什么时候帮倒忙”的核心目标一致。

### 4. 记忆不是越多越好，而应保存“被验证的经验”

长期记忆不应保存完整对话，而应保存：

```
任务类型+ 失败类型+ 错误证据+ 无效策略+ 成功修正策略+ 验证结果
```

只有经过单元测试、代码执行或 Fail-to-Pass 验证的经验，才进入长期记忆，避免错误反思在后续任务中传播。

### 5. SelfCorrect 可以成为 ECHO-Repro 的通用机制层

我之前的 ECHO-Repro 已经包含：

```
生成 Harness→ 执行→ 识别失败→ 修改→ 再验证
```

因此课程项目不是重新开一个方向，而是先在标准编程题上研究自纠错机制，再迁移到真实仓库：

```
数学 / HumanEval
        ↓
工具反馈自纠错
        ↓
失败类型感知纠错
        ↓
跨任务经验记忆
        ↓
Issue-to-BRT
        ↓
CI Log-to-Reproduction Harness
        ↓
Patch Validation / APR
```

------

## 二、初步项目思路

项目暂定为：

> **ECHO-SC：基于执行反馈、失败分类与经验记忆的编程自纠错智能体**

第一阶段先在 HumanEval 或简单 Python 编程任务上实现四种方法：

| 方法                     | 核心机制                   |
| ------------------------ | -------------------------- |
| Baseline                 | 单次生成，不纠错           |
| Self-Reflection          | 模型自己检查并改写         |
| Tool Feedback            | 根据代码执行和单元测试修改 |
| Failure-Aware Correction | 先分类错误，再选择修正策略 |
| Experience Memory        | 检索历史成功纠错经验       |

重点研究三个问题：

1. **工具反馈是否比模型自评更可靠？**
2. **结构化失败类型是否比原始报错更有效？**
3. **经过验证的历史经验能否提高后续任务成功率？**

评价不仅看最终准确率，还应看：

```
Wrong → Correct：真正纠错
Correct → Wrong：负向纠错
Wrong → Wrong：无效纠错
Correct → Correct：保持正确
```

------

## 三、后续迁移思路

课程阶段先完成一个可控、可评测的 SelfCorrect Agent。

完成后，再将其接入 ECHO-Repro：

```
Issue / Log / Repository
        ↓
生成 Reproduction Harness
        ↓
执行并获取反馈
        ↓
判断失败属于环境、依赖、脚本、触发还是 oracle
        ↓
检索相似经验
        ↓
针对性修改
        ↓
Fail-to-Pass 验证
```

最终希望形成的不是一个只会反复修改答案的 Agent，而是：

> **一个能够利用真实执行证据理解失败原因、选择正确纠错动作，并从经过验证的历史经验中持续改进的软件工程智能体。**



