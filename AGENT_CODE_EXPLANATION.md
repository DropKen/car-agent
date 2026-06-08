# Agent 代码与线上负分排查说明

如果要系统学习这套 Agent 的实现思路、核心代码、主要问题和重做建议，请先读 `AGENT_IMPLEMENTATION_REVIEW.md`。本文更偏提交结构和线上负分排查，复盘文档更偏工程学习。

## 线上分数为什么可能是负数

新 0529 demo 的评分核心是：

```text
net_income = gross_income - distance_cost - preference_penalty
```

所以线上显示 `分数 -9151.93 / 偏好扣分 8400` 时，含义不是只扣了偏好分，而是扣偏好前的经营收益也大约只有 `-751.93`。常见原因有三类：

- 提交包结构不对，线上没有加载最新版 agent，而是加载了官方示例 `demo/agent`。
- 隐藏司机偏好没有完全满足，产生 8400 左右罚分。
- agent 为了守偏好过度等待或空驶，接单收益不足，导致扣分前收益也偏低。

## 0529 提交结构

`demo_docs_release_20260529 (2)/docs/04-提交方式.md` 要求压缩包内顶层为 `demo/`，复赛至少包含：

```text
demo/agent/
```

不要提交 `server/data`、`results` 或测试日志。此前顶层只有 `agent/` 的包不适合新 0529 提交流程，可能导致线上加载不到最新版。

## 当前决策思路

入口是 `agent/model_decision_service.py` 的 `ModelDecisionService.decide(driver_id)`。

每轮流程：

1. 通过环境 API 查询司机状态、货源、有限历史。
2. 从偏好文本构造结构化 profile，LLM 只作为增强解析，不写死司机 ID。
3. 用 tools 过滤明显违规货源，包括禁货类、禁区域、固定禁动作窗口、任务冲突、月度休息冲突。
4. 如果出现强任务或硬时间窗，优先执行 `wait` 或 `reposition`。
5. 如果有明确安全且正收益的候选，直接 `take_order`，跳过 planner/reviewer LLM，避免 60 秒以上慢决策和过度保守。
6. 如果候选存在复杂偏好冲突，再让 LLM planner/reviewer 仲裁。
7. 若没有可接单，才考虑低成本找货空驶或等待。

最新调整重点是“收益底盘优先”：

- 正收益、低软罚、无硬冲突的订单直接接。
- 月度整天休息不再提前一天阻断赚钱订单，只在计划当天、已经落后或月底紧急时锁日。
- 软偏好估计不再一律当硬拒绝，避免未知司机因过度保守整月低收入。
- LLM 单次默认超时调整到约 40 秒，并减少传给 LLM 的候选数量。

## Tools 职责

- `DriverProfileTool`：把偏好文本解析成结构化规则，例如禁货类、固定时间窗、月度休息、到访任务、指定货源。
- `TaskCalendarTool`：把任务和周期偏好变成滚动日历，处理固定日期任务、整天休息、月度到访、区域货源订单日。
- `CargoEvaluationTool`：计算订单净收益、空驶成本、小时收益、回锚点成本和基础风险。
- `ActionPreferenceGuardTool`：评估动作是否会带来未来偏好罚分，例如赶不上回家、休息或任务。
- `TaskPenaltyOptimizerTool`：估算当前动作对任务类偏好的直接罚分风险。
- `TimeTaskProgressTool`：跟踪每日连续休息、月度任务、周期任务的完成进度。
- `CommitmentSequenceTool`：处理临时家事、指定日期任务、先到 A 再到 B、停留到某时等顺序任务。
- `RegionPreferenceTool`：检查区域/坐标类偏好，包括到访、禁入、指定区域相关任务。
- `RouteComplianceTool`：检查路线、装卸区域、货类等硬规则。
- `DecisionSupportTools`：提供货源时效、候选可靠性、未来成本等辅助判断。
- `MemoryTool`：压缩历史、记录失败货源和运行期状态。
- `PreferenceClassificationTool`：把偏好归类为硬约束、软偏好、日程任务、未知风险等。
- `PromptTemplates`：生成 profile/planner/reviewer 的 LLM prompt。

## 关键泛化原则

- 不硬编码司机 ID。
- 不读取 `server/data` 或评分文件参与决策。
- 只通过官方 API 获取状态、货源、历史和 LLM。
- 模型名不写在 agent 源码里，由官方 server/config 指定。
- 对未知司机优先保证正收益底盘，再守高罚/硬偏好。

## 当前仍需关注

- 隐藏司机如果组合了“整天休息 + 每日长休息 + 固定禁动作窗口”，需要避免既过度等待又漏休息。
- 任务类偏好必须通过日历提前安排，但不能过早牺牲整月收益。
- 如果线上仍是负分，第一优先排查提交包是否确实是 `demo/agent/` 结构并包含最新版 `TaskCalendarTool`。
