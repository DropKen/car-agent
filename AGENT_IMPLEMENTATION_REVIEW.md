# 天池卡车司机 Agent 实现复盘与学习文档

本文档整理当前 Agent 的核心代码、实现思路、有效经验、失败原因和后续改进方向。它不是提交说明，而是项目复盘材料，目的是帮助我们理解这套 Agent 为什么能跑、为什么仍然不够好，以及如果重新做应该怎么设计。

## 1. 项目目标

比赛题目是“基于 Agentic AI 的卡车司机连续找货决策”。Agent 每一步只能通过官方环境接口观察状态，然后输出一个动作。

允许动作只有三类：

```text
take_order
wait
reposition
```

核心目标不是单步收益最大，而是整个月净收益最大：

```text
net_income = gross_income - driving_cost - preference_penalty
```

所以决策需要同时考虑：

- 当前订单收益。
- 去装货点的空驶成本。
- 运输距离和时间。
- 订单结束后的位置是否会影响后续接单。
- 是否会错过每日休息、禁动作窗口、固定任务、月度任务。
- 是否会触发司机偏好扣分。
- 是否会因为 LLM 超时导致默认 wait。

## 2. 当前核心代码清单

### 2.1 主入口

```text
agent/model_decision_service.py
```

核心类：

```python
class ModelDecisionService:
    def decide(self, driver_id: str) -> dict[str, Any]:
        ...
```

这是官方模拟器每一步调用的入口。所有动作最终都从这里返回。

它做的事情包括：

- 调用 `get_driver_status(driver_id)` 获取当前位置、时间、偏好、车辆信息。
- 调用 `query_decision_history(driver_id, step)` 获取有限历史。
- 调用 `query_cargo(driver_id, latitude, longitude, k)` 获取附近货源。
- 调用 `model_chat_completion(payload)` 使用官方 LLM 接口。
- 输出 `take_order`、`wait` 或 `reposition`。

### 2.2 工具目录

```text
agent/tools/
```

主要工具如下：

| 文件 | 职责 |
| --- | --- |
| `driver_profile_tool.py` | 把自然语言偏好解析成结构化 profile |
| `task_calendar_tool.py` | 把固定日期、整天休息、禁动作窗口、月度任务变成滚动日历 |
| `commitment_sequence_tool.py` | 处理“先到 A，再到 B，停留到某时”的顺序任务 |
| `time_task_progress_tool.py` | 统计每日休息、月度到访、整天休息等完成进度 |
| `cargo_evaluation_tool.py` | 计算订单收益、空驶成本、小时收益、回锚点成本 |
| `action_preference_guard_tool.py` | 判断某个候选订单是否会导致未来偏好风险 |
| `route_compliance_tool.py` | 检查路线、货类、禁区等硬约束 |
| `region_preference_tool.py` | 判断区域偏好、坐标锚点、月度到访点影响 |
| `task_penalty_optimizer_tool.py` | 估算任务冲突时的罚分风险 |
| `decision_support_tools.py` | 提供候选可靠性、时效、未来价值等辅助特征 |
| `memory_tool.py` | 压缩历史，保留运行期记忆 |
| `preference_classification_tool.py` | 给偏好分硬约束、软偏好、任务、未知风险等类别 |
| `prompt_templates.py` | 构造 LLM profile/planner/reviewer prompt |

## 3. 当前 Agent 的整体架构

当前方案不是纯规则，也不是纯 LLM，而是“LLM 解析 + 工具评分 + 守卫校验”的混合架构。

整体流程：

```text
状态读取
  -> 偏好文本解析
  -> 生成结构化 profile
  -> 检查硬安全动作
  -> 查询货源
  -> 构造候选订单
  -> 工具计算收益/风险
  -> 高置信候选直接接单
  -> 复杂权衡交给 LLM
  -> LLM 输出再校验
  -> 返回官方动作
```

为什么这样设计：

- 纯 LLM 容易超时，线上超时后就会变成 wait 或无效动作。
- 纯 LLM 容易幻觉，不一定严格遵守三类动作。
- 纯规则泛化差，中文偏好表达一变就容易漏解析。
- 工具可以保证基本安全，LLM 可以补充语义理解和复杂取舍。

## 4. `decide()` 的核心决策链路

主流程在 `agent/model_decision_service.py` 的 `ModelDecisionService.decide()`。

### 4.1 读取状态与历史

```python
status = self._api.get_driver_status(driver_id)
current_minute = int(status.get("simulation_progress_minutes", 0) or 0)
self._prepare_decision_history(driver_id, current_minute)
```

这里有一个重要修复：不再每个工具都调用 `query_decision_history(step=-1)`。现在是在每轮开头拉有限历史并缓存。

这样做的原因：

- 月底历史会非常长，反复全量拉历史会越来越慢。
- 线上每个 action 有时间限制，慢了就可能全部 wait。
- 近几十步历史通常足够判断每日休息、近期任务、是否已经接过某个熟货。

不足：

- 有些月度任务需要全月统计，只看有限历史可能丢信息。
- 后来用运行期缓存补了一部分，但如果进程重启，仍可能丢长期记忆。

### 4.2 原始文本硬守卫

```python
raw_forbidden_escape = self._raw_forbidden_circle_escape_action(...)
raw_home = self._raw_night_home_action(...)
visible_no_action = self._visible_daily_no_action_window_action(...)
```

这些逻辑直接看原始偏好文本，不等 LLM/profile。

原因：

- “不得进入某圆形区域”这类约束非常危险，解析失败会造成高罚。
- “晚上几点到几点不接单不空跑”也属于高频硬约束。
- 这些规则表达相对固定，直接正则处理更快更稳。

问题：

- 正则越写越多，代码复杂度会膨胀。
- 有些软表达和硬表达边界不清，比如“尽量回家”不应该当成必须回家。
- 这也是后来补“软回家识别”的原因。

### 4.3 生成结构化 profile

```python
profile = self._planning_profile(driver_id, preferences)
profile = self._profile_scoped_for_current_time(profile, current_minute)
profile = self._profile_without_seen_cargo_commitments(driver_id, profile)
profile = self._profile_without_forbidden_required_cargos(driver_id, profile)
```

profile 是把自然语言偏好变成结构化字段：

```text
avoid_cargo_keywords
avoid_regions
daily_rest
required_off_days
pickup_deadhead_max_km
monthly_deadhead_limit_km
max_haul_km
first_order_deadline_minute
daily_order_limit
geo_fence_bounds
forbidden_circles
required_cargos
temporary_events
long_sequence_commitments
visit_frequency
scheduled_visits
```

设计意图：

- 让后续工具不直接处理自然语言。
- 让 LLM 解析结果和 deterministic parser 结果可以合并。
- 所有候选订单都用同一套 profile 判断。

主要问题：

- profile 字段太多，后期维护成本高。
- 部分字段语义重叠，比如 `scheduled_visits`、`temporary_events`、`long_sequence_commitments` 都可能表达固定任务。
- 字段之间冲突时没有一个统一的约束求解器，只能靠优先级和 guard patch。

### 4.4 任务和日历优先级

任务类逻辑来自：

```text
TaskCalendarTool
CommitmentSequenceTool
TimeTaskProgressTool
```

核心思想：

- 固定日期任务不要等到当天才处理。
- 但也不能提前太久就不接单。
- 日历工具会给候选订单设置 `candidate_finish_deadline`，如果订单会占用任务预留时间，就过滤掉。

这是项目里比较有效的设计。

它解决了：

- 婚礼、搬家、祭祖、培训、盘点、家宴等固定任务。
- “先到 A，停 20 分钟，再到 B”的顺序任务。
- 月度整天休息。
- 月度到访/区域货源任务。
- 每日禁动作窗口。

但它也带来收益问题：

- 如果任务解析得过重，会过早等待。
- 如果周期任务被当成紧急任务，会反复去同一个点。
- 如果整天休息安排太保守，会少接很多单。

### 4.5 查询货源与候选构造

```python
raw = self._query_cargo(driver_id, lat, lng, profile)
items = raw.get("items", [])
candidates = self._build_candidates(driver_id, status, items, profile)
```

每个候选订单会计算：

- 去装货点距离。
- 装货到卸货距离。
- 收入。
- 成本。
- 当前净收益。
- 订单结束时间。
- 日历冲突。
- 偏好风险。
- 是否需要回到家/锚点。
- 软地域越界的收益补偿。

这是实际赚钱的核心层。

当前排序大致是：

```text
score = net + hourly_weight * hourly - pickup_penalty - geo_return_cost - calendar_penalty - soft_penalty
```

问题在于：

- 这个 score 是经验式的，不是全局最优。
- 它只看一步订单，没有真正做月度动态规划。
- 它对未来货源分布没有建模，只能靠目的地、锚点和当前附近货源粗略判断。

### 4.6 高置信候选直接接单

```python
confident = self._confident_income_candidate(candidates)
if confident is not None:
    return take_order
```

这个 fast path 是后来非常重要的修复。

原因：

- 早期过度依赖 LLM，LLM 慢或保守会导致大量 wait。
- 比赛里不接单比轻微偏好罚分更可怕。
- 如果工具已经判断订单正收益且无硬冲突，就应该直接接。

这一步提升了收益底盘。

但问题是：

- “高置信”标准很难调。
- 标准太松会扣分，太严会少赚钱。
- 它仍然是一阶贪心，不保证整月最优。

### 4.7 LLM 的角色

当前 LLM 主要有三个作用：

1. 解析偏好：把自由文本转 profile。
2. 候选比较：在多个工具评分候选之间做取舍。
3. 动作复核：检查推荐动作是否明显违背偏好。

LLM 不应该做的事：

- 不应该直接自由生成最终动作。
- 不应该指定模型名或 key。
- 不应该读取本地数据文件。
- 不应该绕过工具 guard。

当前代码中 LLM 调用必须通过：

```python
self._api.model_chat_completion(payload)
```

这是官方要求。

## 5. 为什么做了很多测试，效果仍然不够好

这是最关键的复盘。

### 5.1 我们优化的是“局部能活”，不是“全局最优”

当前 Agent 的本质仍然是单步贪心加局部修补。

它每一步看当前货源，然后判断：

- 这个订单能不能接。
- 接完会不会触发近期任务冲突。
- 当前收益是否足够好。

但它没有真正求解：

```text
整个月路线 + 任务 + 休息 + 货源分布 + 偏好罚分 的全局最优策略
```

所以它会出现：

- 当前订单看起来赚钱，但把司机带到差区域。
- 当前为了守任务等待，但其实可以先接一个短单。
- 当前为了 0 扣分牺牲太多收益。
- 当前为了高收益接受远单，但未来回不来。

真正强的方案应该有滚动 horizon planning，例如看未来 6-24 小时或 2-3 天的机会成本，而不是只看一个订单。

### 5.2 偏好解析被迫写成了“补丁集合”

中文偏好表达太多：

```text
不接
别派
不能碰
不太想干
尽量别去
如果利润高可以
必须
最好
尽量
待到
停到
随后
先到
再到
```

早期做法是发现一个问题补一个 parser：

- 婚礼补婚礼词。
- 搬家补搬家词。
- 禁入圈补禁入圈正则。
- 熟货坐标错了再补 item-scoped parser。
- 软回家误判再补 soft-home detector。

这种方式短期有效，但长期会变成规则泥潭。

更好的方式应该是：

- 先建立统一的偏好语义 schema。
- 每条偏好先分类：硬约束、软偏好、固定任务、周期任务、收益权衡、禁止货类、地理约束。
- 再统一转换成 constraints，而不是每个工具自己理解文本。
- 用 LLM 输出结构化 JSON 后，再用 deterministic validator 校验。

### 5.3 为了 0 扣分，牺牲了太多收益

很多测试结果看起来“偏好扣分 0”，但收益不一定高。

这是因为策略偏安全：

- 不确定就 wait。
- 任务临近就提前回点。
- 地域不确定就少接远单。
- 日休不确定就提前锁休息。

但比赛目标是净收益，不是偏好扣分最小。

更合理的目标应该是：

```text
最大化：订单利润 - 空驶成本 - 预期偏好罚分 - 未来机会成本
```

也就是说，低罚偏好应该可以被高收益覆盖。

当前虽然加了部分软罚逻辑，但不够系统。

### 5.4 缺少真实“市场模型”

当前 Agent 只知道当前 `query_cargo` 返回的货源。

它不知道：

- 哪些区域未来货源更多。
- 哪些时间段货价更高。
- 某个目的地是否会导致后续无货。
- 某类司机适合在哪个城市循环。

我们尝试用 anchors、preference points、cargo density 等近似，但这不是完整市场模型。

更好的方案应该在线学习：

- 每次 query_cargo 记录区域、时间、货源数量、价格分布。
- 建立区域热度表。
- 对目的地打未来价值分。
- 空驶不是只回家，也可以去高收益货源区。

### 5.5 测试集驱动导致过拟合风险

虽然代码没有写死司机 ID，但优化过程高度依赖已知司机和人工合成司机。

这会导致：

- 我们修的都是看见的问题。
- 未知隐藏司机的表达方式可能完全不同。
- 合成司机虽然数量多，但货源环境和偏好模板仍来自已知数据。

“没有写死司机 ID”不等于“没有过拟合”。

过拟合可能来自：

- 偏好词表过拟合。
- 时间表达过拟合。
- 惩罚权重过拟合。
- 地域范围过拟合。
- 测试货源分布过拟合。

### 5.6 工具太多，缺少统一优化器

当前工具很多，每个工具都在局部判断。

问题是：

- A 工具觉得应该等。
- B 工具觉得可以接单。
- C 工具觉得要去任务点。
- D 工具觉得这个区域不好。

最后只能靠手写优先级判断。

这会导致：

- 优先级冲突。
- 代码难读。
- 修一个司机容易伤另一个司机。
- 行为很难解释为一个统一目标函数。

更理想的是：

```text
所有约束 -> 统一候选动作集合 -> 统一打分函数 -> 统一选择器
```

其中硬约束直接过滤，软约束变成成本，任务变成 deadline/penalty，市场价值变成收益。

### 5.7 没有足够好的反事实评估

我们经常看到某个司机收益低，但不知道：

- 是货源本来少？
- 是任务约束太强？
- 是 Agent 过早等待？
- 是错过了高收益订单？
- 是接了错误方向的单？

真正需要的是反事实工具：

- 如果忽略某条偏好，收益会高多少？
- 如果晚 2 小时去任务点，会扣多少，赚多少？
- 如果接当前第二候选，后面会怎么样？
- 如果不回家，实际罚分与收益差多少？

当前只有局部日志，没有系统反事实分析。

## 6. 当前实现中做得对的地方

虽然效果不理想，但有些设计是有价值的。

### 6.1 官方接口合规

代码没有读取隐藏数据，也没有写死模型名/key。

只使用官方接口：

```text
get_driver_status
query_cargo
query_decision_history
model_chat_completion
```

这是提交底线。

### 6.2 LLM 不直接控制最终动作

LLM 只做解析和选择建议，最终动作经过工具校验。

这避免了：

- 非法 action。
- 幻觉 cargo_id。
- 忽视硬约束。
- 超时导致整月 wait。

### 6.3 日历工具是正确方向

固定日期任务不应该散落在各处判断。

把它抽成 `TaskCalendarTool` 是正确的，因为任务本质上就是：

```text
在某个时间前/某个时间段，到某个地方，停留一段时间
```

它应该影响候选订单的 finish deadline。

### 6.4 必接熟货和禁入圈修复很有学习价值

曾经出现过一个典型 bug：

- 偏好里有禁入圈坐标。
- 后面又有熟货坐标。
- 解析器把全文第一个坐标当成熟货坐标。
- Agent 去追一个实际上位于禁入圈中心的“伪熟货”。

这个问题说明：

- 结构化抽取必须保留文本作用域。
- 不同偏好项不能随便拼接后解析。
- 坐标、货源编号、时间必须绑定到同一句/同一条偏好。

这类经验非常重要。

### 6.5 软硬偏好区分是正确方向

“必须回家”和“尽量回家”不是一回事。

“危险品不能碰”和“食品饮料不太想干但利润好可以接”也不是一回事。

如果全当硬约束，收益会差。

如果全当软约束，罚分会爆。

当前已经开始做软硬区分，但还不够系统。

## 7. 当前实现中主要做错的地方

### 7.1 一开始过度相信 LLM

早期思路偏向：让 LLM 理解偏好，再让 LLM 做规划。

问题是比赛环境不是聊天问答，而是高频在线决策。

LLM 的缺点被放大：

- 慢。
- 不稳定。
- 容易过度保守。
- 输出需要校验。
- 每步都调用成本高且有超时风险。

正确方式应该是：

- LLM 做离线/低频解析。
- 高频 action 用 deterministic planner。
- 只有复杂冲突再调用 LLM。

### 7.2 没有一开始就建立统一约束模型

我们是边发现偏好边加工具。

导致后期代码出现：

- raw guard。
- profile guard。
- calendar guard。
- commitment guard。
- action guard。
- LLM review。

它们都在处理“偏好”，但没有统一中间表示。

如果重做，应该先定义：

```python
Constraint(
    type="time_window|cargo_ban|geo_fence|visit|rest|soft_cost|order_limit",
    hard=True/False,
    penalty_amount=...,
    penalty_cap=...,
    active_window=...,
    target_point=...,
    condition=...,
)
```

然后所有模块都基于 Constraint 工作。

### 7.3 没有真正做多步规划

当前是“当前订单 + 近期 guard”。

真正的问题是连续找货，应该至少做滚动规划：

```text
当前动作 -> 当前订单结束 -> 下一位置货源价值 -> 任务/休息 deadline -> 下一步动作
```

即使不能全月 DP，也应该做 6-12 小时 lookahead。

比如：

- 接长单会不会错过晚间休息？
- 去某个偏远城市后有没有货回来？
- 现在空驶去热点是否比等待更好？
- 今天是否应该主动凑月度访问任务？

这些现在做得很粗。

### 7.4 过度追求 0 扣分

我们很多时候把“无扣分”当成优化成功。

但比赛目标是净收益最大。

如果一个订单能多赚 3000，但会扣 200，其实应该接。

如果为了避免 200 扣分等半天，可能亏更多。

当前代码虽然加入了 soft penalty，但行为仍偏保守。

### 7.5 合成未知司机不能代表真实隐藏司机

我们造了很多未知司机，但仍然使用相似表达、相似地点、相似货源环境。

这能发现部分问题，但不能证明泛化强。

更好的测试应该包括：

- 完全不同的中文表达。
- 模糊、省略、口语化偏好。
- 不同城市和坐标范围。
- 矛盾偏好。
- 低收益货源稀疏环境。
- 随机扰动货源价格和时间。

## 8. 如果重新做，推荐架构

### 8.1 第一层：偏好语义抽取

输入：原始 preferences。

输出：统一 constraints。

示例：

```json
{
  "constraints": [
    {
      "id": "c1",
      "type": "cargo_category",
      "mode": "ban",
      "hardness": "hard",
      "keywords": ["危险品", "化工塑料"],
      "penalty_amount": 1500
    },
    {
      "id": "c2",
      "type": "daily_no_action_window",
      "hardness": "hard",
      "start_minute": 1320,
      "end_minute": 300
    },
    {
      "id": "c3",
      "type": "home_return",
      "hardness": "soft",
      "deadline_minute": 1380,
      "point": [23.12, 113.28],
      "penalty_amount": 500
    }
  ]
}
```

LLM 可以参与抽取，但必须有 schema validator。

### 8.2 第二层：候选动作生成

每步生成候选动作：

```text
take_order candidates
wait candidates
reposition candidates
schedule-task candidates
rest candidates
```

不要只在没订单时才考虑 reposition。主动去热点也应该是候选。

### 8.3 第三层：统一评分

每个候选动作都算：

```text
score = immediate_profit
      - immediate_cost
      - expected_preference_penalty
      - schedule_risk_cost
      + future_market_value
      - uncertainty_penalty
```

硬约束直接不可行。

软约束进入成本函数。

### 8.4 第四层：滚动规划

至少做短 horizon：

```text
1-step: 当前订单
2-step: 订单结束后附近货源
schedule-step: 最近任务/休息 deadline
```

不用精确模拟全月，也能比当前贪心强很多。

### 8.5 第五层：反事实评估与日志

每次低收益或等待时记录：

- 为什么不接最赚钱订单？
- 被哪条约束挡住？
- 如果接会扣多少？
- 如果等，预期收益是多少？
- 当前等待是任务等待、休息等待，还是无货等待？

这样后续优化不会靠猜。

## 9. 当前文件里的关键学习点

### 9.1 `model_decision_service.py`

学习点：

- 如何接入官方 API。
- 如何把 LLM 限制在官方 `model_chat_completion`。
- 如何把决策拆成状态、profile、候选、guard、LLM review。
- 如何避免 `step=-1` 历史全量查询。
- 如何在 action 返回前统一附带 `model_usage`。

不足：

- 文件太大，职责过多。
- 决策优先级靠 if-else 堆叠。
- 很多策略参数散落在环境变量里。
- 缺少统一的候选动作评分器。

### 9.2 `driver_profile_tool.py`

学习点：

- 中文时间、坐标、货类、任务语义的基础抽取。
- 偏好项作用域的重要性。
- 软硬偏好区分的重要性。

不足：

- 正则和词表太多。
- 对未知表达泛化有限。
- schema 不够统一，后续工具字段依赖复杂。

### 9.3 `task_calendar_tool.py`

学习点：

- 固定任务必须进入日历，而不是临时判断。
- 任务要给候选订单设置 finish deadline。
- 远期任务不能太早压制接单。

不足：

- 月度任务和固定任务冲突时没有全局优化。
- 整天休息选日策略仍偏经验。
- 有些任务会导致过早等待。

### 9.4 `commitment_sequence_tool.py`

学习点：

- 顺序任务需要状态机。
- “到达 + 停留”必须用历史证明，不能只看当前时间。
- 任务完成判断比任务触发更难。

不足：

- 任务步骤来源仍依赖解析质量。
- 多任务并发时优先级不够统一。

### 9.5 `cargo_evaluation_tool.py`

学习点：

- 收益必须扣空驶成本。
- 小时收益很重要。
- 回锚点成本能缓解地域司机乱跑。

不足：

- 没有准确未来货源价值。
- 没有真正建模“目的地区域机会”。

## 10. 实验结果怎么解读

我们跑了多批合成未知司机，很多结果是：

```text
偏好扣分 0
收益为正
```

这说明：

- Guard 和日历确实能防很多扣分。
- 不容易 all-wait。
- 基础兼容性比早期好。

但这不能说明 Agent 强。

因为：

- 合成司机不等于隐藏司机。
- 0 扣分可能是过度保守。
- 收益正不等于收益最大。
- 没有和更强 baseline 比较。
- 没有系统反事实。

真正应该比较：

```text
当前 Agent vs 忽略偏好贪心
当前 Agent vs 简单收益贪心 + 硬约束
当前 Agent vs 带 12 小时 lookahead 的 planner
当前 Agent vs 人工策略
```

## 11. 为什么线上/未知可能还是差

即使本地多批未知司机表现不错，线上仍可能差，原因包括：

- 隐藏偏好表达没有被 parser 覆盖。
- 隐藏货源分布和本地不同。
- 任务组合更极端。
- LLM 响应慢导致线上超时。
- 提交包结构或依赖不一致。
- 某条软偏好被当硬约束。
- 某条硬偏好被当软约束。
- 一步贪心把司机带到低货源区域。
- 月度统计因为有限历史丢失。

## 12. 最重要的经验总结

### 12.1 Agentic AI 不等于把决策交给 LLM

LLM 更适合：

- 理解文本。
- 生成结构化解释。
- 辅助复杂权衡。

LLM 不适合：

- 高频实时决策。
- 无校验地输出动作。
- 精确时间/距离/收益计算。

### 12.2 先建模，再写规则

这次最大的问题是边测边补规则。

正确顺序应该是：

```text
定义状态
定义动作
定义约束
定义收益函数
定义未来价值
定义 planner
最后才做 parser 和 LLM
```

### 12.3 不要只盯扣分

低扣分只是安全，不代表高分。

比赛目标是净收益。

正确优化方向是：

```text
高收益订单 - 可接受小罚分 - 避免大罚分
```

### 12.4 要有反事实工具

如果没有反事实，只看最终收益很难知道哪里错。

应该记录：

- 被拒订单。
- 拒绝原因。
- 预计罚分。
- 预计净收益。
- 等待原因。
- 如果不等会怎样。

### 12.5 合成测试要随机化，不要模板化

人工合成司机有帮助，但容易自我安慰。

应该随机化：

- 偏好表达方式。
- 惩罚金额。
- 时间窗口。
- 坐标。
- 任务顺序。
- 货源分布。
- 城市名和区域名。

## 13. 当前代码适合学习的阅读顺序

建议按这个顺序看：

1. `agent/model_decision_service.py` 的文件头和 `ModelDecisionService.decide()`。
2. `agent/tools/driver_profile_tool.py` 的 `build_profile_from_preferences()`。
3. `agent/tools/task_calendar_tool.py` 的 `calendar_report()`。
4. `agent/tools/cargo_evaluation_tool.py` 的 `evaluate_candidate()`。
5. `agent/tools/action_preference_guard_tool.py` 的 `evaluate_candidate()`。
6. `agent/tools/commitment_sequence_tool.py` 的 `commitment_report()`。
7. `agent/tools/time_task_progress_tool.py` 的 `progress_report()`。
8. `agent/tools/prompt_templates.py` 看 LLM 输入是什么。

重点不是记住每个 if，而是理解：

```text
文本偏好 -> 结构化约束 -> 候选动作 -> 收益/风险评分 -> 守卫校验 -> 最终动作
```

## 14. 后续如果继续做，最值得投入的方向

优先级从高到低：

1. 重构统一 Constraint schema。
2. 建一个统一 CandidateAction scorer，把 wait/reposition/take_order 都放在一起评分。
3. 加短 horizon lookahead。
4. 加市场热度在线学习。
5. 加反事实日志和拒单分析。
6. 用 LLM 做偏好解析时强制 JSON schema + validator。
7. 减少散落的 raw 正则 guard。
8. 建随机偏好生成器和随机货源扰动测试。
9. 把大文件拆小，降低维护成本。
10. 用更少的手调参数，更多地用统一目标函数。

## 15. 一句话复盘

这套 Agent 最大的问题不是“不够聪明”，而是没有从一开始把问题建模成一个统一的约束优化问题。后面虽然通过 LLM、工具、日历和 guard 修补了很多具体场景，但本质仍然是局部贪心 + 规则补丁，所以能避免很多明显扣分，却很难稳定做到整月收益最大化。

真正的改进方向是：

```text
统一约束模型 + 统一动作评分 + 短期滚动规划 + 市场价值学习 + 反事实诊断
```

这才更接近“连续找货决策”的本质。
