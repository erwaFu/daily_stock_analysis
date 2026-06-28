# Agent 应用开发实战学习文档

> 基于 `daily_stock_analysis` 项目的深入剖析

---

## 目录

1. [Agent 架构模式：单 Agent vs 多 Agent](#1-agent-架构模式)
2. [ReAct 推理循环](#2-react-推理循环)
3. [工具系统（Tool Use / Function Calling）](#3-工具系统)
4. [Prompt 工程与结构化输出](#4-prompt-工程与结构化输出)
5. [多 Agent 编排与上下文传递](#5-多-agent-编排与上下文传递)
6. [LLM 适配层与多模型 Fallback](#6-llm-适配层与多模型-fallback)
7. [风控与护栏机制](#7-风控与护栏机制)
8. [技能系统（自然语言扩展）](#8-技能系统)
9. [记忆与校准系统](#9-记忆与校准系统)
10. [超时与降级容错](#10-超时与降级容错)
11. [结构化输出解析与修复](#11-结构化输出解析与修复)
12. [报告生成流水线](#12-报告生成流水线)

---

## 1. Agent 架构模式

### 理论知识

Agent 架构通常分为两种模式：

- **单 Agent 模式**：一个 Agent 负责整个任务流程，通过工具调用完成所有子任务。优点是实现简单，缺点是 prompt 复杂度高、上下文窗口压力大。
- **多 Agent 模式**：多个专业 Agent 各司其职，通过编排器协调执行。每个 Agent 有专属的 prompt、工具子集和输出格式。优点是职责分离、可维护性强，缺点是编排复杂度增加。

选择依据：任务复杂度、模型上下文窗口限制、是否需要专业化分工。

### 项目实践

本项目通过 `AGENT_ARCH` 配置项支持双模式切换：

**工厂模式构建** — `src/agent/factory.py`

```python
def build_agent_executor(config=None, skills=None):
    arch = getattr(config, "agent_arch", "single")
    if arch == "multi":
        return _build_orchestrator(...)  # 多 Agent 模式
    return AgentExecutor(...)            # 单 Agent 模式
```

**单 Agent**：`AgentExecutor` 拥有全部工具和完整 prompt，一人包办四个阶段。

**多 Agent**：`AgentOrchestrator` 管理流水线，每个 Agent 继承 `BaseAgent`，声明自己的工具子集：

| Agent | 工具数 | 职责 |
|---|---|---|
| TechnicalAgent | 8 | 行情、K线、趋势、均线、量能、形态、筹码 |
| IntelAgent | 4 | 新闻搜索、综合情报、股票信息、资金流向 |
| RiskAgent | 3 | 新闻搜索、实时行情、股票信息 |
| DecisionAgent | 0 | 纯合成，无工具 |
| SkillAgent | 动态 | 由 skill 定义的 required_tools 决定 |

**BaseAgent 抽象基类** — `src/agent/agents/base_agent.py`

```python
class BaseAgent(ABC):
    agent_name: str = "base"
    tool_names: Optional[List[str]] = None  # None → 全部工具可用
    max_steps: int = 6

    @abstractmethod
    def system_prompt(self, ctx: AgentContext) -> str: ...

    @abstractmethod
    def build_user_message(self, ctx: AgentContext) -> str: ...

    def post_process(self, ctx, raw_text) -> Optional[AgentOpinion]:
        return None  # 子类可覆写以解析结构化输出
```

**关键设计决策**：`DecisionAgent` 不持有任何工具，只综合前面 Agent 的意见做最终决策。这避免了决策阶段被工具调用干扰，确保其专注"合成"而非"探索"。

---

## 2. ReAct 推理循环

### 理论知识

ReAct（Reasoning + Acting）是当前 Agent 开发最核心的推理范式。其核心循环为：

1. **Reasoning**：LLM 根据当前上下文思考下一步
2. **Acting**：如果需要外部信息，LLM 发起工具调用
3. **Observation**：工具返回结果，追加到上下文
4. **重复**：直到 LLM 认为已有足够信息生成最终答案

CoT（Chain-of-Thought）是一种让 LLM 显式展开推理步骤的 prompting 技术，但原始定义中不包含与外部环境的交互。ReAct 在 CoT 的基础上增加了 Action（工具调用）和 Observation（工具结果）环节，使 Agent 能在推理过程中获取实时数据，弥补 LLM 知识的时效性和准确性不足。简单来说：ReAct = CoT + Tool Use。

### 项目实践

核心实现位于 `src/agent/runner.py` 的 `run_agent_loop()`：

```python
def run_agent_loop(
    *,
    messages: List[Dict[str, Any]],
    tool_registry: ToolRegistry,
    llm_adapter: LLMToolAdapter,
    max_steps: int = 10,
    max_wall_clock_seconds: Optional[float] = None,
    ...
) -> RunLoopResult:

    for step in range(max_steps):
        # 1. 检查超时预算
        remaining_timeout = _remaining_timeout_seconds(...)
        if timeout_exhausted or budget_guard_triggered:
            return _build_timeout_result(...)

        # 2. 调用 LLM
        response = llm_adapter.call_with_tools(messages, tool_decls, timeout=remaining_timeout)

        if response.tool_calls:
            # 3a. 工具调用分支：追加 assistant 消息 → 执行工具 → 追加 tool 结果
            messages.append(assistant_msg)
            tool_results = _execute_tools(response.tool_calls, ...)
            for tr in tool_results:
                messages.append({"role": "tool", "content": tr["result_str"], ...})
        else:
            # 3b. 最终答案分支：LLM 不再调用工具，直接返回文本
            return RunLoopResult(success=True, content=final_content, ...)
```

**循环的核心判断逻辑**：
- 如果 LLM 响应包含 `tool_calls` → Agent 还在"行动"阶段，继续循环
- 如果 LLM 响应只有纯文本 → Agent 认为已收集足够信息，输出最终答案

**CoT/思维链支持** — `src/agent/llm_adapter.py`

```python
# 自动思维模型：不需要额外参数，API 自动返回 reasoning_content
_AUTO_THINKING_MODELS = ["deepseek-reasoner", "deepseek-r1", "qwq"]

# 显式启用模型：需要通过 extra_body 启用
_OPT_IN_THINKING_MODELS = {
    "deepseek-chat": {"thinking": {"type": "enabled"}},
}
```

推理内容在多轮对话中保留回传（`reasoning_content` 和 `provider_blocks`），确保思维链的连续性。

---

## 3. 工具系统

### 理论知识

工具系统是 Agent 与外部世界交互的桥梁。核心设计要素：

1. **工具定义**：名称、描述、参数 Schema（类型、是否必需、枚举值）
2. **工具注册**：集中管理，支持动态增删
3. **Schema 生成**：将内部定义转换为 LLM 能理解的格式（如 OpenAI function calling 格式）
4. **工具执行**：处理 LLM 返回的调用请求，执行并回传结果
5. **安全守卫**：限制 Agent 只能调用授权的工具和参数

### 项目实践

**工具定义数据类** — `src/agent/tools/registry.py`

```python
@dataclass
class ToolParameter:
    name: str
    type: str        # "string" | "number" | "integer" | "boolean" | "array" | "object"
    description: str
    required: bool = True
    enum: Optional[List[str]] = None
    default: Any = None

@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: List[ToolParameter]
    handler: Callable          # 实际执行的 Python 函数
    category: str = "data"    # data | analysis | search | action
```

**注册中心** — 支持注册、查询、OpenAI 格式生成、执行：

```python
class ToolRegistry:
    def register(self, tool_def): ...
    def execute(self, name, **kwargs) -> Any: ...
    def to_openai_tools(self) -> List[dict]: ...
```

**具体工具示例** — `src/agent/tools/data_tools.py`

```python
get_realtime_quote_tool = ToolDefinition(
    name="get_realtime_quote",
    description="Get real-time stock quote including price, change%, volume ratio, "
                "turnover rate, PE, PB, market cap. Returns live market data.",
    parameters=[
        ToolParameter(
            name="stock_code",
            type="string",
            description="Stock code, e.g., '600519' (A-share), 'AAPL' (US), 'hk00700' (HK)",
        ),
    ],
    handler=_handle_get_realtime_quote,
    category="data",
)
```

**工具分类体系**：

| 文件 | 工具类别 | 工具数量 | 示例 |
|---|---|---|---|
| data_tools.py | data | 7 | 实时行情、K线、筹码、分析上下文、股票信息、组合快照、资金流向 |
| analysis_tools.py | analysis | 4 | 趋势分析、均线计算、量能分析、K线形态识别 |
| search_tools.py | search | 2 | 新闻搜索、综合情报搜索 |
| market_tools.py | data | 2 | 大盘指数、行业排名 |
| backtest_tools.py | data | 3 | 技能回测、策略回测、个股回测 |

**安全守卫：Stock Scope** — `src/agent/runner.py`

防止 Agent 在分析某只股票时偷偷查询其他股票：

```python
def _guard_tool_stock_scope(tool_registry, tool_name, arguments, stock_scope):
    # 检查工具是否是股票范围工具（有 stock_code 参数）
    # 如果请求的股票不在允许列表中，返回拒绝结果
    return {
        "error": "stock_scope_violation",
        "expected_stock_code": expected,
        "requested_stock_code": requested,
        "retriable": False,
    }
```

**工具并行执行** — 多个 tool_call 时用 `ThreadPoolExecutor` 并行（最多 5 线程）：

```python
pool = ThreadPoolExecutor(max_workers=min(len(tool_calls), 5))
futures = {pool.submit(contextvars.copy_context().run, _exec_single, tc): tc for tc in tool_calls}
```

**工具结果缓存**：`non_retriable_tool_results` 避免重复调用已明确失败的工具。

---

## 4. Prompt 工程与结构化输出

### 理论知识

Prompt 工程是 Agent 开发中成本最高、最需迭代的环节。关键原则：

1. **角色定义**：明确 Agent 的身份和能力边界
2. **工作流约束**：强制 Agent 按阶段执行，避免跳跃
3. **输出格式约束**：对于需要后续解析的场景，要求 LLM 输出结构化 JSON
4. **分层注入**：将 prompt 拆分为可组合的区段，按需拼接
5. **护栏指令**：明确禁止行为（如"绝不编造数字"）

### 项目实践

**分层 Prompt 组装** — `src/agent/executor.py`

```python
system_prompt = prompt_template.format(
    market_role=market_role,                     # 1. 市场角色（A股/港股/美股专家）
    market_guidelines=market_guidelines,         # 2. 市场准则
    default_skill_policy_section=...,            # 3. 技能基线策略
    skills_section=skills_section,               # 4. 激活的交易技能指令
    language_section=_build_language_section(...), # 5. 输出语言指引
)
```

**四阶段强制工作流**（嵌入 prompt 中）：

```
第一阶段 · 行情与K线（首先执行）
- get_realtime_quote 获取实时行情
- get_daily_history 获取历史K线

第二阶段 · 技术与筹码（等第一阶段结果返回后执行）
...

> ⚠️ 禁止将不同阶段的工具合并到同一次调用中
```

**结构化输出约束** — 完整 JSON Schema 直接嵌入 prompt，包含：
- 7 个顶层字段（stock_name, sentiment_score, decision_type...）
- `dashboard` 嵌套对象（core_conclusion, data_perspective, intelligence, battle_plan, phase_decision, signal_attribution）
- 每个字段的取值范围和格式说明

**可操作性护栏**：

```
- 不得仅因为单日涨跌或评分跨线就在"买入/卖出"之间剧烈切换
- 盘前、非交易日不得伪造今日盘中走势
- 数据存在 stale/fallback/missing 时，confidence_level 不得为高
```

**多 Agent 中的专业 Prompt** — `src/agent/agents/technical_agent.py`

每个 Agent 有独立的 system_prompt，只描述自己的职责和输出格式：

```python
def system_prompt(self, ctx: AgentContext) -> str:
    return f"""\
You are a **Technical Analysis Agent** specialising in Chinese A-shares...
## Output Format
Return **only** a JSON object:
{{
  "signal": "strong_buy|buy|hold|sell|strong_sell",
  "confidence": 0.0-1.0,
  "reasoning": "2-3 sentence summary",
  "key_levels": {{"support": <float>, "resistance": <float>, "stop_loss": <float>}},
  "trend_score": 0-100,
  "ma_alignment": "bullish|neutral|bearish",
  ...
}}
"""
```

**两种响应模式**：
- `dashboard` 模式：输出严格 JSON（用于自动解析和推送）
- `chat` 模式：输出自然语言（用于对话交互）

---

## 5. 多 Agent 编排与上下文传递

### 理论知识

多 Agent 系统的核心挑战是**编排**和**上下文共享**：

1. **编排策略**：顺序执行 vs 并行执行 vs 条件分支
2. **上下文传递**：Agent 间如何共享中间结果
3. **结果聚合**：多个 Agent 的意见如何综合
4. **失败处理**：非关键 Agent 失败时的降级策略

### 项目实践

**共享上下文对象** — `src/agent/protocols.py`

```python
@dataclass
class AgentContext:
    query: str = ""           # 用户查询
    stock_code: str = ""      # 股票代码
    stock_name: str = ""      # 股票名称
    data: Dict[str, Any] = field(default_factory=dict)   # 共享数据
    opinions: List[AgentOpinion] = field(default_factory=list)  # 各Agent意见
    risk_flags: List[Dict] = field(default_factory=list)  # 风控标记
    meta: Dict[str, Any] = field(default_factory=dict)    # 元数据
```

所有 Agent 读写同一个 `AgentContext`，实现数据传递和意见汇聚。

**AgentOpinion 结构化意见** — 每个 Agent 产出一个标准化的意见：

```python
@dataclass
class AgentOpinion:
    agent_name: str = ""     # 产出来源
    signal: str = ""         # buy/hold/sell
    confidence: float = 0.0  # 0.0-1.0
    reasoning: str = ""      # 推理摘要
    key_levels: Dict[str, float] = field(default_factory=dict)
    raw_data: Dict[str, Any] = field(default_factory=dict)  # 任意附加数据
```

**四种编排模式** — `src/agent/orchestrator.py`

```python
def _build_agent_chain(self, ctx):
    technical = TechnicalAgent(**common_kwargs)
    intel = IntelAgent(**common_kwargs)
    risk = RiskAgent(**common_kwargs)
    decision = DecisionAgent(**common_kwargs)

    if self.mode == "quick":     return [technical, decision]
    elif self.mode == "standard": return [technical, intel, decision]
    elif self.mode == "full":    return [technical, intel, risk, decision]
    elif self.mode == "specialist":
        # Skill Agent 在 decision 前动态插入
        return [technical, intel, risk, decision]
```

**上下文数据传递** — 前序 Agent 的数据自动注入后续 Agent 的 prompt：

`src/agent/agents/base_agent.py` 中 `_inject_cached_data()` 会将 `ctx.data` 中已有数据序列化注入：

```python
def _inject_cached_data(self, ctx: AgentContext) -> str:
    parts = []
    for key, value in ctx.data.items():
        if value is not None:
            parts.append(f"[Pre-fetched: {key}]\n{serialised}")
    return "\n\n".join(parts)
```

这意味着 Technical Agent 获取的行情数据，后续的 Intel/Risk/Decision Agent 不需要再次调用工具。

**降级容错** — 非关键阶段失败时流水线继续运行：

```python
if result.status == StageStatus.FAILED:
    non_critical = agent.agent_name in ("intel", "risk") or \
                   agent.agent_name in self._skill_agent_names
    if not non_critical:
        return OrchestratorResult(success=False, error=...)  # 关键阶段失败
    else:
        logger.warning("stage '%s' failed (non-critical, degrading)", ...)
```

---

## 6. LLM 适配层与多模型 Fallback

### 理论知识

生产级 Agent 必须考虑 LLM 的可靠性问题：

1. **多供应商支持**：不同模型有不同的 API 格式和参数
2. **Fallback 链**：主模型失败时自动切换到备选模型
3. **速率限制处理**：遇到 429 错误时退避或切换
4. **上下文窗口溢出**：自动降级到支持更大窗口的模型

### 项目实践

**统一适配层** — `src/agent/llm_adapter.py`

基于 LiteLLM 实现，所有供应商走统一的 `litellm.completion()` 接口：

```python
class LLMToolAdapter:
    def call_with_tools(self, messages, tools, provider=None, timeout=None):
        return self.call_completion(messages, tools=tools, ...)
```

**多模型 Fallback**

```python
def call_completion(self, messages, *, tools=None, ...):
    models_to_try = get_effective_agent_models_to_try(config)

    for idx, model in enumerate(models_to_try):
        try:
            return self._call_litellm_model(messages, tools, model, ...)
        except RateLimitError:
            # 同供应商则退避，跨供应商则直接切换
            if providers[idx] == providers[idx + 1]:
                time.sleep(backoff_sleep)
            continue
        except ContextWindowExceededError:
            continue  # 跳到下一个模型
```

**多 Key 负载均衡** — 使用 `litellm.Router` 实现多 API Key 轮询：

```python
self._router = Router(
    model_list=model_list,
    routing_strategy="simple-shuffle",
    num_retries=2,
)
```

**响应统一解析** — 将不同供应商的响应统一为 `LLMResponse`：

```python
@dataclass
class LLMResponse:
    content: Optional[str] = None           # 文本回答
    tool_calls: List[ToolCall] = ...        # 工具调用请求
    reasoning_content: Optional[str] = None # CoT 推理内容
    provider_blocks: List[Dict] = ...       # 供应商特定块（如 Claude thinking）
    usage: Dict[str, Any] = ...             # Token 使用信息
    provider: str = ""                      # 实际使用的供应商
    model: str = ""                         # 实际使用的模型
```

---

## 7. 风控与护栏机制

### 理论知识

Agent 在金融场景中必须具备风控能力，否则可能给出危险建议。关键设计：

1. **独立风控 Agent**：与决策 Agent 分离，避免利益冲突
2. **信号覆写**：风控可以否决或降级最终决策
3. **分级处理**：根据风险严重程度执行不同强度的干预
4. **幂等性**：风控操作可以安全重复执行

### 项目实践

**风控覆写机制** — `src/agent/orchestrator.py`

```python
def _apply_risk_override(self, ctx: AgentContext):
    # 幂等检查：已执行过则跳过
    if ctx.get_data("risk_override_applied"):
        return

    # 提取 Risk Agent 的信号调整意见
    adjustment = risk_raw.get("signal_adjustment", "").lower()
    has_high_flag = any(flag["severity"] == "high" for flag in ctx.risk_flags)

    # 否决买入
    veto_buy = bool(risk_raw.get("veto_buy")) or adjustment == "veto" or has_high_flag

    # 信号调整
    if veto_buy and current_signal == "buy":
        new_signal = "hold"           # 买入 → 观望
    elif adjustment == "downgrade_one":
        new_signal = _downgrade_signal(current_signal, steps=1)  # 降一级
    elif adjustment == "downgrade_two":
        new_signal = _downgrade_signal(current_signal, steps=2)  # 降两级

    # 修改 dashboard 的所有关联字段
    dashboard["decision_type"] = new_signal
    dashboard["sentiment_score"] = _adjust_sentiment_score(score, new_signal)
    dashboard["analysis_summary"] = f"[风控下调: {current_signal} -> {new_signal}] {summary}"
    core["signal_type"] = "🟡持有观望"  # 更新信号图标
```

**信号降级链**：buy → hold → sell

```python
def _downgrade_signal(signal, steps=1):
    order = ["buy", "hold", "sell"]
    index = order.index(signal)
    return order[min(len(order) - 1, index + steps)]
```

**风控覆写的影响面**：不仅修改 `decision_type`，还同步修改：
- `sentiment_score`（情感分数调整到对应区间）
- `operation_advice`（操作建议措辞）
- `analysis_summary`（添加下调标注）
- `position_advice`（持仓建议改为保守策略）
- `signal_type`（信号图标从绿改黄/红）

---

## 8. 技能系统

### 理论知识

技能系统是 Agent 能力扩展的核心机制，关键设计目标：

1. **零代码扩展**：非开发者也能通过自然语言定义新能力
2. **动态加载**：运行时选择激活哪些技能
3. **技能路由**：根据市场状态/用户意图自动选择合适的技能
4. **意见聚合**：多个技能的意见如何加权综合

### 项目实践

**技能定义** — `src/agent/skills/base.py`

技能是纯自然语言的 YAML 文件，无需写 Python 代码：

```yaml
name: bull_trend
display_name: 多头趋势
description: 识别均线多头排列和趋势确认信号
instructions: |
  1. 检查 MA5 > MA10 > MA20 是否成立
  2. 确认成交量配合趋势方向
  3. ...
category: trend
core_rules: [1, 3]
required_tools: ["get_realtime_quote", "analyze_trend"]
```

**SkillManager** — 管理技能的加载、激活和指令生成：

```python
class SkillManager:
    def load_builtin_skills(self): ...      # 从 strategies/ 加载
    def load_custom_skills(self, directory): ...  # 从自定义目录加载
    def activate(self, skill_names): ...    # 激活指定技能
    def get_skill_instructions(self) -> str: ...  # 生成注入 prompt 的文本
```

**技能路由** — `SkillRouter` 根据市场状态和上下文自动选择技能：

```python
class SkillRouter:
    def select_skills(self, ctx: AgentContext) -> List[str]:
        # 根据 ctx 中的市场阶段、用户请求、已有数据
        # 返回最适合的技能列表
```

**技能意见聚合** — `SkillAggregator` 加权合并多个 SkillAgent 的意见：

```python
def _aggregate_skill_opinions(self, ctx):
    aggregator = SkillAggregator()
    consensus = aggregator.aggregate(ctx)
    # consensus 包含: signal, confidence, reasoning
    ctx.opinions.append(consensus)
    ctx.set_data("skill_consensus", {
        "signal": consensus.signal,
        "confidence": consensus.confidence,
        "reasoning": consensus.reasoning,
    })
```

**动态技能插入** — 在 specialist 模式下，Skill Agent 在 Decision Agent 之前动态插入：

```python
if self.mode == "specialist" and agent.agent_name == "decision" and not specialist_agents_inserted:
    specialist_agents = self._build_specialist_agents(ctx)
    agents[index:index] = specialist_agents  # 插入到 decision 之前
    continue
```

---

## 9. 记忆与校准系统

### 理论知识

Agent 记忆系统解决的核心问题：

1. **上下文延续**：Agent 能记住过去的分析结果
2. **置信度校准**：根据历史准确率调整置信度，避免系统性过度自信/不足
3. **技能权重**：根据回测表现自动调整技能的影响力

### 项目实践

**AgentMemory** — `src/agent/memory.py`

```python
class AgentMemory:
    def get_stock_history(self, stock_code, limit=5) -> List[AnalysisMemoryEntry]:
        """获取该股票的历史分析记录，注入 Agent 上下文"""

    def get_calibration(self, agent_name, stock_code=None) -> CalibrationResult:
        """计算校准因子：如果历史准确率 < 平均置信度 → factor < 1（降权）"""

    def calibrate_confidence(self, agent_name, raw_confidence, stock_code=None) -> float:
        """应用校准：adjusted = raw * calibration_factor"""

    def compute_skill_weights(self, skill_ids) -> Dict[str, float]:
        """基于回测胜率计算技能权重，归一化到均值=1.0"""
```

**校准算法**：

```python
# 如果 Agent 历史准确率 40%，但平均置信度 80%
# factor = 0.4 / 0.8 = 0.5 → 置信度被减半
calibration_factor = historical_accuracy / avg_confidence
calibration_factor = max(0.5, min(1.5, calibration_factor))  # 限制在 [0.5, 1.5]
```

**记忆注入** — `src/agent/agents/base_agent.py`

```python
def _build_memory_context(self, ctx):
    entries = self.memory.get_stock_history(ctx.stock_code, limit=3)
    lines = ["[Memory: recent analysis history]"]
    for entry in entries:
        parts = [entry.date, f"signal={entry.signal}", f"sentiment={entry.sentiment_score}"]
        if entry.was_correct is not None:
            parts.append(f"was_correct={entry.was_correct}")
        lines.append("- " + ", ".join(parts))
    lines.append("Use this memory as context only; do not copy it verbatim.")
```

**校准在执行中应用** — `src/agent/agents/base_agent.py`

```python
def _apply_memory_calibration(self, ctx, opinion, result):
    calibration = self.memory.get_calibration(agent_name=self.agent_name, ...)
    if calibration.calibrated:
        opinion.confidence = max(0.0, min(1.0, raw_confidence * calibration.calibration_factor))
        result.meta["memory_calibration"] = {
            "raw_confidence": raw_confidence,
            "calibrated_confidence": opinion.confidence,
            "factor": calibration.calibration_factor,
        }
```

---

## 10. 超时与降级容错

### 理论知识

生产环境中的 Agent 必须处理：
1. **LLM 调用超时**：模型响应慢或无响应
2. **工具执行超时**：数据源不可用
3. **整体预算耗尽**：多 Agent 流水线耗时过长
4. **非关键组件失败**：部分 Agent 崩溃不应拖垮整体

### 项目实践

**三级超时体系**：

| 级别 | 位置 | 默认值 | 机制 |
|---|---|---|---|
| 流水线级 | orchestrator._execute_pipeline | agent_orchestrator_timeout_s | 整体预算守卫 |
| 步骤级 | runner.run_agent_loop | _MIN_STEP_BUDGET_S = 8s | 单步最小预算 |
| 工具级 | runner._execute_tools | tool_call_timeout_seconds | 单批工具超时 |

**预算守卫** — 在每一步前检查剩余时间：

```python
# 流水线级
_MIN_STAGE_BUDGET_S = 15  # 最少需要 15 秒才能完成一个阶段
if remaining_budget < stage_min_budget_s:
    return self._build_budget_skip_result(...)

# 步骤级
_MIN_STEP_BUDGET_S = 8.0  # 最少需要 8 秒才能完成一次 LLM 调用
if remaining_timeout <= _MIN_STEP_BUDGET_S:
    return _build_budget_guard_result(...)
```

**超时后的降级生成** — 即使超时，也尝试从已完成阶段生成结果：

```python
def _build_timeout_result(self, ...):
    if ctx is not None:
        dashboard, content = self._resolve_final_output(ctx)
        dashboard = self._mark_partial_dashboard(dashboard,
            note="多 Agent 超时，以下结论基于已完成阶段自动降级生成。")
```

**降级标记** — 超时/降级生成的结果会被明确标记：

```python
def _mark_partial_dashboard(dashboard, note):
    summary = tagged.get("analysis_summary")
    tagged["analysis_summary"] = "[降级结果] " + summary
    tagged["risk_warning"] = f"{note} {warning}"
```

---

## 11. 结构化输出解析与修复

### 理论知识

LLM 输出 JSON 的常见问题：
1. JSON 被包裹在 Markdown 代码块中
2. JSON 末尾缺少闭合括号
3. 字符串值中包含未转义的引号
4. JSON 前后有多余的说明文字

健壮的解析策略应该逐级尝试，从最严格到最宽容。

### 项目实践

**4 层解析策略** — `src/agent/runner.py`

```python
def parse_dashboard_json(content: str) -> Optional[Dict]:
    # 策略 1：Markdown 代码块 ```json ... ```
    json_blocks = re.findall(r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL)
    for block in json_blocks:
        if _try_parse_json(block): return parsed

    # 策略 2：直接解析全文
    parsed = _try_parse_json(content)
    if parsed: return parsed

    # 策略 3：json_repair 修复全文
    parsed = _try_repair_json(content, repair_json)
    if parsed: return parsed

    # 策略 4：花括号定界提取
    brace_start = content.find("{")
    brace_end = content.rfind("}")
    candidate = content[brace_start:brace_end + 1]
    parsed = _try_parse_json(candidate) or _try_repair_json(candidate, repair_json)
    return parsed
```

**Orchestrator 级别的 Dashboard 标准化** — 即使 LLM 输出不完整，也会补齐缺失字段：

```python
def _normalize_dashboard_payload(self, payload, ctx):
    # 缺失字段用默认值填充
    payload.setdefault("decision_type", "hold")
    payload.setdefault("sentiment_score", 50)
    # 从各 Agent 意见中提取关键位
    key_levels = self._collect_key_levels(ctx, payload, dashboard_block)
    risk_alerts = self._collect_risk_alerts(ctx, intelligence)
    # 填充 sniper_points（精确买入/止损/止盈价位）
    sniper["ideal_buy"] = ideal_buy if ideal_buy is not None else "N/A"
```

---

## 12. 报告生成流水线

### 理论知识

Agent 的最终产出需要经过多步后处理才能变为用户可见的报告：

1. **原始输出** → LLM 生成的文本/JSON
2. **解析修复** → 结构化 Dashboard 对象
3. **标准化填充** → 补齐缺失字段
4. **风控覆写** → 根据风险信号调整
5. **模板渲染** → 按目标平台格式输出
6. **通知推送** → 发送到微信/飞书/Telegram 等

### 项目实践

**完整流水线**：

```
Agent(Dashboard JSON)
    ↓
parse_dashboard_json()        # 解析与修复
    ↓
_normalize_dashboard_payload() # 标准化填充
    ↓
_apply_risk_override()         # 风控覆写
    ↓
report_renderer.render()       # Jinja2 模板渲染
    ↓
notify()                       # 多渠道推送
```

**渲染引擎** — 使用 Jinja2 模板支持多平台格式：

| 格式 | 用途 |
|---|---|
| markdown | Web/API 展示 |
| wechat | 微信公众号推送 |
| brief | 简报摘要 |
| telegram | Telegram 机器人 |

---

## 总结：架构特征速查表

| 关键点 | 理论概念 | 项目实践 |
|---|---|---|
| 架构模式 | 单 Agent vs 多 Agent | `AGENT_ARCH` 配置切换，工厂模式构建 |
| 推理循环 | ReAct | `run_agent_loop()` 核心循环 |
| 工具系统 | Function Calling | `ToolRegistry` + `ToolDefinition` + `@tool` 装饰器 |
| Prompt 工程 | 分层注入 + 结构化输出 | 5 层模板组装 + 完整 JSON Schema 嵌入 |
| 多 Agent 编排 | 顺序 + 条件分支 | 4 种编排模式 + 共享 AgentContext |
| LLM 适配 | 多供应商 + Fallback | LiteLLM + Router + 多 Key 负载均衡 |
| 风控护栏 | 信号覆写 + 降级 | `_apply_risk_override()` 幂等覆写 |
| 技能系统 | 零代码扩展 | YAML/SKILL.md + SkillRouter + SkillAggregator |
| 记忆校准 | 置信度校准 + 权重 | AgentMemory + calibration_factor |
| 超时降级 | 三级超时 + 降级生成 | 流水线/步骤/工具三级 + 降级标记 |
| 输出解析 | 多策略解析修复 | 4 层解析 + json_repair + Dashboard 标准化 |
| 报告生成 | 模板渲染 + 多渠道 | Jinja2 + markdown/wechat/brief/telegram |

**核心文件索引**：

| 文件 | 核心职责 |
|---|---|
| `src/agent/runner.py` | ReAct 循环引擎 + 工具执行 + JSON 解析 |
| `src/agent/tools/registry.py` | 工具注册中心 + OpenAI schema 生成 |
| `src/agent/orchestrator.py` | 多 Agent 编排 + 风控覆写 + Dashboard 标准化 |
| `src/agent/executor.py` | 单 Agent 执行器 + Prompt 模板 |
| `src/agent/factory.py` | 工厂模式 + 技能解析 + 缓存管理 |
| `src/agent/agents/base_agent.py` | Agent 抽象基类 + 记忆注入 + 工具过滤 |
| `src/agent/protocols.py` | 共享数据结构（AgentContext, AgentOpinion, StageResult） |
| `src/agent/llm_adapter.py` | LLM 适配层 + 多模型 Fallback + 响应解析 |
| `src/agent/memory.py` | 记忆系统 + 置信度校准 + 技能权重 |
| `src/agent/skills/base.py` | 技能定义 + YAML 加载 + SkillManager |

**推荐阅读顺序**：`protocols.py` → `registry.py` → `runner.py` → `base_agent.py` → `technical_agent.py` → `orchestrator.py` → `executor.py` → `factory.py` → `llm_adapter.py` → `memory.py`
