# ALFworld 消融实验指南

## 实验目的
对比 **Agent + ReAct 框架** 和 **原始 LLM**（无框架）在 ALFworld 任务上的表现。

## 两种测试模式

### 1. Agent 模式（有框架）
- **文件**: `benchmarkTest/ALFworld.py`
- **特点**: 使用完整的 ReAct 框架（Thought → Action → Observation → Final Answer）
- **提示词**: 包含工具描述、ReAct 格式要求
- **执行方式**: Agent 类封装，带有工具调用机制

### 2. Baseline 模式（无框架）
- **文件**: `benchmarkTest/ALFworld_base.py`
- **特点**: 直接使用原始 LLM，不经过 Agent 包装
- **提示词**: 简化的任务描述和命令列表
- **执行方式**: 直接调用 `llm.chat()` 方法

## 使用方法

### 方式 1: 单独运行某个测试

```bash
# 只测试 Agent 框架
python main_ablation.py --mode agent --num_games 10

# 只测试原始 LLM
python main_ablation.py --mode baseline --num_games 10
```

### 方式 2: 连续运行两个测试（推荐）

```bash
# 同时运行两种测试，便于直接对比
python main_ablation.py --mode both --num_games 30
```

### 参数说明
- `--mode`: 测试模式
  - `agent`: 使用 Agent + ReAct 框架
  - `baseline`: 使用原始 LLM（无框架）
  - `both`: 连续运行两种测试
  
- `--num_games`: 测试游戏数量（1-30），不指定则会交互式询问

- `--config`: ALFworld 配置文件路径（默认: `configs/base_config.yaml`）

## 结果分析

### 结果保存位置
```
benchmark_results/
├── agent_session_20251202_143000/       # Agent 框架测试结果
│   ├── detailed_results.csv
│   ├── summary.json
│   └── statistics.txt
└── baseline_session_20251202_150000/    # 基线测试结果
    ├── detailed_results.csv
    ├── summary.json
    └── statistics.txt
```

### 对比指标

1. **任务完成率** (Task Completion Rate)
   - 成功完成任务的游戏数 / 总游戏数

2. **动作成功率** (Action Success Rate)
   - 成功执行的步数 / 总执行步数

3. **平均步数** (Average Steps)
   - 完成任务平均需要的步数

4. **错误类型分布**
   - 超时次数 (timeout_count)
   - 解析错误次数 (parse_error_count)

### 分析方法

```bash
# 查看统计摘要
cat benchmark_results/agent_session_*/statistics.txt
cat benchmark_results/baseline_session_*/statistics.txt

# 用 Excel 打开详细结果对比
# 重点关注：
# - success 列：哪些任务类型成功率不同
# - steps 列：平均步数是否有差异
# - action_success_rate 列：单步执行效率
```

## 预期发现

### Agent 框架可能的优势
- 更结构化的思考过程
- 工具调用机制可能减少格式错误
- ReAct 循环可能提供更清晰的错误反馈

### 原始 LLM 可能的优势
- 更直接的命令生成
- 没有框架开销
- 可能在简单任务上更高效

## 实验建议

1. **样本量**: 至少测试 20-30 个游戏才有统计意义

2. **多次运行**: 因为环境随机性，建议每种模式运行 2-3 次取平均

3. **任务分类**: 从 detailed_results.csv 中按任务类型分组分析

4. **失败案例分析**: 找出两种方法都失败的任务，分析是否有共性

## 故障排查

### 问题: 基线测试无法提取命令
**解决**: 检查 `extract_actions_from_response()` 函数，可能需要调整提取逻辑

### 问题: 两种模式结果完全相同
**检查**: 确认基线测试确实调用的是 `llm.chat()` 而不是 `agent.text()`

### 问题: 超时过多
**调整**: 修改 `time_limit(180)` 中的超时秒数
