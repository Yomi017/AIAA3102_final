# 通用能力测试指南

## 测试内容

本测试模块评估 Agent 在以下方面的能力：

### 1. RAG 测试（知识检索与推理）
- 知识库检索与整合
- 概念对比与层次关系理解
- 结合检索结果进行计算
- 跨文档信息综合

**测试案例**: `benchmark/RAG/test_cases_RAG.txt`（10个案例）

### 2. Web Search 测试（网络搜索与信息整合）
- 电商比价与配置推荐
- 实时新闻检索与总结
- 政策文件查询
- 航班/天气信息检索
- 安全漏洞信息汇总
- 学术会议信息收集
- 财报数据提炼
- 本地活动信息筛选

**测试案例**: `benchmark/web/test_cases_web.txt`（10个案例）

## 评分机制

使用 **DeepSeek API** 进行自动评分：

- **1.0**: 完美回答，完全满足所有评估要点
- **0.8-0.9**: 优秀回答，满足大部分评估要点
- **0.6-0.7**: 良好回答，满足部分评估要点，有明显遗漏
- **0.4-0.5**: 及格回答，基本理解任务但执行不完整
- **0.2-0.3**: 差回答，严重偏离任务要求
- **0.0-0.1**: 极差回答，完全未理解任务

## 使用方法

### 1. 安装依赖

```bash
pip install openai
```

### 2. 设置 API Key

```bash
export DEEPSEEK_API_KEY='your-deepseek-api-key'
```

获取 API Key: https://platform.deepseek.com/

### 3. 运行测试

**测试所有类型**（默认）:
```bash
python test_general_capability.py
```

**只测试 RAG**:
```bash
python test_general_capability.py --type rag
```

**只测试 Web Search**:
```bash
python test_general_capability.py --type web
```

**限制测试数量**（每类最多5个）:
```bash
python test_general_capability.py --max_cases 5
```

**指定 GPU**:
```bash
python test_general_capability.py --gpu_ids "0,1,2,3"
```

## 结果输出

测试结果保存在 `benchmark_results/general_capability/session_YYYYMMDD_HHMMSS/`：

1. **detailed_results.json**: 每个案例的详细结果
   - 用户输入
   - Agent 完整输出
   - Gemini 评分
   - 评分理由
   - 响应时间

2. **summary.txt**: 测试汇总报告
   - 总体平均分
   - 按任务类型统计
   - 分数分布
   - 各案例概览

## 结果示例

```
============================================================
📊 测试总结
============================================================
总测试案例数: 20
平均分数: 0.75 / 1.00
平均响应时间: 12.3秒

分数分布:
  🌟 优秀 (≥0.8): 8 个
  👍 良好 (0.6-0.8): 6 个
  📖 及格 (0.4-0.6): 4 个
  ⚠️  不及格 (<0.4): 2 个
============================================================
```

## 注意事项

1. **RAG 测试需要知识库**: 
   - 确保已运行 `python rag/build_vector_db.py` 构建知识库
   - 知识库路径: `rag/wiki_vector_db/`

2. **Web Search 测试需要 Google Search API**:
   - 需要在 `config.yaml` 中配置 Google Search 凭证
   - 或设置 `GOOGLE_SEARCH_CREDENTIALS` 环境变量

3. **API 限流**:
   - Gemini API 有速率限制
   - 每个案例测试后会自动暂停 2 秒

4. **成本考虑**:
   - 每个案例会调用一次 Gemini API（用于评分）
   - 建议先用 `--max_cases` 限制测试数量

## 测试案例说明

### RAG 测试重点
- 能否正确检索相关文档
- 能否整合多个文档信息
- 计算能力（结合知识库数据）
- 概念对比与层次理解

### Web Search 测试重点
- 能否使用搜索工具
- 信息来源是否权威
- 数据是否实时准确
- 信息整合是否完整

## 故障排查

### 问题: DeepSeek API 不可用
**解决**: 
1. 检查 API Key 是否正确: `echo $DEEPSEEK_API_KEY`
2. 检查网络连接
3. 可以继续测试但不进行评分

### 问题: RAG 测试失败
**解决**: 
1. 确认知识库存在: `ls rag/wiki_vector_db/`
2. 重新构建: `python rag/build_vector_db.py`

### 问题: Web Search 不工作
**解决**: 
1. 检查 Google Search API 配置
2. 查看 `jsons/goole_search.json` 是否正确
