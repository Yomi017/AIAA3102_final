# AIAA3102_final

本仓库为一个功能完整的 ReAct Agent 系统，集成了本地/远程 LLM、工具调用框架、RA2) LLM 封装（`llm.py`）
- 封装了 `BaseLLM` 抽象基类，支持多种模型实现：
  - **`Qwen3VL`**：视觉-语言多模态模型，支持图像+文本输入（推荐用于基准测试）
  - **`Qwen3`**：纯文本模型，轻量级选项
- 所有模型在初始化时会尝试导入 `vllm`：若可用则使用 `vllm.LLM`（支持更好的并行和推理性能），否则回退到 `transformers` 的 `AutoModelForCausalLM`。
- `chat` 方法负责将历史、system 指令、用户输入拼接为模型输入（使用 tokenizer 的聊天模板），并对生成的输出做后处理。
- **模型选择建议**：
  - 交互式使用：`Qwen3VL`（功能完整，支持多模态）
  - 基准测试：`Qwen3VL`（与测试脚本配置一致）
  - 仅文本任务：`Qwen3`（性能更快）库、网络搜索、数学计算器等能力，并提供完整的基准测试框架。

## 项目简介

基于 ReAct (Reasoning and Acting) 范式实现的智能代理系统，支持：
- 🤖 **多模型支持**：Qwen3VL、Qwen3-8B 等本地模型，通过 vLLM 或 transformers 加载
- 🛠️ **丰富工具集**：Google/Tavily 搜索、知识库检索、数学计算、时间查询、网页爬取
- 🔐 **安全防护**：提示注入检测、输入过滤、会话锁定机制
- 📊 **完整评测**：RAG、Web、ALFworld 等多场景基准测试
- ⚙️ **配置驱动**：通过 `config.yaml` 管理所有参数，支持一键运行

---

## 目录结构

```
AIAA3102_final/
├── agent.py                    # Agent 主逻辑（ReAct 循环、工具调用解析）
├── base.py                     # BaselineAgent（无工具版本，用于消融实验）
├── llm.py                      # LLM 封装（vLLM/transformers）
├── tool.py                     # 工具管理器（所有工具的实现与注册）
├── calculator.py               # 数学计算工具集
├── main.py                     # 交互式终端入口
├── config.yaml                 # 全局配置文件
├── requirements.txt            # Python 依赖
│
├── rag/                        # RAG（检索增强生成）模块
│   ├── document_processor.py  # 文档加载与分块
│   ├── vector_store.py        # FAISS 向量存储
│   ├── rag_engine.py          # RAG 检索引擎
│   └── build_vector_db.py     # 向量库构建工具
│
├── security/                   # 安全模块
│   └── prompt_guard.py        # 提示注入检测器
│
├── testsh/selfbenchmark/       # 基准测试框架
│   ├── test_rag_capability.py        # RAG 测试（Agent 版本）
│   ├── test_baseline_capability.py   # Baseline 测试（无工具）
│   ├── run_agent_test.sh             # Agent 测试一键脚本
│   ├── run_baseline_test.sh          # Baseline 测试一键脚本
│   └── benchmarkTest/                # 测试工具与数据
│       ├── general_capability.py     # 通用能力测试器（F1/DeepSeek 评分）
│       ├── ALFworld.py               # ALFworld 环境测试
│       └── result_logger.py          # 结果记录器
│
├── benchmark/                  # 测试数据集
│   ├── RAG/                    # 知识库问答测试集
│   │   ├── test_cases_RAG_AI.txt
│   │   └── test_cases_RoboMaster_RAG.txt
│   └── web/                    # 网络搜索测试集
│       └── test_cases_web.txt
│
└── wiki_docs/                  # 知识库文档（用于构建向量数据库）
    ├── en_Artificial intelligence.txt
    ├── en_Deep learning.txt
    └── ...
```

## 设计要点与实现细节

1) Agent 与 ReAct 模式
- `agent.py` 构造了一个严格的 prompt 模板（`REACT_PROMPT`），并对模型输出做格式校验（要求出现 `Thought:`, `Action:`, `Action Input:`, `Final Answer:` 等字段）。
- `parse_latest_plugin_call` 会从模型输出中提取最近一次的工具调用并返回工具名与 JSON 参数，若存在多个 `Action:` 会发出警告并只取第一个。
- `call_plugin` 将解析好的调用路由到 `Tools` 提供的实际实现，工具的输出会被追加到下一轮模型的观察（Observation）中。

2) LLM 封装（`llm.py`）
- 封装了 `BaseLLM` 抽象与 `Qwen3` 实现。
- `Qwen3` 在初始化时会尝试导入 `vllm`：若可用则使用 `vllm.LLM`（支持更好的并行和推理性能），否则回退到 `transformers` 的 `AutoModelForCausalLM`。
- `chat` 方法负责将历史、system 指令、用户输入拼接为模型输入（使用 tokenizer 的聊天模板），并对生成的输出做后处理（在非 vLLM 情况下使用特殊 token 截断“思考”部分）。

3) Tools 与 Calculator
- `tool.py` 将所有外部能力以工具（tool）形式注册，`toolConfig` 列表定义了提供给模型的工具描述（用于让模型选择应调用哪个工具并传参）。
- `Calculator`（`calculator.py`）实现了数学运算（标量、三角、矩阵、定积分与表达式评估），并被 `Tools` 的多个工具方法复用。
- 一些网络工具（如 `google_search`、`query_time`）依赖外部 HTTP API；`testapi.py` 提供了对 `query_time` API 的连通性与返回格式测试脚本。

4) RAG（检索增强生成）
- 文档被 `DocumentProcessor` 加载、清洗并分块，然后由 `VectorStore` 生成嵌入并用 FAISS 建索引。
- `build_vector_db.py` 提供从 `wiki_docs/` 等文件夹批量生成向量库的 CLI，用法：
	```bash
	python rag/build_vector_db.py -i wiki_docs -o rag/wiki_vector_db
	```
- 运行时，主程序 `main.py` 会优先检查 `rag/wiki_vector_db`，若存在则把路径传给 `Agent`（通过 `Tools` 的 `RAGEngine`），并允许模型使用 `knowledge_base_query` 工具获取检索上下文。

## 依赖与环境
- 推荐使用 `pyenv` 管理 Python 版本（仓库已包含 `.python-version`，建议版本 `3.11.6`）。有关 `pyenv` 的快速说明参考仓库中的 `README_pyenv.md`。
- 代码中检测到的主要第三方包（已写入 `requirements.txt` 占位）：
	- `numpy`, `scipy`, `requests`, `PyYAML` (`yaml`), `transformers`, `vllm`, `json5`, `faiss-cpu`（或 `faiss-cuda`），`sentence-transformers`, `langchain_google_community`, `modelscope`（若使用 modelscope 下载模型）
- 注意：`faiss` 的安装依赖于平台与是否使用 GPU，请根据硬件选择 `faiss-cpu` 或 `faiss-cuda`。

## 快速上手（本地交互）

### 1. 环境准备
使用 pyenv（或你偏好的 Python 管理工具）创建并激活 Python 环境，推荐 `3.11.6`：
```bash
pyenv install 3.11.6
pyenv virtualenv 3.11.6 AIAA3102_final
pyenv local AIAA3102_final
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. 模型下载

#### 选项 A：Qwen3VL（推荐，支持多模态）
```bash
# 从 HuggingFace 下载（需要科学上网或配置镜像）
pip install huggingface_hub
huggingface-cli download Qwen/Qwen2-VL-7B-Instruct --local-dir Qwen2-VL-7B-Instruct

# 或使用 ModelScope（国内镜像）
pip install modelscope
modelscope download --model Qwen/Qwen2-VL-7B-Instruct
```

**GPU 要求**：Qwen2-VL-7B-Instruct 需要约 16GB 显存（单卡），或使用多卡并行：
```python
# 在 main.py 或测试脚本中指定 GPU
llm = Qwen3VL(
    model_path="Qwen/Qwen2-VL-7B-Instruct",
    gpu_ids=[4, 5, 6, 7],  # 使用多卡
    max_tokens=2048
)
```

#### 选项 B：Qwen3-8B（仅文本，轻量级）
```bash
pip install modelscope
modelscope download --model Qwen/Qwen3-8B
```

**注意**：本项目的基准测试脚本（`testsh/selfbenchmark/`）默认使用 **Qwen3VL**，如果使用其他模型需要修改相应脚本。

### 3. 配置文件设置（`config.yaml`）
```yaml
# 运行模式：0=基准测试，1=交互式对话
STATE_OF_MODEL: 1

# 安全配置
security:
  enable_prompt_guard: true      # 启用提示注入检测
  lock_on_violation: true        # 检测到攻击时锁定会话

# Google 搜索配置（可选）
google:
  credentials_path: jsons/goole_search.json

# DeepSeek API（用于 Web 测试评分，可选）
DEEPSEEK_API: ""
```

### 4. 构建知识库（用于 RAG 功能）
```bash
python rag/build_vector_db.py -i wiki_docs -o rag/wiki_vector_db
```

### 5. 启动交互式 Agent
```bash
python main.py
```

**交互示例**：
```
用户：今天天气怎么样？
Agent：[调用 get_time 获取时间] → [调用 google_search 搜索天气] → 返回天气信息

用户：帮我计算 (123 + 456) * 2
Agent：[调用 calculate 工具] → 返回结果 1158

用户：什么是深度学习？
Agent：[调用 knowledge_base_query 检索知识库] → 返回定义和解释
```

运行期间如果提示注入防护触发并锁定会话，可直接在 CLI 输入 `reset guard`（或 `/reset_guard`）清空锁定状态，然后继续提问。

在出现错误时，请关注终端输出的异常提示，常见问题：模型路径错误、`faiss` 安装异常、缺少 GPU 驱动/内存不足等。

## 开发者指南（核心函数说明）
- `Agent.text(question, history)`：主对话步进函数，会循环调用模型生成、解析工具调用、执行工具并把 observation 拼回 prompt，直到无工具调用或达到步数上限。
- `Tools` 的各方法：封装外部能力并打印调用日志，返回供模型消费的字符串观察结果。
- `DocumentProcessor._chunk_text`：滑动窗口 + 句子边界切分策略，适配中文标点优先切分。
- `VectorStore.create_embeddings`：使用 `SentenceTransformer` 生成嵌入并归一化，便于在 FAISS 中使用内积近似余弦相似度检索。

## 安全与提示注入防护
- `security/prompt_guard.py` 汇总了四类高危提示（Prompt Injection / Insecure Output Handling / Insecure Plugin or Tool Usage / Excessive Agency）的启发式检测规则，并统一返回分类、严重级别与修复建议。
- 在 `config.yaml` 的 `security.enable_prompt_guard`/`lock_on_violation` 控制开关和锁定策略；默认启用并在高危时直接锁死当前终端会话，所有后续输入都会被拒绝。
- CLI 支持管理员手动解锁：在主程序运行过程中输入 `reset guard`（或 `/reset_guard`）即可调用 `Agent.reset_security()`，日志中会记录此次解锁操作。
- 仓库附带 `tests/test_prompt_guard.py` 与 `tests/test_agent_prompt_guard.py`，分别覆盖检测规则与锁定/解锁的集成路径，可作为演示攻击与缓解流程的最小案例。

## 测试
- `python testapi.py`：运行查询时间 API 的一组测试用例，检查网络连通性与响应格式。
- `pytest tests/test_prompt_guard.py tests/test_agent_prompt_guard.py`：验证提示注入检测器针对四类攻击样例、安全提示、会话锁定与重置流程的行为。

## 常见问题与排查建议
- 如果 `Qwen3` 模型加载失败：确认模型已下载到指定路径，或 `transformers` / `vllm` 是否正确安装并支持当前模型。
- 如果出现 `faiss` 安装或导入失败：优先尝试 `pip install faiss-cpu`（CPU 环境），或参照 FAISS 官方文档安装适配 CUDA 的包。
- 如果 RAG 检索返回空或不准确：确认 `rag/wiki_vector_db` 已正确构建并包含 `faiss.index` 与 `documents.pkl`。

---

## 工具规范文档

### 已实现工具列表

| 工具名称 | 功能描述 | 必需参数 | 超时时间 | 依赖配置 |
|---------|---------|---------|---------|---------|
| **google_search** | Google 网络搜索 | `search_query` (str) | 15s | Google API 密钥 |
| **tavily_search** | Tavily AI 搜索 | `search_query` (str) | 15s | Tavily API 密钥 |
| **knowledge_base_query** | RAG 知识库检索 | `question` (str) | 10s | 向量数据库 |
| **get_time** | 获取当前时间 | 无 | 5s | 无 |
| **calculate** | 基础数学运算 | `expression` (str) | 5s | 无 |
| **trigonometric** | 三角函数计算 | `function`, `angle`, `unit` | 5s | 无 |
| **matrix_operations** | 矩阵运算 | `operation`, `matrix_a`, `matrix_b` | 5s | 无 |
| **definite_integral** | 定积分计算 | `expression`, `lower`, `upper` | 10s | 无 |
| **evaluate_expression** | 通用表达式求值 | `expression` | 5s | 无 |
| **web_crawler** | 网页内容抓取 | `url` | 20s | BeautifulSoup4 |

---

### 工具详细说明

#### 1. google_search - Google 搜索
**功能**：访问互联网，查询百科知识、时事新闻等。

**参数**：
```json
{
  "search_query": "搜索关键词或短语"
}
```

**返回格式**：
```
1. [标题]
[摘要]
[链接]

2. [标题]
[摘要]
[链接]
```

**配置步骤**：
1. 访问 [Google Cloud Console](https://console.cloud.google.com/)，创建项目并启用 **Custom Search JSON API**
2. 创建 API 密钥（凭据 → API 密钥）
3. 在 [Programmable Search Engine](https://programmablesearchengine.google.com/) 创建搜索引擎，获取 `cx`
4. 创建配置文件 `jsons/goole_search.json`：
```json
{
  "api_key": "your_google_api_key",
  "search_engine_id": "your_cx"
}
```

**使用示例**：
```python
Action: google_search
Action Input: {"search_query": "2024年人工智能最新进展"}
```

---

#### 2. tavily_search - Tavily AI 搜索
**功能**：基于 AI 的智能搜索，提供更结构化的结果。

**配置步骤**：
```bash
# 设置环境变量
export TAVILY_API_KEY="tvly-your-api-key"

# 或在 Python 代码中设置
import os
os.environ["TAVILY_API_KEY"] = "tvly-your-api-key"
```

**申请 API Key**：访问 [Tavily.com](https://tavily.com/) 注册并获取

---

#### 3. knowledge_base_query - 知识库检索
**功能**：从本地向量数据库检索相关文档片段（基于 FAISS + SentenceTransformers）。

**参数**：
```json
{
  "question": "要查询的问题"
}
```

**配置步骤**：
1. 准备文档：将 `.txt` 文件放入 `wiki_docs/` 目录
2. 构建向量库：
```bash
python rag/build_vector_db.py -i wiki_docs -o rag/wiki_vector_db
```
3. 在代码中指定路径：
```python
tools = ToolsManager(enable_rag=True, rag_db_path="rag/wiki_vector_db")
```

**工作原理**：
- 文档分块（400 字符，80 字符重叠）
- SentenceTransformer 生成嵌入（默认模型：`paraphrase-multilingual-MiniLM-L12-v2`）
- FAISS 索引检索（Top-5 相似片段）
- 返回拼接的上下文

---

#### 4. calculate - 基础数学运算
**功能**：执行加减乘除、幂运算等基本数学计算。

**参数**：
```json
{
  "expression": "数学表达式"
}
```

**支持运算符**：`+`, `-`, `*`, `/`, `**`, `()`, `sqrt()`, `abs()`

**使用示例**：
```python
Action: calculate
Action Input: {"expression": "(123 + 456) * 2 / 3"}
# 返回：386.0
```

---

#### 5. trigonometric - 三角函数
**参数**：
```json
{
  "function": "sin|cos|tan|arcsin|arccos|arctan",
  "angle": "数值",
  "unit": "deg|rad"
}
```

**使用示例**：
```python
Action: trigonometric
Action Input: {"function": "sin", "angle": "30", "unit": "deg"}
# 返回：0.5
```

---

#### 6. matrix_operations - 矩阵运算
**参数**：
```json
{
  "operation": "add|multiply|transpose|inverse|determinant",
  "matrix_a": "[[1,2],[3,4]]",
  "matrix_b": "[[5,6],[7,8]]"
}
```

**使用示例**：
```python
Action: matrix_operations
Action Input: {
  "operation": "multiply",
  "matrix_a": "[[1,2],[3,4]]",
  "matrix_b": "[[2,0],[1,2]]"
}
# 返回：[[4, 4], [10, 8]]
```

---

#### 7. web_crawler - 网页爬取
**功能**：抓取指定 URL 的网页文本内容（去除 HTML 标签）。

**依赖安装**：
```bash
pip install beautifulsoup4
```

**使用示例**：
```python
Action: web_crawler
Action Input: {"url": "https://example.com/article"}
```

---

### 自定义工具开发指南

#### 步骤 1：继承 ToolBase 基类
```python
from tool import ToolBase
from typing import List, Dict, Any

class MyCustomTool(ToolBase):
    TIMEOUT = 15  # 超时时间（秒）
    
    @property
    def name_for_human(self) -> str:
        return "My Custom Tool"
    
    @property
    def name_for_model(self) -> str:
        return "my_custom_tool"  # 模型调用时使用的名称
    
    @property
    def description_for_model(self) -> str:
        return "This tool does something useful for the model."
    
    @property
    def parameters(self) -> List[Dict[str, Any]]:
        return [
            {
                'name': 'param1',
                'description': 'First parameter description',
                'required': True,
                'schema': {'type': 'string'}
            }
        ]
    
    def execute(self, param1: str) -> str:
        # 实现工具逻辑
        result = f"Processed: {param1}"
        return result
```

#### 步骤 2：注册到 ToolsManager
在 `tool.py` 的 `ToolsManager` 类中添加：
```python
def __init__(self, ...):
    # 其他工具初始化
    self.my_tool = MyCustomTool()
    self.tools['my_custom_tool'] = self.my_tool

def get_all_tool_configs(self) -> List[Dict[str, Any]]:
    configs = []
    if hasattr(self, 'my_tool'):
        configs.append(self.my_tool.get_config())
    # ... 其他工具
    return configs
```

---

## 基准测试框架

### 测试架构

本项目提供完整的 Agent 评测框架，支持三种测试场景：

1. **RAG 测试**：知识库问答能力（AI知识库、RoboMaster知识库）
2. **Web 测试**：网络搜索与信息整合能力
3. **ALFworld 测试**：具身智能环境交互能力

每种场景提供两个版本：
- **Agent 版本**：完整 ReAct 框架 + 工具调用
- **Baseline 版本**：纯 LLM，无工具，用于消融实验

---

### 快速运行测试（一键脚本）

#### RAG 测试
```bash
# Agent 版本（完整工具）
cd testsh/selfbenchmark
./run_agent_test.sh ai           # 测试 AI 知识库
./run_agent_test.sh robomaster   # 测试 RoboMaster 知识库

# Baseline 版本（无工具）
./run_baseline_test.sh ai
./run_baseline_test.sh robomaster
```

#### Web 搜索测试
```bash
./run_agent_test.sh web
./run_baseline_test.sh web
```

#### 测试所有数据集
```bash
./run_agent_test.sh all
./run_baseline_test.sh all
```

---

### 测试结果

**输出目录结构**：
```
benchmark_results/
├── agent/
│   ├── ai/
│   │   ├── results_20241207_143052.json       # 详细结果
│   │   ├── summary_20241207_143052.txt        # 文本摘要
│   │   └── scores_20241207_143052.csv         # 评分表格
│   ├── robomaster/
│   └── web/
└── baseline/
    ├── ai/
    ├── robomaster/
    └── web/
```

**评分方法**：
- **RAG 测试**（有标准答案）：F1 Score（基于 3-gram 字符匹配）
- **Web 测试**（无标准答案）：DeepSeek API 评分（0-10分制）

---

### 手动运行测试

#### Agent 测试
```bash
cd testsh/selfbenchmark
python test_rag_capability.py --dataset ai --gpu 4,5,6,7
```

#### Baseline 测试
```bash
python test_baseline_capability.py --dataset robomaster --gpu 4,5,6,7
```

#### 比较结果
```bash
python compare_results.py \
  --agent benchmark_results/agent/ai/results_xxx.json \
  --baseline benchmark_results/baseline/ai/results_xxx.json
```

---

### 配置测试参数

在测试脚本中修改：
```python
# GPU 配置
GPU_IDS = [4, 5, 6, 7]

# 模型路径（确保使用 Qwen3VL）
MODEL_PATH = "Qwen/Qwen2-VL-7B-Instruct"

# 生成参数
MAX_TOKENS = 2048
TEMPERATURE = 0.7

# 测试数据路径
TEST_FILE = "benchmark/RAG/test_cases_RAG_AI.txt"
```

---

### 测试数据格式

**RAG 测试**（`benchmark/RAG/test_cases_RAG_*.txt`）：
```
问题: 什么是深度学习？
标准答案: 深度学习是机器学习的一个分支，使用多层神经网络...
---
问题: 反向传播算法的原理是什么？
标准答案: 反向传播通过链式法则计算梯度...
---
```

**Web 测试**（`benchmark/web/test_cases_web.txt`）：
```
问题: 2024年诺贝尔物理学奖得主是谁？
评分要点: 1. 准确性：是否包含正确的得主姓名 2. 完整性：是否说明获奖理由 3. 时效性：信息是否为2024年最新
---
```

---

## 开发者指南（核心 API）

### Agent 类
```python
from agent import Agent
from llm import Qwen3VL
from tool import ToolsManager

# 初始化 LLM
llm = Qwen3VL(model_path="Qwen/Qwen2-VL-7B-Instruct", gpu_ids=[0,1])

# 初始化工具管理器
tools = ToolsManager(
    enable_google=True,
    enable_tavily=False,
    enable_rag=True,
    rag_db_path="rag/wiki_vector_db"
)

# 创建 Agent
agent = Agent(
    llm=llm,
    tools=tools,
    max_step=10,           # 最大推理步数
    allowed_tools=None     # None=所有工具，或指定列表限制工具
)

# 对话
response = agent.text(
    question="什么是深度学习？",
    history=[]             # 对话历史
)
print(response)
```

### BaselineAgent 类（无工具版本）
```python
from base import BaselineAgent

baseline = BaselineAgent(llm=llm)
response = baseline.text(
    text="什么是深度学习？",
    history=[],
    images=None
)
```

### 工具过滤（消融实验）
```python
# 仅允许使用知识库查询工具
agent = Agent(
    llm=llm,
    tools=tools,
    allowed_tools=["knowledge_base_query"]
)

# 仅允许搜索工具
agent = Agent(
    llm=llm,
    tools=tools,
    allowed_tools=["google_search", "tavily_search", "get_time"]
)
```
