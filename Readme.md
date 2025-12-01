# AIAA3102_final

本仓库为一个轻量级的 Agent 系统工程样例，集成了本地 LLM（Qwen3-8B）、ReAct 风格的工具调用框架、数学计算工具、以及基于向量检索的 RAG 知识库模块。下文为完整说明（中文）。

**目录**
- `agent.py`：Agent 主逻辑，负责构建 system prompt、解析模型输出中的工具调用（Action / Action Input），按步调用工具并收集观察结果；实现了 ReAct 风格的交互控制与格式校验。 
- `llm.py`：LLM 封装层（`BaseLLM` 与 `Qwen3`），支持优先使用 `vllm`（若可用）或回退到 `transformers`，负责加载模型、tokenizer，并封装 `chat` 接口。
- `main.py`：交互式终端启动器，负责加载模型、初始化 `Agent`、选择是否启用 RAG 知识库并驱动对话循环（包含简单的控制台输出格式）。
- `tool.py`：工具集合（`Tools`），包含网络查询（Google Search wrapper）、天气/时间查询、各种数学计算器（基于 `calculator.py`）、以及知识库问答入口（依赖 RAG 模块）。
- `calculator.py`：数学工具集，实现常用算术、三角函数、矩阵运算、定积分以及通用表达式求值接口。
- `testapi.py`：用于测试 `query_time` 接口连通性与返回格式的脚本。
- `rag/`：RAG（检索增强生成）相关模块
	- `rag/document_processor.py`：文档加载、清洗与分块（滑动窗口策略，默认 chunk=400、overlap=80）
	- `rag/vector_store.py`：基于 `sentence-transformers` 进行文本嵌入，并使用 `faiss` 构建检索索引，支持构建/保存/加载索引与检索接口。
	- `rag/rag_engine.py`：RAG 引擎，封装检索与上下文格式化，供 `Tools` 的 `knowledge_base_query` 调用。
	- `rag/build_vector_db.py`：从文本文件夹构建向量数据库的命令行工具。
- `rag/wiki_vector_db/`：仓库默认的 RAG 向量库目录（可能为空或已预构建数据）。
- `wiki_docs/`：示例或待构建为向量库的文档集合（中英文主题文件若干）。

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
1. 使用 pyenv（或你偏好的 Python 管理工具）创建并激活 Python 环境，推荐 `3.11.6`：
```bash
pyenv install 3.11.6
pyenv virtualenv 3.11.6 AIAA3102_final
pyenv local AIAA3102_final
python -m pip install --upgrade pip
pip install -r requirements.txt
```

2. 下载 Qwen3-8B（仓库根 `Readme.md` 原说明）：
```bash
pip install modelscope
modelscope download --model Qwen/Qwen3-8B
```
安装 `transformers` 后如遇性能或兼容问题，可参考提示安装 `accelerate`：
```bash
pip install accelerate
```

3. （可选）构建知识库：
```bash
python rag/build_vector_db.py -i wiki_docs -o rag/wiki_vector_db
```

4. 启动交互式 Agent：
```bash
python main.py
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

## Google 搜索配置
- 在 Programmable Search Engine 控制台创建搜索引擎，记录 `cx`，并在 Google Cloud Console 申请一个 API Key。
- 将上述两个值写入 `jsons/goole_search.json`（或其他自定义路径）中，例如：
	```json
	{
		"search_engine_id": "your_cx",
		"api_key": "your_api_key"
	}
	```
- 若凭据文件不在默认位置，可在 `config.yaml` 的 `google.credentials_path` 指定路径，或在运行前设置环境变量 `GOOGLE_SEARCH_CREDENTIALS=/path/to/your.json`。
- 工具会在运行时读取 JSON 中的配置并调用官方 Custom Search JSON API，无需再配置额外的 OAuth/Service Account。

## Project B 进度清单（Tool-Using LLM Agent with ReAct Pattern）
- [x] 实现 Thought→Action→Observation→Final Answer 的最小 ReAct 循环。
- [x] 解析结构化工具调用（JSON 参数）并在 `agent.py` 中路由执行。
- [x] 集成 ≥2 项工具（搜索/RAG、天气时间、计算器等）供模型调用。
- [x] 在 `main.py` 提供可运行 CLI 入口，便于一键启动代理。
- [x] 记录每轮 Thought/Action/Observation 日志，便于调试追踪。
- [x] 设置 `max_step` 以防无限循环，并对工具调用做好异常捕获。
- [ ] 为每个工具添加调用超时/失败重试等进一步的防护。
- [ ] 构建 20–30 条含“黄金答案”的评估查询集。
- [ ] 编写自动评分脚本以验证代理输出与黄金答案的偏差。
- [ ] 实施自洽性/多轨迹投票或类似策略提升可靠性（高级方向之一）。
- [ ] 基于检索置信度/引用的幻觉控制（高级方向之二）。
- [x] 基于启发式模式实现一个提示词注入检测器，并演示攻击和你的缓解措施（高级方向之三）。
- [ ] 撰写 6–8 页技术报告，覆盖设计、实验与失效分析。
- [ ] 录制 ≤3 分钟的项目演示视频。
- [ ] 准备第13周最终展示所需的幻灯片/讲稿。
