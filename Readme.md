# AIAA3102_final

一个基于 Qwen3 系列大模型的轻量级多模态 ReAct Agent 系统，支持：

- 终端 CLI 交互（支持图片输入、多轮工具调用、RAG 检索）
- FastAPI 后端 + 前端网页 Chat UI
- 多工具调用（Google 搜索、天气/时间、数学计算、RAG 知识库等）
- 提示注入安全检测与会话锁定机制

---

## 1. 功能概览

- **ReAct Agent**：`agent.py`
	- 固定 Thought → Action → Observation → Final Answer 模式
	- 解析 LLM 输出中的工具调用（`Action` / `Action Input`，JSON 参数）
	- 统一路由到 `tool.py` 中定义的工具并回填 Observation
- **LLM 封装**：`llm.py`
	- `Qwen3`：纯文本模型
	- `Qwen3VL`：多模态模型（图片 + 文本），当前主用
- **CLI 入口**：`main.py`
	- 加载配置与日志
	- 自动检测 `rag/wiki_vector_db` 开启 RAG
	- 支持图片命令：`@image:path` / `@paste` / `@clear` / `@show`
	- 支持 `reset guard` 解锁安全防护
- **工具系统**：`tool.py` / `calculator.py`
	- Google 搜索、时间/天气查询
	- 标量/三角/矩阵/定积分/表达式求值
	- 知识库问答（RAG）
- **RAG 模块**：`rag/`
	- `document_processor.py`：文档加载 + 滑动窗口分块
	- `vector_store.py`：`sentence-transformers` + `faiss` 向量索引
	- `rag_engine.py`：检索 + 上下文拼接
	- `build_vector_db.py`：从 `wiki_docs/` 构建向量库
- **安全防护**：`security/prompt_guard.py`
	- 检测四类 Prompt 攻击，支持锁定会话
	- `config.yaml` 中可配置是否开启及锁定策略
- **Web & API**：
	- 后端：`backend/main.py`（FastAPI，多会话管理、文件上传、工具信息、WebSocket 等）
	- 前端：`frontend/`（基于 Vite + TypeScript + Tailwind 的 Chat UI）
- **Benchmark & 测试**：
	- `benchmark/`、`benchmarkTest/ALFworld.py`：示例评测集与 ALFworld 测试入口
	- `tests/`：pytest 用例（PromptGuard + Agent 安全集成）

---

## 2. 环境与依赖

推荐使用 Python 3.11（仓库提供 `.python-version`，示例使用 3.11.6）。

### 2.1 创建虚拟环境并安装依赖

```bash
cd AIAA3102_final

# 可选：使用 pyenv
pyenv install 3.11.6
pyenv virtualenv 3.11.6 AIAA3102_final
pyenv local AIAA3102_final

python -m pip install --upgrade pip
pip install -r requirements.txt
```

主要第三方包（详见 `requirements.txt`）：

- `transformers`, `vllm`（可选）、`sentence-transformers`
- `faiss-cpu` 或 `faiss-cuda`
- `numpy`, `scipy`, `requests`, `PyYAML`, `json5`
- `langchain_google_community`, `modelscope`（如需使用 modelscope 下载模型）
- 后端：`fastapi`, `uvicorn`, `loguru`, `pydantic`

> 注意：FAISS 需根据硬件选择 CPU 或 CUDA 版本。

### 2.2 模型准备

当前默认以 **Qwen3-VL** 为主（多模态）。

```bash
pip install modelscope
modelscope download --model Qwen/Qwen3-VL-8B-Instruct  # 示例名称，请按实际模型替换
```

下载后，在 `llm.py` / `config` 中按需要配置模型路径（如使用本地权重目录）。若使用纯文本模式，可切换为 `Qwen3`。

---

## 3. 配置说明

根目录 `config.yaml`：

```yaml
STATE_OF_MODEL: 1   # 0=Benchmark 测试模式, 1=正常交互模式

security:
	enable_prompt_guard: true
	lock_on_violation: true

google:
	credentials_path: jsons/goole_search.json
```

- **安全配置**：控制是否启用提示注入检测，以及高危时是否锁定当前会话。
- **Google 搜索**：指定自定义搜索引擎凭据 JSON 路径。

### 3.1 Google 搜索凭据

在 Programmable Search Engine + Google Cloud Console 中创建：

1. 创建搜索引擎，获得 `cx`。
2. 申请 API Key。
3. 写入 `jsons/goole_search.json`：

	 ```json
	 {
		 "search_engine_id": "your_cx",
		 "api_key": "your_api_key"
	 }
	 ```

若使用自定义路径，可修改 `config.yaml` 或设置环境变量：

```bash
export GOOGLE_SEARCH_CREDENTIALS=/path/to/your.json
```

---

## 4. 本地 CLI 使用

### 4.1 构建（可选）RAG 知识库

默认文档位于 `wiki_docs/`，构建向量库到 `rag/wiki_vector_db/`：

```bash
python rag/build_vector_db.py -i wiki_docs -o rag/wiki_vector_db
``;

成功后，`main.py` 会自动检测并启用 RAG 工具。

### 4.2 启动终端 Agent

```bash
python main.py
```

特性：

- 普通对话：直接输入自然语言
- 图片输入（多模态，仅在 Qwen3VL 下生效）：
	- `@image:/path/to/img.png` 添加图片路径
	- `@paste` 从剪贴板抓取图片（需安装 Pillow 且系统支持）
	- `@clear` 清空当前会话图片
	- `@show` 显示当前已添加图片列表
- 安全锁定解锁：
	- 当触发高危提示注入时，会话可能被锁定
	- 在 CLI 中输入 `reset guard` 或 `/reset_guard` 解锁

---

## 5. Web 前后端与 API

### 5.1 启动后端 API（FastAPI）

后端入口：`backend/main.py`。

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

主要能力：

- 会话管理（创建/重命名/删除/列出）
- 聊天接口（文本 + 图片，内部调用 `Agent`）
- 工具列表查询
- 图片上传（保存到 `uploads/` 并暴露静态路径）

### 5.2 启动前端网页

前端位于 `frontend/`，基于 Vite + TypeScript + React/Tailwind：

```bash
cd frontend
npm install
npm run dev
```

默认会在本地 5173 端口启动开发服务器，与 FastAPI 后端联动形成一个多模态 Chat UI。

---

## 6. 测试与 Benchmark / 评估

本项目已经实现四类主要评估：

- **ReAct 评估**：围绕工具调用正确性、多步推理能力和错误恢复能力的测试（包括 ALFWorld 环境评估）。
- **RAG 评估**：针对 Wiki 文档知识库的检索效果与答案质量评估。
- **搜索增强评估**：在 Web 搜索（Google / Tavily）+ RAG 的组合场景下，评估事实性问答与时效性问题的回答质量。
- **安全/危险测试**：多种提示注入与不安全指令场景，用于验证 `PromptGuard` 的检测与拦截能力。

### 6.1 单元测试

```bash
pytest tests/test_prompt_guard.py tests/test_agent_prompt_guard.py
```

- `test_prompt_guard.py`：验证提示注入检测规则
- `test_agent_prompt_guard.py`：验证 Agent + Guard 的集成行为（锁定/解锁等）

### 6.2 网络 API 测试

```bash
python testapi.py
```

用于测试 `query_time` 等外部 HTTP API 的连通性与返回格式。

### 6.3 ReAct / 环境交互评估（ALFWorld）

与 ALFWorld 的交互评估入口位于：

- `benchmarkTest/ALFworld.py`

使用前需要先安装 ALFWorld 及其依赖（建议使用独立虚拟环境或在当前环境中安装）：

```bash
pip install alfworld
# 某些版本可能需要从源码安装或额外依赖，请参考 ALFWorld 官方文档
```

然后运行评测脚本，例如：

```bash
python benchmarkTest/ALFworld.py
```

> 该评估主要用于测试 ReAct Agent 在交互式环境中的规划与工具使用能力。

### 6.4 RAG 评估

RAG 评估脚本和数据位于 `benchmark/` 与 `rag/` 相关模块中（如针对 `wiki_docs/` 的问答评估）。

典型流程：

1. 确保已构建 `rag/wiki_vector_db`。
2. 准备一组带“黄金答案”的问题集合（已在项目中编写）。
3. 通过评测脚本对 Agent 的 RAG 回答进行自动或半自动评分（BLEU/ROUGE/简单准确率等）。

> 具体指标与脚本细节可在课程报告中详细说明，这里 README 只给出入口和整体流程。

### 6.5 搜索增强评估（Web Search + RAG）

- 相关模块：`tool.py` 中的 `GoogleSearchTool` / `TavilySearchTool`，以及 `rag/rag_engine.py`。
- 典型场景：事实性问答、新闻与时效性信息、需要跨多网页整合的开放域问题。
- 典型设置：
	- 仅使用 LLM 内部知识（不调用搜索 / RAG）。
	- 启用 Web 搜索但不拼接本地向量库。
	- Web 搜索 + RAG 联合（先外部检索再利用本地向量库提供背景知识）。
- 主要观察指标：回答的准确性、时效性、引用内容与真实网页的一致性，以及错误来源（检索失败 vs 推理错误）。

---

## 7. 安全机制与提示注入防护

- `security/prompt_guard.py`：
	- 四类高危模式：Prompt Injection / Insecure Output Handling / Insecure Plugin or Tool Usage / Excessive Agency
	- 返回分类、严重级别、修复建议
- 与 `Agent` 集成：
	- 在检测到高危输入时可锁定当前会话
	- 日志中记录攻击样例与处理结果
- 通过 CLI 命令 `reset guard` 进行管理员人工解锁

---

## 8. 开发者提示与TODO

已完成：

- ReAct 循环与工具调用框架
- 多模态 LLM 接入（Qwen3 / Qwen3VL）
- RAG 检索模块及 Wiki 向量库示例
- FastAPI 后端与前端网页 Demo
- 提示注入检测与安全日志

- 为各工具增加调用超时、失败重试与熔断机制

后续可拓展方向（建议写入课程报告中）：
- 构建系统化评估数据集与自动评分脚本
- 引入自洽性、多轨迹投票或代理协作机制提升鲁棒性
- 基于检索置信度/引用的幻觉控制与答案置信度展示
