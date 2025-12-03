**一、引言与任务背景**

- 课程背景与项目目标：面向《高级人工智能应用》课程的最终项目，我们需要证明自己能够在有限资源内落地一套“可运行、可评测、可防护”的多模态智能体系统。本工作以 Qwen3/Qwen3-VL 系列模型为核心，构建一个既能在终端 CLI 运行、又能通过 FastAPI + Web UI 对外服务的 ReAct Agent，并在工具调用、RAG、ALFWorld、Web 搜索及安全评估等多个维度完成可复现实验。  
- 问题空间与挑战：  
  - **多轮推理 + 工具使用**：ReAct 框架强调 Thought → Action → Observation 回路，需要让大模型在有限步数内稳定调用工具、解析观测、产出 Final Answer。  
  - **多模态感知**：Qwen3-VL 支持图片输入，意味着 CLI、前端、后端都必须提供图片上传/管理能力，并保证这些信息能被注入 LLM 会话上下文。  
  - **开放环境与安全风险**：Agent 可访问 Google/Tavily、RAG 知识库、Web API 和外部文件，必须继续防范提示注入、信息泄露、滥用插件等安全威胁。  
- 项目贡献纵览：  
  1. 提供一个文件级模块化的 ReAct Agent（`agent.py`、`tool.py`、`rag/`、`security/`），即使在课程服务器上也能快速部署；  
  2. 打通“终端 CLI ↔ FastAPI ↔ Vite 前端”三种交互路径，形成多会话 Web UI；  
  3. 构建 Wiki RAG 数据管线与 `rag_engine`，支撑向量检索 + Prompt 拼接；  
  4. 借鉴 LLM_OWASP_Scanner 的启发式规则实现 PromptGuard，并编写 `tests/` 验证其有效性；  
  5. 设计 ALFWorld、RAG、Web、Security 四条 Benchmark/实验脚本，为后续分析提供数据与日志。  

**二、系统整体设计**

- 架构概览：整体参照“多模态 ReAct Agent + 工具集 + 安全层 + 多端入口”的思路搭建。图 2-1 展示了典型链路：用户输入在 CLI 或 Web 前端产生，经 FastAPI 或本地主进程转发给 `Agent`，由 `ToolsManager` 统一调度各类工具，并可选择性读取 `rag/wiki_vector_db`；安全模块在输入侧执行拦截，日志模块负责追踪每次 Thought/Action。  

  ![系统整体结构图](/images/structure_zh.png)

- 核心模块角色细化：  
  - `main.py`：CLI 入口，集成图片命令、Prompt Guard 解锁、日志打印，并基于配置自动加载 RAG。  
  - `agent.py`：封装 ReAct 回路、工具描述拼装、格式校验、工具调用、历史管理，是系统“中枢神经”。  
  - `llm.py`：抽象 `BaseLLM` 接口，并提供 `Qwen3`（文本）与 `Qwen3VL`（多模态）实现，统一 chat API 与思考模式处理。  
  - `tool.py` + `calculator.py`：集中描述工具签名、参数校验和调用逻辑，既包括 Google/Tavily/时间天气等外部 API，也包括矩阵/积分/表达式计算与 RAG Query。  
  - `rag/`：含 `document_processor.py`、`vector_store.py`、`rag_engine.py` 等，用于构建滑窗分片、FAISS 索引和 top-k 拼接。  
  - `security/prompt_guard.py`：提供启发式检测、风险评级、锁定策略，并被 `agent.py` 的 `_apply_prompt_guard` 调用。  
  - `backend/main.py`：FastAPI 多会话服务，负责聊天路由、工具列表、图片上传、静态文件分发和 WebSocket 推送。  
  - `frontend/`：Vite + TS + Tailwind UI，实现多会话列表、图片消息、Latex 渲染与工具日志视图。  
- 系统设计原则：  
  - **轻量化**：避免复杂框架，所有逻辑拆解到少量可读文件，方便课程答辩与快速 Debug。  
  - **可替换性**：LLM、工具、RAG、安全模块通过清晰接口隔离，可替换为其他模型或向量库。  
  - **运行鲁棒性**：大量日志（`loguru`）、格式校验、工具错误提示和 Guard 锁定，确保 Agent 在真实场景下不会失控。  
  - **多入口一致性**：CLI 与 Web 共享同一 `Agent`/工具集，保证实验结果与线上 Demo 一致。  

**三、关键设计选择说明**

1. **ReAct Prompt 与 Agent 设计**  
   - 采用双语系统提示与硬约束格式：`REACT_PROMPT` 在英文 + 中文两套规则中明确列出 10 条关键准则，强调“一次只用一个工具、所有事实必须来自 Observation、区分闲聊与事实查询”等；这直接减少了 Qwen3VL 在工具调用时的随机性。  
   - 历史管理与格式校验：`Agent.check_response` 会统计 Thought/Action/Final Answer 出现次数，若格式错误则向 LLM 发送 system 纠错提示；`parse_latest_plugin_call` 则解析最近一次 Action，确保多 Action 时只执行第一个并追加警告。  
   - 工具调用协议：`call_plugin` 使用 `json5` 解包参数，可容忍模型输出的单引号、尾逗号等非标准 JSON；`ToolsManager` 则保证所有工具共享一致的入参/出参模板，并在 CLI、API、ALFWorld 中复用。  
   - 多模态上下文：Agent 只在首轮向模型传图片，后续轮次依赖历史，既避免重复传输大图，也与 Qwen3VL 的上下文缓存机制兼容。  

2. **多模态模型的选择：Qwen3-VL**  
   - 选择理由：Qwen3-VL 在阿里云开源版本中提供 8B 等轻量参数，能够在单机 RTX 3090/4090 上加载；同时具备图片理解能力，适合 Web 页面解析、截图问答、ALFWorld 场景图等需求。  
   - 适配措施：CLI 端提供 `@image`、`@paste`、`@show`、`@clear` 命令；前端支持本地上传与 URL 图片，并通过 WebSocket 将多模态消息传回后端；LLM 封装暴露 `supports_multimodal` 标记，Agent 据此决定是否允许图片。  
   - 资源与部署权衡：`llm.py` 同时保留纯文本 `Qwen3` 备选，可在 GPU 资源紧张时退回；配置文件允许切换本地权重或远程推理服务，为后续扩展留出空间。  

3. **工具系统设计**  
   - 统一声明：所有工具通过 `TOOL_DESC` 描述模板向 LLM 暴露用途与参数，这些描述同时用于 CLI/Web UI 的“可用工具列表”，保证人机对齐。  
   - 分类与路由：`tool.py` 将工具分为外部搜索/时间天气 API、数学计算（`calculator.py`）、RAG 检索和系统工具（如 `get_cur_time`）。ToolsManager 在初始化时读取 `rag_db_path`，自动注入/禁用 RAG 工具。  
   - 容错策略：对 Action Input 做 `json5` 容错、对多 Action 给出警告、对工具执行异常生成 system 消息提示用户改参；这些机制在 ALFWorld 与 Web Benchmarks 中显著降低了崩溃概率。  

4. **RAG 管线设计**  
   - 数据与预处理：`rag/document_processor.py` 针对 `wiki_docs/` 中的 AI/ML 文章执行滑动窗口切分（窗口 512/步长 256，可配置），并保留段落原始标题用于引用。  
   - 向量化与存储：`vector_store.py` 使用 `sentence-transformers` 生成嵌入，再写入 `faiss` 索引；脚本 `rag/build_vector_db.py` 接受输入/输出路径参数，方便扩容或替换语料。  
   - 检索与拼接：`rag_engine.py` 暴露 `retrieve` 和 `format_context`，允许 Agent 控制 top-k、评分阈值和上下文拼接方式；最终 context 与用户问题一起注入 LLM Prompt，形成检索增强回答。  

5. **安全模块 PromptGuard 设计**  
   - 规则来源：`security/prompt_guard.py` 从 LLM_OWASP_Scanner 中筛选四类高危场景（Prompt Injection、Insecure Output Handling、Insecure Plugin/Tool Usage、Excessive Agency），并提炼多语言正则规则以涵盖英文/中文攻击指令。  
   - 启发式策略：通过 Unicode 归一化、零宽字符清理和空白折叠得到 `normalized_prompt`，再执行正则匹配返回 `PromptGuardFinding`；`has_high_risk` 将 severity=High 的项视为阻断条件。  
   - 与 Agent 集成：`agent.text` 在收到用户输入后先调用 `_apply_prompt_guard`，若命中规则且配置了 `lock_on_violation`，则直接返回警告并将会话标记为锁定，管理员需在 CLI 输入 `reset guard` 才能继续；`tests/test_prompt_guard.py` 与 `test_agent_prompt_guard.py` 已覆盖核心逻辑。  

6. **接口与前端设计**  
   - FastAPI 后端：`backend/main.py` 提供多会话 CRUD、聊天、图片上传、工具列表等 REST + WebSocket 接口，采用 `loguru` 记录每次工具调用，方便 Benchmark 回溯。  
   - 前端：`frontend/` 使用 Vite + TypeScript + Tailwind，实现多会话侧栏、消息气泡、图片预览、LaTeX 渲染和工具日志面板；与后端保持 JSON 格式一致，便于课堂展示。  


**四、实验设置与评估方案**

1. **ReAct / 环境交互评估（ALFWorld）**  
  - 评测脚本：`benchmarks/React/ALFworld.py`，统一使用官方 ALFWorld 20 个标准关卡（“放置”“加热”“清洁”等混合任务），每回合限制 40 步。  
  - 数据收集：对「React」与「React_baseline」两个配置分别运行一整套关卡，自动落盘 `detailed_results.csv/json`。  
  - 评估指标：  
    - **任务成功率**：成功局数 / 总局数。  
    - **动作执行质量**：平均步数、平均成功步数、动作成功率（成功步 / 总步）。  
    - **失败原因分类**：解析 `task_preview` 或 `error` 字段，统计触发频率，用于定位瓶颈（例如 decoder prompt limit）。  
  - 以 `analysis/analyze_react.py` 生成的 Markdown 与图表为准，可一键扩展到更多实验批次。  

2. **RAG 评估**  
  - 数据来源与构造方式（针对 wiki_docs 的问答对）。  
  - 评估设置：
    - 仅使用 LLM 内部知识（不调用 RAG）；  
    - 启用 RAG 检索；
  - 评估方式：使用更大的deepseek3.2模型进行打分，通过查看曾经的打分记录保证评分的一致性。

3. **搜索增强评估（Web Search + RAG）**  
  - 工具与模块：`tool.py` 中的 `GoogleSearchTool` / `TavilySearchTool`，结合 `rag_engine.py`。  
  - 典型任务：事实性问答、时效性信息查询（新闻 / 事件）、需要跨多页面整合的信息检索。  
  - 评估设置：  
    - 仅使用 LLM 内部知识（不调用搜索 / RAG）；  
    - 启用 Web 搜索；  
  - 指标与观察：
    - 回答逻辑正确性 （使用 deepseek3.2 模型评分）；
    - 事实引用准确性（人工评分）。

4. **安全/提示注入评估**  
  - test_prompt_guard.py 和 test_agent_prompt_guard.py 的测试设计。  
  - 攻击样例类别：试图覆盖系统指令、诱导泄露机密、指使关闭安全防护等。  
  - 评估指标：检测率和误报率。  

**五、实验结果与失败分析**

1. **ALFWorld 结果汇总**  
   - 数据来源：`benchmark_results/React` 与 `React_baseline`，由 `analysis/analyze_react.py` 自动汇总为 `react_analysis.md`。  
   - 关键指标：  
     - **React（ReAct Agent）**：22 局中完成 17 局，成功率 **77.27%**，平均步数 17.95，动作成功率 83.03%。失败 5 局全部可追溯到 “decoder prompt limit” 的截断。  
     - **React_baseline（无 ReAct Loop）**：20 局仅完成 8 局，成功率 **40.00%**，平均步数 27.10、动作成功率 63.00%。除个别 “clean/heat” 任务外，大部分因为世界状态探索失败或同样的解码截断。  
  
    ![React vs. Baseline 成功率对比](benchmark_results/charts/react_success_rates.png)
   - 结果解读：  
     - ReAct Agent 在长链条任务（例如 *put two keychain in safe*、*find two pillow and put them in sofa*）保持 80%+ 动作准确率，说明“Thought→Action→Observation” 带来的上下文记忆确实减少了无效动作。  
     - 基线在所有 “examine with desklamp”/“heat food and trash it” 任务上几乎全败（动作成功率 < 30%），体现出缺少工具调用/规划时，模型更容易在高分支环境中迷路。  
   - 失败案例分析：  
     - React 的 5 次失败全部是 “Decoder prompt limit”——即多轮 ReAct 导致 token 超限后，中断了 Observation 写回；后续会通过截断思维链或适配 `max_new_tokens` 来规避。  
     - Baseline 的失败则更多出现在「Examine + 清洁」组合，说明没有链式思考时很难保持场景状态。  
  
  ![React 失败原因分布](benchmark_results/charts/react_failures.png)

3. **RAG / Web模块结果汇总**  
   - 实验数据位置：  
     - `benchmarks/RAG/`, `benchmarks/web/`。
   - 核心结果：  
     - 评分统计（平均分、方差等）。
   - 结果解读：  
     - 哪些任务类型评分高（例如简单“RAG查询” vs “RAG查询+计算”）。  
     - 哪些任务容易评分低（例如需要多轮搜索的任务）。  
   - 失败案例分析：  
     - 
4. **安全模块结果汇总**
   - 


**六、消融实验设计与分析**

1. **有/无 React 循环**  
   - 设置：同一套 ALFWorld 20 局，分别启用 ReAct Prompt（`React`）与关闭 ReAct（`React_baseline`）。  
   - 对比结果：`React` 成功率 **+37.3 pct**（77.27% vs 40.00%），平均动作成功率 +20 pct（83.03% vs 63.00%），同时平均步数更低（17.95 vs 27.10）。  
   - 分析：ReAct 带来的显著收益主要来自（1）显式记录 Thought，减少重复探索；（2）Observation 反馈让模型及时矫正错误，从而在多步骤任务中保持 80%+ 的步骤准确率。基线由于一次性生成整条执行序列，在“examine/clean/heat”复合任务中经常迷失或堆叠无效动作。  
   - 失败案例：ReAct 的 5 次失败集中在 `decoder prompt limit`——ReAct 输出过长导致截断；基线失败遍布所有需要工具联动的地图，且常见 “Examine the cd with the desklamp.”、 “Cool some bread and put it in countertop.” 等动作用尽上限仍未完成。  

2. **有/无检索（RAG Ablation）**  
   - 设置：  
     - 有检索：启用 wiki_vector_db，RAG 工具可用。  
     - 无检索：关闭向量库或禁用 RAG 工具（只靠模型内知识）。  
   - 对比方式：  
     - 知识密集问题的评分结果。  
   - 对比结果：
     - 
   - 分析：  
     - RAG 对于长尾知识、课程相关专业知识的加成程度。  
     - 失败案例：。  

3. **有/无搜索增强（Web Search Ablation）**  
   - 设置：  
     - 关闭 Web 搜索，仅使用本地 RAG 或模型内部知识；  
     - 启用 Google/Tavily 搜索，但不使用本地向量库；  
     - 同时开启 Web 搜索 + RAG（搜索增强检索）。  
   - 对比方式：  
     - 对时效性问题、跨网页聚合问题的逻辑评分和事实评分的加权评分。
   - 对比结果：
     - 
   - 分析：  
     - Web搜索对时效性问题的加成程度。  
     - 失败案例：。

4. **有/无安全模块（Verifier / Guard Ablation）**  
   - 设置：  
     - 开启 `enable_prompt_guard`：所有输入经过检测，高危时锁定会话。  
     - 关闭 PromptGuard：直接将用户指令传给 LLM。  
   - 对比内容：  
     - 能否成功阻断明显的提示注入攻击。  
     - 误报情况：正常指令被错误拦截的例子。  
   - 对比结果：
     - 
   - 分析：  
     - 安全模块的收益与成本（用户体验、自由度 vs 安全性）。  
     - 未来可以引入更强 Verifier（例如基于第二个模型或规则+LLM 判别器）。  

**七、讨论与未来工作**

- 当前系统的主要优势：  
  - 架构清晰、模块化（工具、RAG、安全易替换）。  
  - 多模态支持与 ReAct 框架结合自然。  
  - 已有完整的测试与 Benchmark 管线。  
  - 工具调用的超时、重试与熔断机制。  
- 未来改进方向：  
  - 引入多轨迹投票、自洽性检查。  
  - 更精细的幻觉控制策略：基于检索置信度和引用粒度来给出答案置信度。  

**八、结论**

- 总结本项目在多模态 ReAct Agent、RAG 检索和安全防护方面的整体设计与实验结果。  
- 强调关键发现：  
  -
- 简要展望：如何将本系统推广到更大规模、更多模态与更真实的应用场景。