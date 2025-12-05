"""
通用能力测试模块 (RAG + Web Search)
用于评估 Agent 在知识检索、网络搜索、工具调用等方面的能力

使用 DeepSeek API 对回答质量进行自动评分 (0-1)
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple
from loguru import logger

from agent import Agent


# 尝试导入 OpenAI (用于 DeepSeek API)
try:
    from openai import OpenAI
    DEEPSEEK_AVAILABLE = True
except ImportError:
    DEEPSEEK_AVAILABLE = False
    logger.warning("OpenAI SDK not installed. Install with: pip install openai")


class GeneralCapabilityTester:
    """通用能力测试器"""
    
    def __init__(self, agent: Agent, result_dir: str = "benchmark_results/general_capability"):
        """
        初始化测试器
        
        Args:
            agent: Agent 实例
            result_dir: 结果保存目录
        """
        self.agent = agent
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成时间戳
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.result_dir / f"session_{self.timestamp}"
        self.session_dir.mkdir(exist_ok=True)
        
        # 初始化 DeepSeek
        self.deepseek_client = None
        if DEEPSEEK_AVAILABLE:
            # 优先从环境变量读取，其次从配置文件读取
            api_key = os.getenv("DEEPSEEK_API_KEY")
            
            if not api_key:
                # 尝试从 config.yaml 读取
                try:
                    import yaml
                    config_path = Path(__file__).parent.parent.parent.parent / "config.yaml"
                    if config_path.exists():
                        with open(config_path, 'r', encoding='utf-8') as f:
                            config = yaml.safe_load(f)
                            api_key = config.get("DEEPSEEK_API")
                            if api_key:
                                logger.info("DeepSeek API key loaded from config.yaml")
                except Exception as e:
                    logger.warning(f"Failed to load config.yaml: {e}")
            
            if api_key:
                self.deepseek_client = OpenAI(
                    api_key=api_key,
                    base_url="https://api.deepseek.com"
                )
                logger.info("DeepSeek API initialized")
            else:
                logger.warning("DEEPSEEK_API_KEY not set (check environment variable or config.yaml). Scoring will be disabled.")
        
        self.results = []
    
    def load_test_cases(self, file_path: str) -> List[Dict[str, Any]]:
        """
        加载测试案例文件
        
        Args:
            file_path: 测试案例文件路径
            
        Returns:
            测试案例列表
        """
        test_cases = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 解析测试案例（按分隔符切割）
        sections = content.split("=" * 80)
        
        current_case = {}
        for section in sections:
            section = section.strip()
            if not section:
                continue
            
            lines = section.split('\n')
            
            mode = None
            for line in lines:
                line = line.strip()
                if line.startswith('测试案例 #'):
                    if current_case:
                        test_cases.append(current_case)
                    current_case = {}
                    mode = None
                elif line.startswith('任务ID:'):
                    current_case['id'] = line.split(':', 1)[1].strip()
                elif line.startswith('任务类型:'):
                    current_case['type'] = line.split(':', 1)[1].strip()
                elif line.startswith('任务描述:'):
                    current_case['description'] = []
                    mode = 'description'
                elif line.startswith('用户输入:'):
                    current_case['user_input'] = []
                    mode = 'user_input'
                elif line.startswith('评估要点:'):
                    current_case['evaluation_points'] = []
                    mode = 'evaluation'
                elif line.startswith('标准答案:'):
                    current_case['standard_answer'] = []
                    mode = 'standard_answer'
                elif line and mode == 'description':
                    current_case['description'].append(line)
                elif line and mode == 'user_input':
                    current_case['user_input'].append(line)
                elif line and mode == 'evaluation':
                    current_case['evaluation_points'].append(line)
                elif line and mode == 'standard_answer':
                    current_case['standard_answer'].append(line)
        
        # 添加最后一个案例
        if current_case:
            test_cases.append(current_case)
        
        # 合并多行文本
        for case in test_cases:
            if 'description' in case:
                case['description'] = '\n'.join(case['description'])
            if 'user_input' in case:
                case['user_input'] = '\n'.join(case['user_input'])
            if 'evaluation_points' in case:
                case['evaluation_points'] = '\n'.join(case['evaluation_points'])
            if 'standard_answer' in case:
                case['standard_answer'] = '\n'.join(case['standard_answer'])
        
        logger.info(f"Loaded {len(test_cases)} test cases from {file_path}")
        return test_cases
    
    def _format_trajectory(self, history: List, final_output: str) -> str:
        """
        格式化完整的执行轨迹（包含工具调用）
        
        Args:
            history: 执行历史
            final_output: 最终输出
            
        Returns:
            格式化的轨迹字符串
        """
        trajectory_parts = []
        tool_results_summary = []  # 收集所有工具返回内容
        
        for msg in history:
            role = msg.get('role', '')
            content = msg.get('content', '')
            
            if role == 'system':
                # 系统指令（通常是提示词）
                trajectory_parts.append(f"[System Instruction]\n{content}\n")
            elif role == 'user':
                # 用户输入
                if isinstance(content, str):
                    trajectory_parts.append(f"[User Input]\n{content}\n")
                else:
                    # 多模态内容
                    trajectory_parts.append(f"[User Input - Multimodal]\n{str(content)}\n")
            elif role == 'assistant':
                # Agent 的思考和行动
                trajectory_parts.append(f"[Agent Response]\n{content}\n")
            elif role == 'tool':
                # 工具执行结果 - 高亮显示
                tool_name = msg.get('name', 'unknown_tool')
                tool_content = f"{'='*60}\n⚠️ 【工具返回内容 - {tool_name}】⚠️\n{'='*60}\n{content}\n{'='*60}\n"
                trajectory_parts.append(tool_content)
                tool_results_summary.append(f"- {tool_name}: {content[:200]}...")  # 收集摘要
        
        # 在开头添加工具调用摘要
        if tool_results_summary:
            summary = "【工具调用摘要】\n" + "\n".join(tool_results_summary) + "\n\n"
            trajectory_parts.insert(0, summary)
        
        # 添加最终输出 - 也高亮显示
        trajectory_parts.append(f"{'='*60}\n🎯 【Agent 最终回答】\n{'='*60}\n{final_output}\n{'='*60}\n")
        
        return "\n".join(trajectory_parts)
    
    def run_test_case(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        运行单个测试案例
        
        Args:
            test_case: 测试案例
            
        Returns:
            测试结果
        """
        case_id = test_case.get('id', 'unknown')
        case_type = test_case.get('type', 'unknown')
        user_input = test_case.get('user_input', '')
        
        print(f"\n{'='*70}")
        print(f"🧪 测试案例 #{case_id} - {case_type}")
        print(f"{'='*70}")
        print(f"\n📝 用户输入:\n{user_input}\n")
        
        logger.info(f"Running test case #{case_id}")
        logger.debug(f"user_input type: {type(user_input)}, length: {len(user_input) if user_input else 0}")
        logger.debug(f"user_input repr: {repr(user_input[:100]) if user_input else 'EMPTY'}")
        
        # 记录开始时间
        start_time = time.time()
        
        try:
            # 调用 Agent（保存历史记录以供评分使用）
            agent_output, execution_history = self.agent.text(user_input, history=None)
            
            elapsed_time = time.time() - start_time
            
            print(f"\n{'─'*70}")
            print(f"🤖 Agent 回答:")
            print(f"{'─'*70}")
            print(agent_output)
            print(f"\n⏱️  响应时间: {elapsed_time:.1f}秒")
            
            # 提取 Final Answer
            final_answer_marker = "Final Answer:"
            if final_answer_marker in agent_output:
                final_answer = agent_output[agent_output.rfind(final_answer_marker) + len(final_answer_marker):].strip()
            else:
                final_answer = agent_output.strip()
            
            # 构建完整轨迹（包含工具调用历史）
            full_trajectory = self._format_trajectory(execution_history, agent_output)
            
            result = {
                'case_id': case_id,
                'case_type': case_type,
                'user_input': user_input,
                'agent_output': agent_output,
                'final_answer': final_answer,
                'full_trajectory': full_trajectory,  # 新增：完整执行轨迹
                'elapsed_time': elapsed_time,
                'evaluation_points': test_case.get('evaluation_points', ''),
                'standard_answer': test_case.get('standard_answer', ''),  # 新增：标准答案
                'error': None
            }
            
            logger.info(f"Test case #{case_id} completed in {elapsed_time:.1f}s")
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            error_msg = str(e)
            
            print(f"\n❌ 执行出错: {error_msg}")
            logger.error(f"Test case #{case_id} error: {e}")
            
            result = {
                'case_id': case_id,
                'case_type': case_type,
                'user_input': user_input,
                'agent_output': None,
                'final_answer': None,
                'elapsed_time': elapsed_time,
                'evaluation_points': test_case.get('evaluation_points', ''),
                'standard_answer': test_case.get('standard_answer', ''),
                'error': error_msg
            }
        
        return result
    
    def calculate_f1_score(self, result: Dict[str, Any]) -> Tuple[float, str]:
        """
        计算 F1 分数：衡量最终答案与标准答案的匹配度
        如果没有标准答案（如 Web 搜索测试），则使用 DeepSeek API 评分
        
        Args:
            result: 测试结果
            
        Returns:
            (F1分数/DeepSeek分数, 评分说明)
        """
        if result['error']:
            return 0.0, f"Agent execution error: {result['error']}"
        
        # 获取标准答案
        standard_answer = result.get('standard_answer', '')
        
        # 提取最终答案
        final_answer = result.get('final_answer', '')
        if not final_answer:
            # 尝试从 agent_output 中提取
            agent_output = result.get('agent_output', '')
            if 'Final Answer:' in agent_output:
                final_answer = agent_output.split('Final Answer:')[-1].strip()
            else:
                final_answer = agent_output
        
        if not final_answer:
            return 0.0, "未检测到最终答案"
        
        # 如果没有标准答案，使用 DeepSeek 评分（适用于 Web 搜索等无标准答案的测试）
        if not standard_answer:
            logger.info("No standard answer found, using DeepSeek scoring")
            return self.score_with_deepseek(result)
        
        # 有标准答案：计算 F1 分数
        f1, precision, recall = self._compute_text_f1(final_answer, standard_answer)
        
        reason = f"""F1 评分分析（与标准答案对比）:
- Precision (精确率): {precision:.2%} - Agent 答案中有 {precision:.0%} 与标准答案匹配
- Recall (召回率): {recall:.2%} - 标准答案中有 {recall:.0%} 被 Agent 覆盖
- F1 Score: {f1:.2%} - 综合准确度

标准答案长度: {len(standard_answer)} 字符
Agent 答案长度: {len(final_answer)} 字符
"""
        
        return f1, reason
    
    def _compute_text_f1(self, answer: str, reference: str) -> Tuple[float, float, float]:
        """
        计算文本 F1 分数（基于字符级别的匹配）
        
        Args:
            answer: 最终答案文本
            reference: 参考文本（工具返回内容）
            
        Returns:
            (f1, precision, recall)
        """
        # 简单的字符级 n-gram 匹配（使用 3-gram）
        def get_ngrams(text: str, n: int = 3) -> set:
            """提取 n-gram"""
            text = ''.join(text.split())  # 移除空格
            return set(text[i:i+n] for i in range(len(text) - n + 1))
        
        answer_ngrams = get_ngrams(answer, n=3)
        reference_ngrams = get_ngrams(reference, n=3)
        
        if not answer_ngrams or not reference_ngrams:
            return 0.0, 0.0, 0.0
        
        # 计算交集
        common = answer_ngrams & reference_ngrams
        
        # 计算 precision 和 recall
        precision = len(common) / len(answer_ngrams) if answer_ngrams else 0.0
        recall = len(common) / len(reference_ngrams) if reference_ngrams else 0.0
        
        # 计算 F1
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        return f1, precision, recall
    
    def score_with_deepseek(self, result: Dict[str, Any], is_web_search: bool = False) -> Tuple[float, str]:
        """
        使用 DeepSeek API 对回答进行评分
        
        Args:
            result: 测试结果
            is_web_search: 是否为在线搜索任务（自动检测：无标准答案时为 True）
            
        Returns:
            (分数, 评分理由)
        """
        if not self.deepseek_client:
            logger.warning("DeepSeek client not available, skipping scoring")
            return 0.0, "DeepSeek API 未配置（需设置 DEEPSEEK_API_KEY 环境变量）"
        
        if result['error']:
            return 0.0, f"Agent execution error: {result['error']}"
        
        # 自动检测是否为 Web 搜索任务（无标准答案）
        if not is_web_search and not result.get('standard_answer'):
            is_web_search = True
            logger.info("Auto-detected as web search task (no standard answer)")
        
        # 根据任务类型选择不同的评分提示
        if is_web_search:
            # 在线搜索任务：只评估流程和完成度，不评估事实准确性
            scoring_prompt = f"""你是一个 AI Agent 流程评估专家。请评估 Agent 完成在线搜索任务的**流程和方法**，不要评判搜索结果的事实准确性。

【任务类型】
{result['case_type']} - 在线搜索任务

【用户问题】
{result['user_input']}

【评估要点】
{result['evaluation_points']}

【Agent 完整执行轨迹】（包含工具调用过程和返回的内容）
{result.get('full_trajectory', result['agent_output'])}

【重要评估规则】
⚠️ 请仔细查看上面的【Agent 完整执行轨迹】：
1. 查看 ⚠️【工具返回内容】部分 - 这是搜索工具实际返回的内容
2. 查看 🎯【Agent 最终回答】部分 - 这是 Agent 整理后的答案
3. 核心评估点（仅评估流程，不评判事实准确性）：
   - Agent 是否正确调用了搜索工具？
   - Agent 是否基于搜索结果进行了合理整理？
   - Agent 是否按要求的格式输出？
   - ⚠️ 不要评判搜索结果的事实准确性（如日期、新闻真实性等）！

【评分标准】（仅评估流程和完成度）
- 1.0: 完美流程，正确调用搜索工具，基于返回内容合理整理，格式规范
- 0.8-0.9: 优秀流程，调用搜索正确，整理基本合理，格式清晰
- 0.6-0.7: 良好流程，调用了搜索，但整理有遗漏或格式问题
- 0.4-0.5: 及格流程，尝试搜索但整合不完整
- 0.2-0.3: 差流程，搜索方法错误或未能整理信息
- 0.0-0.1: 极差流程，未调用搜索工具或完全不理解任务

请输出：
1. 分数（0.0-1.0 之间的小数，保留一位小数）
2. 评分理由（150-250字，必须说明：①是否正确调用搜索 ②如何整理搜索结果 ③流程的优缺点）

输出格式：
分数: 0.X
理由: [你的评分理由]
"""
        else:
            # RAG 任务：评估内容准确性
            scoring_prompt = f"""你是一个严格的 AI Agent 评估专家。请根据以下信息对 Agent 的回答质量进行评分。

【任务类型】
{result['case_type']}

【用户问题】
{result['user_input']}

【评估要点】
{result['evaluation_points']}

【Agent 完整执行轨迹】（包含工具调用过程和返回的内容）
{result.get('full_trajectory', result['agent_output'])}

【重要评估规则】
⚠️ 请仔细查看上面的【Agent 完整执行轨迹】：
1. 查看 [Tool Result: xxx] 部分 - 这是知识库/工具实际返回的内容
2. 查看 [Final Output] 部分 - 这是 Agent 的最终回答
3. 核心评估点：Agent 的最终回答是否基于工具返回的内容？
   - 如果 Agent 回答的内容在工具返回结果中有依据 → 分数高
   - 比如："根据《RoboMaster 2026 机甲大师高校联盟赛比赛规则手册》相关内容，英雄、步兵、工程三类机器人的对比分析如下：  1. **裁判系统模块与武器规格**  
   - **英雄机器人**：必须安装主控模块、装甲模块及超级电容管理模块（共3个核心模块），允许配置1个42mm发射机构（武器规格上限）。  
   - **步兵机器人**：需安装主控模块和测速模块（共2个核心模块），允许安装近战武器（如刀刃）或辅助武器（如激光指示器）。  
   - **工程机器人**：需安装主控模块和任务执行模块（如搬运臂或搭建模块），禁止安装攻击性武器，武器规格以任务需求为准。 " 评分0.9


【评分标准】
- 1.0: 完美回答，完全基于工具返回内容，满足所有评估要点，信息准确、完整
- 0.8-0.9: 优秀回答，基本基于工具返回内容，满足大部分评估要点，少量合理推理
- 0.6-0.7: 良好回答，部分基于工具内容，但有遗漏或格式问题
- 0.4-0.5: 及格回答，尝试使用工具但整合不当，或有较多不准确推理
- 0.2-0.3: 差回答，大量编造工具未提供的信息，或严重偏离任务
- 0.0-0.1: 极差回答，完全编造内容或未使用工具

请输出：
1. 分数（0.0-1.0 之间的小数，保留一位小数）
2. 评分理由（150-250字，必须说明：①工具返回了什么 ②Agent如何使用这些内容 ③哪些是有依据的，哪些是编造的）

输出格式：
分数: 0.X
理由: [你的评分理由]
"""
        
        try:
            print(f"\n⏳ 正在请求 DeepSeek 进行评分...")
            response = self.deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一个严格的 AI Agent 评估专家。"},
                    {"role": "user", "content": scoring_prompt}
                ],
                stream=False
            )
            response_text = response.choices[0].message.content.strip()
            
            # 解析响应
            lines = response_text.split('\n')
            score = 0.0
            reason = ""
            
            for line in lines:
                if line.startswith('分数:') or line.startswith('Score:'):
                    score_str = line.split(':', 1)[1].strip()
                    # 提取数字
                    import re
                    match = re.search(r'(\d+\.\d+|\d+)', score_str)
                    if match:
                        score = float(match.group(1))
                elif line.startswith('理由:') or line.startswith('Reason:'):
                    reason = line.split(':', 1)[1].strip()
                elif reason:  # 继续添加多行理由
                    reason += '\n' + line
            
            # 限制分数范围
            score = max(0.0, min(1.0, score))
            
            print(f"📊 DeepSeek 评分: {score:.1f}")
            print(f"💬 评分理由: {reason}\n")
            
            logger.info(f"DeepSeek score: {score:.1f}")
            
            # 如果是在线搜索任务，请求人工评分
            if is_web_search:
                human_score = self.get_human_score(result)
                if human_score is not None:
                    final_score = (score + human_score) / 2.0
                    print(f"🎯 最终分数: {final_score:.2f} (AI: {score:.1f} + 人工: {human_score:.1f})")
                    logger.info(f"Final score: {final_score:.2f} (AI: {score:.1f}, Human: {human_score:.1f})")
                    return final_score, f"AI评分({score:.1f}): {reason}\n人工评分: {human_score:.1f}"
            
            return score, reason
            
        except Exception as e:
            logger.error(f"DeepSeek scoring error: {e}")
            return 0.0, f"Scoring error: {str(e)}"
    
    def get_human_score(self, result: Dict[str, Any]) -> float:
        """
        获取人工评分
        
        Args:
            result: 测试结果
            
        Returns:
            人工评分 (0.0-1.0)，如果跳过则返回 None
        """
        print("\n" + "="*70)
        print("📝 请进行人工评分 (基于搜索结果的事实准确性)")
        print("="*70)
        print(f"任务: {result['user_input'][:100]}...")
        print(f"\n请评估搜索结果的准确性（0.0-1.0）:")
        print("  - 1.0: 搜索结果完全准确、时效性强")
        print("  - 0.8: 搜索结果基本准确")
        print("  - 0.6: 搜索结果部分准确")
        print("  - 0.4: 搜索结果准确性一般")
        print("  - 0.2: 搜索结果多数不准确")
        print("  - 0.0: 搜索结果完全不准确")
        print("  - 输入 's' 跳过人工评分\n")
        
        while True:
            try:
                user_input = input("请输入分数 (0.0-1.0 或 's' 跳过): ").strip()
                
                if user_input.lower() == 's':
                    print("⏭️  跳过人工评分")
                    return None
                
                score = float(user_input)
                if 0.0 <= score <= 1.0:
                    logger.info(f"Human score received: {score:.1f}")
                    return score
                else:
                    print("❌ 分数必须在 0.0-1.0 之间，请重新输入")
            except ValueError:
                print("❌ 输入无效，请输入数字或 's'")
            except KeyboardInterrupt:
                print("\n⏭️  跳过人工评分")
                return None
    
    def run_all_tests(self, test_file: str, max_cases: int = None):
        """
        运行所有测试案例
        
        Args:
            test_file: 测试案例文件路径
            max_cases: 最大测试案例数（None 表示全部）
        """
        # 加载测试案例
        test_cases = self.load_test_cases(test_file)
        
        if max_cases:
            test_cases = test_cases[:max_cases]
        
        print(f"\n{'='*70}")
        print(f"  📋 通用能力测试")
        print(f"{'='*70}")
        print(f"测试文件: {test_file}")
        print(f"测试案例数: {len(test_cases)}")
        print(f"结果保存: {self.session_dir}")
        print(f"{'='*70}\n")
        
        # 运行测试
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n进度: [{i}/{len(test_cases)}]")
            
            # 运行测试案例
            result = self.run_test_case(test_case)
            
            # 使用 F1 分数评分（代替 DeepSeek）
            if not result['error']:
                score, reason = self.calculate_f1_score(result)
                result['score'] = score
                result['score_reason'] = reason
                
                print(f"\n📊 F1 评分: {score:.2%}")
                print(f"💬 评分说明:\n{reason}")
            else:
                result['score'] = 0.0
                result['score_reason'] = f"Execution error: {result['error']}"
            
            self.results.append(result)
        
        # 保存结果
        self.save_results()
        
        # 打印总结
        self.print_summary()
    
    def save_results(self):
        """保存测试结果"""
        # 保存详细结果（JSON）
        detailed_file = self.session_dir / "detailed_results.json"
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"\n✅ 详细结果已保存: {detailed_file}")
        
        # 保存汇总（TXT）
        summary_file = self.session_dir / "summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("通用能力测试汇总\n")
            f.write("=" * 70 + "\n\n")
            
            total_cases = len(self.results)
            avg_score = sum(r['score'] for r in self.results) / total_cases if total_cases > 0 else 0
            avg_time = sum(r['elapsed_time'] for r in self.results) / total_cases if total_cases > 0 else 0
            
            f.write(f"总测试案例数: {total_cases}\n")
            f.write(f"平均分数: {avg_score:.2f}\n")
            f.write(f"平均响应时间: {avg_time:.1f}秒\n\n")
            
            # 按任务类型分组
            by_type = {}
            for r in self.results:
                case_type = r['case_type']
                if case_type not in by_type:
                    by_type[case_type] = []
                by_type[case_type].append(r)
            
            f.write("-" * 70 + "\n")
            f.write("按任务类型统计\n")
            f.write("-" * 70 + "\n")
            for case_type, cases in by_type.items():
                type_avg_score = sum(c['score'] for c in cases) / len(cases)
                f.write(f"\n{case_type}:\n")
                f.write(f"  案例数: {len(cases)}\n")
                f.write(f"  平均分: {type_avg_score:.2f}\n")
            
            f.write("\n" + "-" * 70 + "\n")
            f.write("各案例详情\n")
            f.write("-" * 70 + "\n")
            for r in self.results:
                f.write(f"\n案例 #{r['case_id']} ({r['case_type']})\n")
                f.write(f"  分数: {r['score']:.2f}\n")
                f.write(f"  用时: {r['elapsed_time']:.1f}秒\n")
                if r['error']:
                    f.write(f"  错误: {r['error']}\n")
                else:
                    f.write(f"  评分理由: {r['score_reason'][:100]}...\n")
        
        print(f"✅ 汇总报告已保存: {summary_file}")
    
    def print_summary(self):
        """打印测试总结"""
        total_cases = len(self.results)
        avg_score = sum(r['score'] for r in self.results) / total_cases if total_cases > 0 else 0
        avg_time = sum(r['elapsed_time'] for r in self.results) / total_cases if total_cases > 0 else 0
        
        print(f"\n\n{'='*70}")
        print("📊 测试总结 (F1 Score)")
        print(f"{'='*70}")
        print(f"总测试案例数: {total_cases}")
        print(f"平均 F1 分数: {avg_score:.2%} (内容匹配度)")
        print(f"平均响应时间: {avg_time:.1f}秒")
        
        # 按分数段统计
        excellent = sum(1 for r in self.results if r['score'] >= 0.8)
        good = sum(1 for r in self.results if 0.6 <= r['score'] < 0.8)
        fair = sum(1 for r in self.results if 0.4 <= r['score'] < 0.6)
        poor = sum(1 for r in self.results if r['score'] < 0.4)
        
        print(f"\nF1 分数分布:")
        print(f"  🌟 优秀 (≥80%): {excellent} 个")
        print(f"  👍 良好 (60-80%): {good} 个")
        print(f"  📖 及格 (40-60%): {fair} 个")
        print(f"  ⚠️  不及格 (<40%): {poor} 个")
        
        # 显示各案例分数
        print(f"\n各案例 F1 分数:")
        for r in self.results:
            status = "✅" if r['score'] >= 0.6 else "⚠️"
            print(f"  {status} 案例 #{r['case_id']}: {r['score']:.2%}")
        
        print(f"{'='*70}\n")


def test_general_capability(agent: Agent, test_type: str = "both", max_cases: int = None, rag_file: str = None):
    """
    测试 Agent 的通用能力
    
    Args:
        agent: Agent 实例
        test_type: 测试类型 ("rag", "web", "both")
        max_cases: 每个类型最多测试多少个案例
        rag_file: 指定 RAG 测试文件路径（可选）
    """
    tester = GeneralCapabilityTester(agent)
    
    if test_type in ["rag", "both"]:
        # 如果没有指定文件，尝试使用默认文件
        if rag_file is None:
            # 优先使用 AI 测试集
            if os.path.exists("benchmark/RAG/test_cases_RAG_AI.txt"):
                rag_file = "benchmark/RAG/test_cases_RAG_AI.txt"
            elif os.path.exists("benchmark/RAG/test_cases_RAG.txt"):
                rag_file = "benchmark/RAG/test_cases_RAG.txt"
        
        if rag_file and os.path.exists(rag_file):
            print(f"\n{'#'*70}")
            print(f"  🔍 开始 RAG 测试 - {os.path.basename(rag_file)}")
            print(f"{'#'*70}")
            tester.run_all_tests(rag_file, max_cases=max_cases)
        else:
            print(f"⚠️  RAG 测试文件不存在: {rag_file}")
    
    if test_type in ["web", "both"]:
        web_file = "benchmark/web/test_cases_web.txt"
        if os.path.exists(web_file):
            print(f"\n{'#'*70}")
            print("  🌐 开始 Web Search 测试")
            print(f"{'#'*70}")
            tester.run_all_tests(web_file, max_cases=max_cases)
        else:
            print(f"⚠️  Web 测试文件不存在: {web_file}")
