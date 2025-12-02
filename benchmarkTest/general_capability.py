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
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if api_key:
                self.deepseek_client = OpenAI(
                    api_key=api_key,
                    base_url="https://api.deepseek.com"
                )
                logger.info("DeepSeek API initialized")
            else:
                logger.warning("DEEPSEEK_API_KEY not set. Scoring will be disabled.")
        
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
            
            for line in lines:
                line = line.strip()
                if line.startswith('测试案例 #'):
                    if current_case:
                        test_cases.append(current_case)
                    current_case = {}
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
                elif line and mode == 'description':
                    current_case['description'].append(line)
                elif line and mode == 'user_input':
                    current_case['user_input'].append(line)
                elif line and mode == 'evaluation':
                    current_case['evaluation_points'].append(line)
        
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
                # 工具执行结果
                tool_name = msg.get('name', 'unknown_tool')
                trajectory_parts.append(f"[Tool Result: {tool_name}]\n{content}\n")
        
        # 添加最终输出
        trajectory_parts.append(f"[Final Output]\n{final_output}\n")
        
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
                'error': error_msg
            }
        
        return result
    
    def score_with_deepseek(self, result: Dict[str, Any], is_web_search: bool = False) -> Tuple[float, str]:
        """
        使用 DeepSeek API 对回答进行评分
        
        Args:
            result: 测试结果
            is_web_search: 是否为在线搜索任务
            
        Returns:
            (分数, 评分理由)
        """
        if not self.deepseek_client:
            logger.warning("DeepSeek client not available, skipping scoring")
            return 0.0, "DeepSeek API not available"
        
        if result['error']:
            return 0.0, f"Agent execution error: {result['error']}"
        
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

【Agent 完整执行轨迹】（包含工具调用过程）
{result.get('full_trajectory', result['agent_output'])}

【重要说明】
-  不要评判搜索结果的事实准确性（如日期、新闻内容真实性等）！
-  只评估 Agent 是否正确理解任务、使用了搜索工具、按要求整理了信息！！！

【评分标准】（仅评估流程和完成度）
- 1.0: 完美完成任务流程，正确使用搜索工具，信息整理完整规范
- 0.8-0.9: 优秀，基本完成任务流程，搜索和整理方法正确
- 0.6-0.7: 良好，完成了主要流程，但信息整理有遗漏或格式问题
- 0.4-0.5: 及格，理解任务并尝试搜索，但执行不完整
- 0.2-0.3: 差，部分理解任务但搜索方法错误或未整理信息
- 0.0-0.1: 极差，未理解任务或未使用搜索工具

请输出：
1. 分数（0.0-1.0 之间的小数，保留一位小数）
2. 评分理由（100-200字，说明流程执行情况、优点和不足）

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

【Agent 完整执行轨迹】（包含工具调用过程）
{result.get('full_trajectory', result['agent_output'])}

【评分标准】
- 1.0: 完美回答，完全满足所有评估要点，信息准确、完整、来源可靠
- 0.8-0.9: 优秀回答，满足大部分评估要点，信息基本准确完整
- 0.6-0.7: 良好回答，满足部分评估要点，但有明显遗漏或不足
- 0.4-0.5: 及格回答，基本理解任务但执行不完整或有较多错误
- 0.2-0.3: 差回答，严重偏离任务要求或信息错误较多
- 0.0-0.1: 极差回答，完全未理解任务或未能提供有效信息

请输出：
1. 分数（0.0-1.0 之间的小数，保留一位小数）
2. 评分理由（100-200字，说明优点、不足和改进建议）

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
            
            # 使用 DeepSeek 评分
            if not result['error']:
                # 判断是否为在线搜索任务（包含：web、search、搜索、谷歌等关键词）
                case_type_lower = result['case_type'].lower()
                is_web_search = any(keyword in case_type_lower for keyword in ['web', 'search', '搜索', '谷歌', 'google'])
                score, reason = self.score_with_deepseek(result, is_web_search=is_web_search)
                result['score'] = score
                result['score_reason'] = reason
            else:
                result['score'] = 0.0
                result['score_reason'] = f"Execution error: {result['error']}"
            
            self.results.append(result)
            
            # 每个案例后暂停，避免 API 限流
            if i < len(test_cases):
                time.sleep(2)
        
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
        print("📊 测试总结")
        print(f"{'='*70}")
        print(f"总测试案例数: {total_cases}")
        print(f"平均分数: {avg_score:.2f} / 1.00")
        print(f"平均响应时间: {avg_time:.1f}秒")
        
        # 按分数段统计
        excellent = sum(1 for r in self.results if r['score'] >= 0.8)
        good = sum(1 for r in self.results if 0.6 <= r['score'] < 0.8)
        fair = sum(1 for r in self.results if 0.4 <= r['score'] < 0.6)
        poor = sum(1 for r in self.results if r['score'] < 0.4)
        
        print(f"\n分数分布:")
        print(f"  🌟 优秀 (≥0.8): {excellent} 个")
        print(f"  👍 良好 (0.6-0.8): {good} 个")
        print(f"  📖 及格 (0.4-0.6): {fair} 个")
        print(f"  ⚠️  不及格 (<0.4): {poor} 个")
        print(f"{'='*70}\n")


def test_general_capability(agent: Agent, test_type: str = "both", max_cases: int = None):
    """
    测试 Agent 的通用能力
    
    Args:
        agent: Agent 实例
        test_type: 测试类型 ("rag", "web", "both")
        max_cases: 每个类型最多测试多少个案例
    """
    tester = GeneralCapabilityTester(agent)
    
    if test_type in ["rag", "both"]:
        rag_file = "benchmark/RAG/test_cases_RAG.txt"
        if os.path.exists(rag_file):
            print(f"\n{'#'*70}")
            print("  🔍 开始 RAG 测试")
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
