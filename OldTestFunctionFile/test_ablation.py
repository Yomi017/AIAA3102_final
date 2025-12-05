"""
消融实验脚本
直接使用 Qwen3-8B 模型（无 Agent 框架），用于对比测试

对比项：
1. Baseline（本脚本）: 直接 LLM，无工具调用，无 ReAct
2. Full Agent: 完整 Agent 系统（test_general_capability.py）
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from loguru import logger

from llm import Qwen3VL
from log_config import setup_logger


# 尝试导入 OpenAI (用于 DeepSeek API)
try:
    from openai import OpenAI
    DEEPSEEK_AVAILABLE = True
except ImportError:
    DEEPSEEK_AVAILABLE = False
    logger.warning("OpenAI SDK not installed. Install with: pip install openai")


class AblationTester:
    """消融实验测试器 - 直接使用 LLM"""
    
    def __init__(self, llm, result_dir: str = "benchmark_results/ablation"):
        """
        初始化消融测试器
        
        Args:
            llm: LLM 实例（无 Agent）
            result_dir: 结果保存目录
        """
        self.llm = llm
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成时间戳
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.result_dir / f"session_{self.timestamp}"
        self.session_dir.mkdir(exist_ok=True)
        
        self.results = []
        
        # 初始化 DeepSeek
        self.deepseek_client = None
        if DEEPSEEK_AVAILABLE:
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if api_key:
                self.deepseek_client = OpenAI(
                    api_key=api_key,
                    base_url="https://api.deepseek.com"
                )
                logger.info("DeepSeek API initialized for ablation scoring")
            else:
                logger.warning("DEEPSEEK_API_KEY not set. Scoring will be disabled.")
        
        # 简单的系统提示（无工具）
        self.system_prompt = """你是一个有帮助的 AI 助手。请根据你的知识回答用户的问题。

注意：
- 尽你所能提供准确、详细的回答
- 如果不确定，请说明你不确定
- 回答要有条理、清晰
"""
    
    def load_test_cases(self, file_path: str) -> List[Dict[str, Any]]:
        """
        加载测试案例文件（与 general_capability 相同的格式）
        
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
        mode = None
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
    
    def _remove_think_tags(self, text: str) -> str:
        """
        移除响应中的 <think> 标签
        
        Args:
            text: 原始响应文本
            
        Returns:
            移除 <think> 标签后的文本
        """
        # 查找 <think> 和 </think> 标签
        think_start = text.find('<think>')
        think_end = text.find('</think>')
        
        if think_start == -1 and think_end != -1:
            think_start = 0  # 只有结束标签，视为从开头开始
        
        if think_start != -1 and think_end != -1 and think_start < think_end:
            # 移除思考标签和内容
            clean_text = text[:think_start] + text[think_end + len('</think>'):]
            return clean_text.strip()
        
        return text
    
    def score_with_deepseek(self, result: Dict) -> Dict:
        """
        使用 DeepSeek API 对结果进行评分
        
        Args:
            result: 包含问题和响应的结果字典
            
        Returns:
            包含 score 和 score_reason 的字典
        """
        if not self.deepseek_client:
            logger.warning("DeepSeek client not available, skipping scoring")
            return {"score": None, "score_reason": "DeepSeek API not available"}
        
        # 使用正确的字段名
        question = result.get("user_input", "")
        response = result.get("llm_output", "")
        case_type = result.get("case_type", "未知类型")
        
        # 评分提示词（针对无工具的基线模型）
        scoring_prompt = f"""你是一个评估助手，需要评估 AI 模型对问题的回答质量。

**问题类型**: {case_type}
**问题**: {question}

**AI 的回答**: 
{response}

**评估要求**:
- 这是一个**无工具的基线模型**测试，模型只能基于训练知识回答，无法使用外部工具
- 评估回答的相关性、完整性和有用性
- 对于需要实时信息或外部工具的问题，如果模型明确说明限制或给出合理的替代建议，应给予一定分数
- 对于可以通过知识回答的问题，评估答案的准确性和详细程度

**评分标准**（0.0-1.0）:
- 1.0: 完全正确且详细的回答
- 0.7-0.9: 回答基本正确，但可能缺少细节或有小错误
- 0.4-0.6: 部分正确，或明确说明了限制并给出替代建议
- 0.1-0.3: 回答不相关或有严重错误
- 0.0: 完全无用或拒绝回答

请直接给出评分（0.0-1.0之间的小数）和评分理由。

格式：
评分: [分数]
理由: [详细理由]"""

        try:
            response_obj = self.deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": scoring_prompt}],
                temperature=0.3,
                max_tokens=500
            )
            
            score_text = response_obj.choices[0].message.content.strip()
            
            # 解析评分
            score = 0.0
            score_reason = score_text
            
            lines = score_text.split('\n')
            for line in lines:
                if line.startswith('评分:') or line.startswith('评分：'):
                    score_str = line.split(':', 1)[1].strip()
                    score_str = score_str.split('：', 1)[1].strip() if '：' in score_str else score_str
                    try:
                        score = float(score_str)
                    except ValueError:
                        logger.warning(f"Failed to parse score: {score_str}")
                        score = 0.0
                    break
            
            logger.info(f"DeepSeek score: {score}")
            
            return {
                "score": score,
                "score_reason": score_text
            }
            
        except Exception as e:
            logger.error(f"Error calling DeepSeek API: {e}")
            return {
                "score": None,
                "score_reason": f"Error: {str(e)}"
            }
    
    def run_test_case(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        运行单个测试案例（直接调用 LLM，无 Agent）
        
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
        
        logger.info(f"Running ablation test case #{case_id}")
        
        # 记录开始时间
        start_time = time.time()
        
        try:
            # 直接调用 LLM（无 Agent，无工具）
            response, _ = self.llm.chat(
                user_input,
                history=[],
                meta_instruction=self.system_prompt,
                images=None
            )
            
            # 移除 <think> 标签（仅保留实际输出）
            clean_response = self._remove_think_tags(response)
            
            elapsed_time = time.time() - start_time
            
            print(f"\n{'─'*70}")
            print(f"🤖 LLM 直接回答 (无工具):")
            print(f"{'─'*70}")
            print(clean_response)
            print(f"\n⏱️  响应时间: {elapsed_time:.1f}秒")
            
            result = {
                'case_id': case_id,
                'case_type': case_type,
                'user_input': user_input,
                'llm_output': clean_response,
                'elapsed_time': elapsed_time,
                'evaluation_points': test_case.get('evaluation_points', ''),
                'error': None
            }
            
            logger.info(f"Ablation test case #{case_id} completed in {elapsed_time:.1f}s")
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            error_msg = str(e)
            
            print(f"\n❌ 执行出错: {error_msg}")
            logger.error(f"Ablation test case #{case_id} error: {e}")
            
            result = {
                'case_id': case_id,
                'case_type': case_type,
                'user_input': user_input,
                'llm_output': None,
                'elapsed_time': elapsed_time,
                'evaluation_points': test_case.get('evaluation_points', ''),
                'error': error_msg
            }
        
        return result
    
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
        print(f"  📋 消融实验 - Baseline (无 Agent)")
        print(f"{'='*70}")
        print(f"测试文件: {test_file}")
        print(f"测试案例数: {len(test_cases)}")
        print(f"结果保存: {self.session_dir}")
        print(f"模式: 直接 LLM 调用（无工具、无 ReAct）")
        print(f"{'='*70}\n")
        
        # 运行测试
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n进度: [{i}/{len(test_cases)}]")
            
            # 运行测试案例
            result = self.run_test_case(test_case)
            
            # 使用 DeepSeek 评分
            if self.deepseek_client:
                print("\n⏳ 正在评分...")
                score_result = self.score_with_deepseek(result)
                result["score"] = score_result["score"]
                result["score_reason"] = score_result["score_reason"]
                
                if result["score"] is not None:
                    print(f"📊 DeepSeek 评分: {result['score']:.2f}")
                    print(f"理由: {score_result['score_reason'][:100]}...")
            else:
                result["score"] = None
                result["score_reason"] = "DeepSeek API 不可用"
            
            self.results.append(result)
            
            # 暂停一下
            if i < len(test_cases):
                time.sleep(1)
        
        # 保存结果
        self.save_results()
        
        # 打印总结
        self.print_summary()
    
    def save_results(self):
        """保存测试结果"""
        import json
        
        # 保存详细结果（JSON）
        detailed_file = self.session_dir / "ablation_results.json"
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"\n✅ 详细结果已保存: {detailed_file}")
        
        # 保存汇总（TXT）
        summary_file = self.session_dir / "summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("消融实验 - Baseline (无 Agent) 汇总\n")
            f.write("=" * 70 + "\n\n")
            
            total_cases = len(self.results)
            avg_time = sum(r['elapsed_time'] for r in self.results) / total_cases if total_cases > 0 else 0
            success_count = sum(1 for r in self.results if r['error'] is None)
            
            f.write(f"总测试案例数: {total_cases}\n")
            f.write(f"成功完成: {success_count}\n")
            f.write(f"失败: {total_cases - success_count}\n")
            f.write(f"平均响应时间: {avg_time:.1f}秒\n")
            
            # 评分统计
            scored_results = [r for r in self.results if r.get('score') is not None]
            if scored_results:
                avg_score = sum(r['score'] for r in scored_results) / len(scored_results)
                f.write(f"\n评分统计:\n")
                f.write(f"  已评分案例: {len(scored_results)}/{total_cases}\n")
                f.write(f"  平均分数: {avg_score:.3f}\n")
                f.write(f"  最高分: {max(r['score'] for r in scored_results):.3f}\n")
                f.write(f"  最低分: {min(r['score'] for r in scored_results):.3f}\n")
            else:
                f.write(f"\n评分统计: 无 (DeepSeek API 未配置)\n")
            
            f.write("\n")
            
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
                f.write(f"\n{case_type}:\n")
                f.write(f"  案例数: {len(cases)}\n")
                type_success = sum(1 for c in cases if c['error'] is None)
                f.write(f"  成功: {type_success}/{len(cases)}\n")
            
            f.write("\n" + "-" * 70 + "\n")
            f.write("各案例详情\n")
            f.write("-" * 70 + "\n")
            for r in self.results:
                f.write(f"\n案例 #{r['case_id']} ({r['case_type']})\n")
                f.write(f"  用时: {r['elapsed_time']:.1f}秒\n")
                if r['error']:
                    f.write(f"  状态: 失败\n")
                    f.write(f"  错误: {r['error']}\n")
                else:
                    f.write(f"  状态: 成功\n")
                    f.write(f"  回答长度: {len(r['llm_output'])} 字符\n")
                
                # 添加评分信息
                if r.get('score') is not None:
                    f.write(f"  评分: {r['score']:.3f}\n")
        
        print(f"✅ 汇总报告已保存: {summary_file}")
    
    def print_summary(self):
        """打印测试总结"""
        total_cases = len(self.results)
        avg_time = sum(r['elapsed_time'] for r in self.results) / total_cases if total_cases > 0 else 0
        success_count = sum(1 for r in self.results if r['error'] is None)
        
        # 计算评分统计
        scored_results = [r for r in self.results if r.get('score') is not None]
        avg_score = sum(r['score'] for r in scored_results) / len(scored_results) if scored_results else 0.0
        
        print(f"\n\n{'='*70}")
        print("📊 消融实验总结 - Baseline (无 Agent)")
        print(f"{'='*70}")
        print(f"总测试案例数: {total_cases}")
        print(f"成功完成: {success_count}")
        print(f"失败: {total_cases - success_count}")
        print(f"平均响应时间: {avg_time:.1f}秒")
        
        if scored_results:
            print(f"\n评分统计:")
            print(f"  已评分案例: {len(scored_results)}/{total_cases}")
            print(f"  平均分数: {avg_score:.3f}")
            print(f"  最高分: {max(r['score'] for r in scored_results):.3f}")
            print(f"  最低分: {min(r['score'] for r in scored_results):.3f}")
        else:
            print(f"\n⚠️  无评分数据 (DeepSeek API 未配置)")
        
        print(f"\n注意: 此为消融实验 Baseline，无工具调用能力")
        print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="消融实验 - Baseline 测试（无 Agent）")
    parser.add_argument(
        '--type',
        type=str,
        choices=['rag', 'web', 'both'],
        default='both',
        help='测试类型: rag (RAG测试), web (Web搜索测试), both (两者都测)'
    )
    parser.add_argument(
        '--max_cases',
        type=int,
        default=None,
        help='每个类型最多测试多少个案例（None 表示全部）'
    )
    parser.add_argument(
        '--gpu_ids',
        type=str,
        default=None,
        help='使用的GPU编号，用逗号分隔，如 "0,1,2,3"'
    )
    
    args = parser.parse_args()
    
    # 初始化日志
    setup_logger()
    
    print("=" * 70)
    print("  🧪 消融实验 - Baseline (无 Agent 框架)")
    print("=" * 70)
    print(f"\n测试类型: {args.type}")
    print("模式: 直接 LLM 调用（无工具、无 ReAct）")
    if args.max_cases:
        print(f"最大案例数: {args.max_cases}")
    
    # 解析 GPU IDs
    gpu_ids = [4,5,6,7]  # 使用 GPU 4,5,6,7（避免与其他任务冲突）
    if args.gpu_ids:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
        print(f"GPU: {gpu_ids}")
    else:
        print(f"GPU: {gpu_ids} (默认)")
    
    print("=" * 70)
    
    # 加载模型
    print("\n🔄 正在加载模型...")
    try:
        if gpu_ids:
            llm = Qwen3VL(path="Qwen3-8B", gpu_ids=gpu_ids)
        else:
            llm = Qwen3VL(path="Qwen3-8B")
        print("✅ 模型加载成功\n")
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        print(f"❌ 模型加载失败: {e}")
        sys.exit(1)
    
    # 创建消融测试器
    print("🔄 正在初始化消融测试器...")
    tester = AblationTester(llm)
    print("✅ 测试器初始化成功\n")
    
    # 运行测试
    try:
        if args.type in ["rag", "both"]:
            rag_file = "benchmark/RAG/test_cases_RAG.txt"
            if os.path.exists(rag_file):
                print(f"\n{'#'*70}")
                print("  🔍 开始 RAG 消融测试 (Baseline)")
                print(f"{'#'*70}")
                tester.run_all_tests(rag_file, max_cases=args.max_cases)
            else:
                print(f"⚠️  RAG 测试文件不存在: {rag_file}")
        
        if args.type in ["web", "both"]:
            web_file = "benchmark/web/test_cases_web.txt"
            if os.path.exists(web_file):
                # 如果同时测试两种类型，创建新的 tester 实例
                if args.type == "both":
                    tester = AblationTester(llm)
                
                print(f"\n{'#'*70}")
                print("  🌐 开始 Web Search 消融测试 (Baseline)")
                print(f"{'#'*70}")
                tester.run_all_tests(web_file, max_cases=args.max_cases)
            else:
                print(f"⚠️  Web 测试文件不存在: {web_file}")
                
    except KeyboardInterrupt:
        print("\n\n⚠️  测试被用户中断")
        sys.exit(0)
    except Exception as e:
        logger.error(f"测试过程出错: {e}")
        print(f"\n❌ 测试出错: {e}")
        sys.exit(1)
    
    print("\n✅ 消融实验完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
