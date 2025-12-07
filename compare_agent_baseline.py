"""
消融实验结果对比脚本
读取已完成的 Agent 和 Baseline 测试结果，进行对比分析

使用方法:
  python compare_agent_baseline.py --agent benchmark_results/.../session_xxx --baseline benchmark_results/.../session_yyy
  
  或者指定目录，自动找最新的结果：
  python compare_agent_baseline.py --agent_dir benchmark_results/general_capability --baseline_dir benchmark_results/baseline
"""

import argparse
import os
import sys
import json
from pathlib import Path
from datetime import datetime
from loguru import logger


def load_test_results(session_dir: str) -> dict:
    """
    运行 Agent vs Baseline 对比测试
    
    Args:
        test_type: 测试类型 ("rag", "web", "both")
        max_cases: 最大测试案例数
        rag_file: RAG 测试文件路径
        gpu_ids: GPU ID 列表
    """
    print("\n" + "=" * 70)
    print("  📊 消融实验：Full Agent vs Baseline")
    print("=" * 70)
    
    # 加载模型
    print("\n🔄 正在加载模型...")
    if gpu_ids:
        llm = Qwen3VL(path="Qwen3-8B", gpu_ids=gpu_ids)
    else:
        llm = Qwen3VL(path="Qwen3-8B")
    print("✅ 模型加载成功\n")
    
    # 检查 RAG 数据库
    rag_db_path = None
    if os.path.exists("rag/wiki_vector_db"):
        rag_db_path = "rag/wiki_vector_db"
    elif os.path.exists("rag/vector_db"):
        rag_db_path = "rag/vector_db"
    
    # 确定测试文件
    if test_type in ["rag", "both"]:
        if rag_file is None:
            if os.path.exists("benchmark/RAG/test_cases_RAG_AI.txt"):
                rag_file = "benchmark/RAG/test_cases_RAG_AI.txt"
            elif os.path.exists("benchmark/RAG/test_cases_RAG.txt"):
                rag_file = "benchmark/RAG/test_cases_RAG.txt"
    
    results = {}
    
    # ==================== 测试 Full Agent ====================
    print("\n" + "#" * 70)
    print("  🤖 第一轮：测试 Full Agent（完整框架 + ReAct + 工具）")
    print("#" * 70)
    
    # 设置工具限制
    allowed_tools = None
    if test_type == 'rag':
        allowed_tools = ['knowledge_base_query']
        print("🔒 工具限制：仅使用 knowledge_base_query")
    elif test_type == 'web':
        allowed_tools = ['google_search', 'tavily_search', 'get_time']
        print("🔒 工具限制：仅使用搜索工具")
    
    agent = Agent(llm, rag_db_path=rag_db_path, allowed_tools=allowed_tools)
    agent_tester = GeneralCapabilityTester(agent, result_dir="benchmark_results/comparison/agent")
    
    if test_type in ["rag", "both"] and rag_file:
        print(f"\n📝 测试文件: {rag_file}")
        agent_tester.run_all_tests(rag_file, max_cases=max_cases)
        results['agent'] = agent_tester.results
    
    # ==================== 测试 Baseline ====================
    print("\n" + "#" * 70)
    print("  🎯 第二轮：测试 Baseline（纯 LLM，无框架，无工具）")
    print("#" * 70)
    
    baseline = BaselineAgent(llm)
    baseline_tester = GeneralCapabilityTester(baseline, result_dir="benchmark_results/comparison/baseline")
    
    if test_type in ["rag", "both"] and rag_file:
        print(f"\n📝 测试文件: {rag_file}")
        baseline_tester.run_all_tests(rag_file, max_cases=max_cases)
        results['baseline'] = baseline_tester.results
    
    # ==================== 对比分析 ====================
    print("\n" + "=" * 70)
    print("  📊 对比分析报告")
    print("=" * 70)
    
    if 'agent' in results and 'baseline' in results:
        agent_scores = [r['score'] for r in results['agent']]
        baseline_scores = [r['score'] for r in results['baseline']]
        
        agent_avg = sum(agent_scores) / len(agent_scores) if agent_scores else 0
        baseline_avg = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0
        
        improvement = agent_avg - baseline_avg
        improvement_percent = (improvement / baseline_avg * 100) if baseline_avg > 0 else 0
        
        print(f"\n【整体性能】")
        print(f"  Full Agent 平均 F1:   {agent_avg:.2%}")
        print(f"  Baseline 平均 F1:     {baseline_avg:.2%}")
        print(f"  性能提升:             {improvement:+.2%} ({improvement_percent:+.1f}%)")
        
        # 详细对比
        print(f"\n【逐案例对比】")
        print(f"  {'案例ID':<10} {'Agent F1':<12} {'Baseline F1':<12} {'差异':<10}")
        print("  " + "-" * 50)
        
        for agent_r, baseline_r in zip(results['agent'], results['baseline']):
            case_id = agent_r['case_id']
            agent_score = agent_r['score']
            baseline_score = baseline_r['score']
            diff = agent_score - baseline_score
            
            symbol = "✅" if diff > 0 else "⚠️" if diff < 0 else "="
            print(f"  {case_id:<10} {agent_score:>10.2%} {baseline_score:>10.2%} {diff:>+9.2%} {symbol}")
        
        # 胜负统计
        agent_wins = sum(1 for a, b in zip(agent_scores, baseline_scores) if a > b)
        baseline_wins = sum(1 for a, b in zip(agent_scores, baseline_scores) if a < b)
        ties = sum(1 for a, b in zip(agent_scores, baseline_scores) if a == b)
        
        print(f"\n【胜负统计】")
        print(f"  Agent 胜出: {agent_wins} 个")
        print(f"  Baseline 胜出: {baseline_wins} 个")
        print(f"  平局: {ties} 个")
        
        # 保存对比结果
        comparison_result = {
            'timestamp': datetime.now().isoformat(),
            'test_type': test_type,
            'max_cases': max_cases,
            'rag_file': rag_file,
            'agent': {
                'avg_f1': agent_avg,
                'scores': agent_scores
            },
            'baseline': {
                'avg_f1': baseline_avg,
                'scores': baseline_scores
            },
            'improvement': improvement,
            'improvement_percent': improvement_percent,
            'wins': {
                'agent': agent_wins,
                'baseline': baseline_wins,
                'ties': ties
            }
        }
        
        # 保存到文件
        comparison_dir = Path("benchmark_results/comparison")
        comparison_dir.mkdir(parents=True, exist_ok=True)
        comparison_file = comparison_dir / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_result, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 对比结果已保存: {comparison_file}")
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="消融实验：对比 Agent 和 Baseline")
    parser.add_argument(
        '--type',
        type=str,
        choices=['rag', 'web', 'both'],
        default='rag',
        help='测试类型: rag (RAG测试), web (Web搜索测试), both (两者都测)'
    )
    parser.add_argument(
        '--max_cases',
        type=int,
        default=None,
        help='每个类型最多测试多少个案例（None 表示全部）'
    )
    parser.add_argument(
        '--rag_file',
        type=str,
        default=None,
        help='指定 RAG 测试文件路径'
    )
    parser.add_argument(
        '--gpu_ids',
        type=str,
        default='4,5,6,7',
        help='使用的GPU编号，用逗号分隔，如 "4,5,6,7"'
    )
    
    args = parser.parse_args()
    
    # 初始化日志
    setup_logger()
    
    # 解析 GPU IDs
    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
    
    try:
        run_comparison(
            test_type=args.type,
            max_cases=args.max_cases,
            rag_file=args.rag_file,
            gpu_ids=gpu_ids
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  测试被用户中断")
        sys.exit(0)
    except Exception as e:
        logger.error(f"测试过程出错: {e}")
        print(f"\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
