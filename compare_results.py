"""
消融实验结果对比脚本
读取已完成的 Agent 和 Baseline 测试结果，进行对比分析

使用方法:
  python compare_results.py \
    --agent benchmark_results/agent/session_20251205_123456 \
    --baseline benchmark_results/baseline/session_20251205_234567
"""

import argparse
import json
from pathlib import Path
from datetime import datetime


def load_results(session_dir: str) -> list:
    """加载测试结果"""
    result_file = Path(session_dir) / "detailed_results.json"
    if not result_file.exists():
        raise FileNotFoundError(f"结果文件不存在: {result_file}")
    
    with open(result_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def compare_results(agent_results: list, baseline_results: list):
    """对比两组测试结果"""
    
    print("\n" + "=" * 70)
    print("  📊 消融实验对比分析")
    print("=" * 70)
    
    # 整体统计
    agent_scores = [r['score'] for r in agent_results]
    baseline_scores = [r['score'] for r in baseline_results]
    
    agent_avg = sum(agent_scores) / len(agent_scores) if agent_scores else 0
    baseline_avg = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0
    
    improvement = agent_avg - baseline_avg
    improvement_percent = (improvement / baseline_avg * 100) if baseline_avg > 0 else 0
    
    print(f"\n【整体性能】")
    print(f"  测试案例数:           {len(agent_results)}")
    print(f"  Full Agent 平均 F1:   {agent_avg:.2%}")
    print(f"  Baseline 平均 F1:     {baseline_avg:.2%}")
    print(f"  性能提升:             {improvement:+.2%} ({improvement_percent:+.1f}%)")
    
    # 逐案例对比
    print(f"\n【逐案例对比】")
    print(f"  {'案例ID':<12} {'Agent F1':<12} {'Baseline F1':<12} {'差异':<10}")
    print("  " + "-" * 55)
    
    for agent_r, baseline_r in zip(agent_results, baseline_results):
        case_id = agent_r['case_id']
        agent_score = agent_r['score']
        baseline_score = baseline_r['score']
        diff = agent_score - baseline_score
        
        symbol = "✅" if diff > 0 else "⚠️" if diff < 0 else "="
        print(f"  {case_id:<12} {agent_score:>10.2%}  {baseline_score:>10.2%}  {diff:>+9.2%} {symbol}")
    
    # 胜负统计
    agent_wins = sum(1 for a, b in zip(agent_scores, baseline_scores) if a > b)
    baseline_wins = sum(1 for a, b in zip(agent_scores, baseline_scores) if a < b)
    ties = sum(1 for a, b in zip(agent_scores, baseline_scores) if a == b)
    
    print(f"\n【胜负统计】")
    print(f"  Agent 胜出:    {agent_wins} 个 ({agent_wins/len(agent_results)*100:.1f}%)")
    print(f"  Baseline 胜出: {baseline_wins} 个 ({baseline_wins/len(baseline_results)*100:.1f}%)")
    print(f"  平局:          {ties} 个 ({ties/len(agent_results)*100:.1f}%)")
    
    print("=" * 70)
    
    return {
        'agent_avg': agent_avg,
        'baseline_avg': baseline_avg,
        'improvement': improvement,
        'improvement_percent': improvement_percent,
        'wins': {'agent': agent_wins, 'baseline': baseline_wins, 'ties': ties}
    }


def main():
    parser = argparse.ArgumentParser(description="对比 Agent 和 Baseline 测试结果")
    parser.add_argument(
        '--agent',
        type=str,
        required=True,
        help='Agent 测试结果目录路径（session_xxx 目录）'
    )
    parser.add_argument(
        '--baseline',
        type=str,
        required=True,
        help='Baseline 测试结果目录路径（session_xxx 目录）'
    )
    
    args = parser.parse_args()
    
    try:
        print("📂 加载测试结果...")
        print(f"  Agent:    {args.agent}")
        print(f"  Baseline: {args.baseline}")
        
        agent_results = load_results(args.agent)
        baseline_results = load_results(args.baseline)
        
        if len(agent_results) != len(baseline_results):
            print(f"⚠️  警告: 测试案例数量不同 (Agent: {len(agent_results)}, Baseline: {len(baseline_results)})")
        
        compare_results(agent_results, baseline_results)
        
    except FileNotFoundError as e:
        print(f"❌ 错误: {e}")
        return 1
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
