"""
Baseline 能力测试脚本（消融实验）
测试纯 LLM（无 Agent 框架、无 ReAct、无工具）在 RAG 和 Web Search 任务上的表现

使用方法:
  python test_baseline_capability.py --test_sets ai                  # 只测试 AI 数据集
  python test_baseline_capability.py --test_sets robomaster          # 只测试 RoboMaster 数据集  
  python test_baseline_capability.py --test_sets web                 # 只测试 Web 数据集
  python test_baseline_capability.py --test_sets ai robomaster web   # 测试所有三个数据集
  python test_baseline_capability.py --test_sets ai --max_cases 5    # 限制每个数据集的案例数
"""

import argparse
import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from llm import Qwen3VL
from base import BaselineAgent
from testsh.selfbenchmark.benchmarkTest.general_capability import GeneralCapabilityTester
from log_config import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Baseline 能力测试（消融实验）")
    parser.add_argument(
        '--test_sets',
        type=str,
        nargs='+',
        choices=['ai', 'robomaster', 'web'],
        default=['ai', 'robomaster', 'web'],
        help='测试数据集: ai (AI知识), robomaster (RoboMaster), web (Web搜索)，可同时指定多个'
    )
    parser.add_argument(
        '--max_cases',
        type=int,
        default=None,
        help='每个数据集最多测试多少个案例（None 表示全部）'
    )
    parser.add_argument(
        '--gpu_ids',
        type=str,
        default=None,
        help='使用的GPU编号，用逗号分隔，如 "4,5,6,7"'
    )
    
    args = parser.parse_args()
    
    # 初始化日志
    setup_logger()
    
    print("=" * 70)
    print("  🧪 AIAA3102 Baseline 测试（消融实验 - 无 Agent 框架）")
    print("=" * 70)
    print(f"\n⚠️  注意: Baseline 模式")
    print("  - ❌ 无 ReAct 推理框架")
    print("  - ❌ 无工具调用（RAG/搜索）")
    print("  - ❌ 无多轮优化")
    print("  - ✅ 仅使用纯 LLM 直接回答\n")
    print(f"测试数据集: {', '.join(args.test_sets)}")
    if args.max_cases:
        print(f"每个数据集最大案例数: {args.max_cases}")
    
    # 解析 GPU IDs
    gpu_ids = [4,5,6,7]
    if args.gpu_ids:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
        print(f"GPU: {gpu_ids}")
    else:
        print("GPU: 使用默认 [0,1,2,3,4,5,6,7]")
    
    print("=" * 70)
    
    # 加载模型
    print("\n🔄 正在加载模型...")
    model_path = os.path.join(project_root, "Qwen3-8B")
    try:
        if gpu_ids:
            llm = Qwen3VL(gpu_ids=gpu_ids)
        else:
            llm = Qwen3VL()
        print("✅ 模型加载成功\n")
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        print(f"❌ 模型加载失败: {e}")
        sys.exit(1)
    
    # 定义测试集配置（使用项目根目录的路径）
    test_configs = {
        'ai': {
            'file': os.path.join(project_root, 'benchmark/RAG/test_cases_RAG_AI.txt'),
            'name': 'AI 知识库'
        },
        'robomaster': {
            'file': os.path.join(project_root, 'benchmark/RAG/test_cases_RoboMaster_RAG.txt'),
            'name': 'RoboMaster 知识库'
        },
        'web': {
            'file': os.path.join(project_root, 'benchmark/web/test_cases_web.txt'),
            'name': 'Web 搜索'
        }
    }
    
    # 创建 BaselineAgent（一次性创建，所有测试共用）
    print("\n🔄 正在初始化 BaselineAgent...")
    print("⚠️  BaselineAgent 特性:")
    print("  - 直接调用 LLM")
    print("  - 不使用任何工具")
    print("  - 不使用 ReAct 推理")
    print("  - 仅依赖模型的预训练知识\n")
    
    baseline = BaselineAgent(llm)
    
    # 运行测试
    all_results = {}
    
    try:
        for test_set in args.test_sets:
            config = test_configs[test_set]
            
            if not os.path.exists(config['file']):
                print(f"\n⚠️  跳过 {config['name']}: 测试文件不存在 ({config['file']})")
                continue
            
            print("\n" + "=" * 70)
            print(f"  🧪 测试: {config['name']} (Baseline)")
            print("=" * 70)
            print(f"  文件: {config['file']}")
            print(f"  模式: 纯 LLM，无工具")
            
            # 运行测试（结果保存到项目根目录）
            result_dir = os.path.join(project_root, f"benchmark_results/baseline/{test_set}")
            tester = GeneralCapabilityTester(baseline, result_dir=result_dir)
            tester.run_all_tests(config['file'], max_cases=args.max_cases)
            
            all_results[test_set] = {
                'avg_f1': sum(r['score'] for r in tester.results) / len(tester.results) if tester.results else 0,
                'total_cases': len(tester.results),
                'session_dir': str(tester.session_dir)
            }
        
    except KeyboardInterrupt:
        print("\n\n⚠️  测试被用户中断")
        sys.exit(0)
    except Exception as e:
        logger.error(f"测试过程出错: {e}")
        print(f"\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 打印总结
    print("\n" + "=" * 70)
    print("  📊 Baseline 测试总结")
    print("=" * 70)
    
    for test_set, results in all_results.items():
        config = test_configs[test_set]
        print(f"\n{config['name']}:")
        print(f"  案例数: {results['total_cases']}")
        print(f"  平均 F1: {results['avg_f1']:.2%}")
        print(f"  结果目录: {results['session_dir']}")
    
    print("\n✅ 所有 Baseline 测试完成！")
    print("\n💡 提示: 使用 compare_results.py 对比 Agent 和 Baseline 的结果")
    print("=" * 70)


if __name__ == "__main__":
    main()
