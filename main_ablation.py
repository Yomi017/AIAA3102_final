"""
ALFworld 消融实验主脚本
用于对比 Agent 框架和原始 LLM 的性能

使用方法:
  python main_ablation.py --mode agent       # 使用 Agent + ReAct 框架
  python main_ablation.py --mode baseline    # 使用原始 LLM（无框架）
  python main_ablation.py --mode both        # 同时运行两种测试
"""

import argparse
from loguru import logger

from llm import Qwen3VL
from agent import Agent
from benchmarkTest import testALFworld, testALFworld_base


def main():
    parser = argparse.ArgumentParser(description="ALFworld 消融实验")
    parser.add_argument(
        '--mode',
        type=str,
        choices=['agent', 'baseline', 'both'],
        default='agent',
        help='测试模式: agent (使用框架), baseline (无框架), both (两者都测)'
    )
    parser.add_argument(
        '--num_games',
        type=int,
        default=None,
        help='测试游戏数量 (1-30)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/base_config.yaml',
        help='ALFworld 配置文件路径'
    )
    parser.add_argument(
        '--gpu_ids',
        type=str,
        default=None,
        help='使用的GPU编号，用逗号分隔，如 "4,5,6,7"'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  🧪 ALFworld 消融实验 - Ablation Study 🧪")
    print("=" * 70)
    print(f"\n模式: {args.mode}")
    print(f"配置: {args.config}")
    if args.num_games:
        print(f"游戏数: {args.num_games}")
    
    # 解析 GPU IDs
    gpu_ids = None
    if args.gpu_ids:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
        print(f"GPU: {gpu_ids}")
    else:
        print("GPU: 自动检测")
    
    print("=" * 70)
    
    # 初始化 LLM
    print("\n🔄 正在加载模型...")
    if gpu_ids:
        llm = Qwen3VL(path="Qwen3-8B", gpu_ids=gpu_ids)
    else:
        llm = Qwen3VL(path="Qwen3-8B")
    print("✅ 模型加载完成\n")
    
    if args.mode == 'agent' or args.mode == 'both':
        print("\n" + "=" * 70)
        print("  📋 测试 1: Agent + ReAct 框架")
        print("=" * 70)
        
        # 创建 Agent
        agent = Agent(llm)
        
        # 运行 Agent 测试
        testALFworld(agent, config_path=args.config, num_games=args.num_games)
        
        if args.mode == 'both':
            print("\n\n⏸️  第一个测试完成，按回车继续第二个测试...")
            input()
    
    if args.mode == 'baseline' or args.mode == 'both':
        print("\n" + "=" * 70)
        print("  📋 测试 2: 原始 LLM（无框架）")
        print("=" * 70)
        
        # 运行基线测试（直接使用 LLM）
        testALFworld_base(llm, config_path=args.config, num_games=args.num_games)
    
    print("\n\n" + "=" * 70)
    print("  ✅ 所有测试完成！")
    print("=" * 70)
    print("\n📊 结果保存位置:")
    print("  - Agent 框架测试: benchmark_results/agent_session_YYYYMMDD_HHMMSS/")
    print("  - 基线测试:       benchmark_results/baseline_session_YYYYMMDD_HHMMSS/")
    print("\n💡 对比分析:")
    print("  1. 查看各自的 statistics.txt 文件")
    print("  2. 对比 detailed_results.csv 中的成功率")
    print("  3. 分析哪种方法在哪些任务类型上表现更好")
    print("=" * 70)


if __name__ == "__main__":
    main()
