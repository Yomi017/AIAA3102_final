"""
通用能力测试主脚本
测试 Agent 在 RAG 和 Web Search 任务上的表现

使用方法:
  python test_general_capability.py --type rag       # 只测试 RAG
  python test_general_capability.py --type web       # 只测试 Web Search
  python test_general_capability.py --type both      # 两者都测试（默认）
  python test_general_capability.py --max_cases 5    # 每类最多测试 5 个案例
"""

import argparse
import os
import sys
from loguru import logger

from llm import Qwen3VL
from agent import Agent
from benchmarkTest.general_capability import test_general_capability
from log_config import setup_logger


def main():
    parser = argparse.ArgumentParser(description="通用能力测试")
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
    print("  🧪 AIAA3102 Agent - 通用能力测试")
    print("=" * 70)
    print(f"\n测试类型: {args.type}")
    if args.max_cases:
        print(f"最大案例数: {args.max_cases}")
    
    # 检查 DeepSeek API Key
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("\n⚠️  警告: 未设置 DEEPSEEK_API_KEY 环境变量")
        print("   自动评分功能将被禁用")
        print("   请设置: export DEEPSEEK_API_KEY='your-api-key'")
        print("   获取 API Key: https://platform.deepseek.com/")
        
        response = input("\n是否继续（不进行自动评分）？(y/n): ")
        if response.lower() != 'y':
            print("测试已取消")
            sys.exit(0)
    
    # 解析 GPU IDs
    gpu_ids = [4,5,6,7]
    if args.gpu_ids:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
        print(f"GPU: {gpu_ids}")
    else:
        print("GPU: 自动检测")
    
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
    
    # 检查 RAG 数据库
    rag_db_path = None
    if os.path.exists("rag/wiki_vector_db"):
        rag_db_path = "rag/wiki_vector_db"
        print(f"✅ 发现 AI 维基知识库: {rag_db_path}")
    elif os.path.exists("rag/vector_db"):
        rag_db_path = "rag/vector_db"
        print(f"✅ 发现知识库: {rag_db_path}")
    else:
        if args.type in ['rag', 'both']:
            print("⚠️  未找到知识库，RAG 测试可能无法正常工作")
            print("   请先运行: python rag/build_vector_db.py")
    
    # 创建 Agent
    print("\n🔄 正在初始化 Agent...")
    agent = Agent(llm, rag_db_path=rag_db_path)
    print("✅ Agent 初始化成功\n")
    
    # 运行测试
    try:
        test_general_capability(agent, test_type=args.type, max_cases=args.max_cases)
    except KeyboardInterrupt:
        print("\n\n⚠️  测试被用户中断")
        sys.exit(0)
    except Exception as e:
        logger.error(f"测试过程出错: {e}")
        print(f"\n❌ 测试出错: {e}")
        sys.exit(1)
    
    print("\n✅ 所有测试完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
