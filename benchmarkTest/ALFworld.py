"""
ALFworld Benchmark Test Module
用于测试 Agent 在 ALFworld 环境中的表现
"""

import json
import os
from loguru import logger
from typing import List

from agent import Agent


def testALFworld(agent: Agent, test_file: str = "benchmark/ALFworld/test_cases_valid_unseen_30.json"):
    """使用 ALFworld 测试案例测试 Agent"""
    logger.info("=" * 60)
    logger.info("AIAA3102 Agent System - ALFworld Test Mode")
    logger.info("=" * 60)
    
    # 加载测试数据
    if not os.path.exists(test_file):
        logger.error(f"测试文件不存在: {test_file}")
        print(f"❌ 错误: 测试文件 {test_file} 不存在")
        return
    
    try:
        with open(test_file, 'r', encoding='utf-8') as f:
            test_cases = json.load(f)
        logger.info(f"成功加载 {len(test_cases)} 个测试案例")
    except Exception as e:
        logger.error(f"加载测试文件失败: {e}")
        print(f"❌ 错误: 无法加载测试文件 - {e}")
        return
    
    # 打印测试横幅
    print("╔════════════════════════════════════════════════════════╗")
    print("║                                                        ║")
    print("║          🧪 ALFworld Benchmark Test Mode 🧪          ║")
    print("║                                                        ║")
    print("╚════════════════════════════════════════════════════════╝")
    print(f"\n📊 总共加载了 {len(test_cases)} 个测试案例")
    print("━" * 60)
    
    # 交互式选择测试案例
    while True:
        print("\n请选择操作:")
        print("  1. 测试单个案例")
        print("  2. 测试所有案例")
        print("  3. 测试指定范围")
        print("  0. 退出")
        
        choice = input("\n💬 请输入选项 (0-4): ").strip()
        
        if choice == '0':
            logger.info("User exited test mode")
            break
        
        elif choice == '1':
            # 测试单个案例
            print(f"\n请输入案例编号 (1-{len(test_cases)}):")
            try:
                case_num = int(input("💬 案例编号: ").strip())
                if 1 <= case_num <= len(test_cases):
                    test_single_case(agent, test_cases[case_num - 1], case_num)
                else:
                    print(f"❌ 无效的案例编号，请输入 1-{len(test_cases)}")
            except ValueError:
                print("❌ 请输入有效的数字")
            except KeyboardInterrupt:
                print("\n\n⚠️ 测试中断")
                continue
        
        elif choice == '2':
            test_all_cases(agent, test_cases)
        
        elif choice == '3':
            # 测试指定范围
            try:
                start = int(input(f"💬 起始案例编号 (1-{len(test_cases)}): ").strip())
                end = int(input(f"💬 结束案例编号 (1-{len(test_cases)}): ").strip())
                if 1 <= start <= end <= len(test_cases):
                    test_range_cases(agent, test_cases, start, end)
                else:
                    print(f"❌ 无效的范围，请输入 1-{len(test_cases)} 之间的数字")
            except ValueError:
                print("❌ 请输入有效的数字")
            except KeyboardInterrupt:
                print("\n\n⚠️ 测试中断")
                continue
        else:
            print("❌ 无效的选项，请重新选择")
    
    logger.info("ALFworld test session ended")
    logger.info("=" * 60)


def test_single_case(agent: Agent, test_case: dict, case_num: int):
    """测试单个案例"""
    logger.info(f"Testing case #{case_num}: {test_case.get('task_id')}")
    
    print("\n" + "="*60)
    print(f"测试案例 #{case_num}")
    print("="*60)
    print(f"任务ID: {test_case.get('task_id')}")
    print(f"任务类型: {test_case.get('task_type')}")
    print(f"场景编号: {test_case.get('scene_num')}")
    print(f"\n📝 任务描述:")
    print(f"  {test_case.get('task_desc')}")
    
    # 显示高层级步骤
    high_level_steps = test_case.get('high_level_steps', [])
    if high_level_steps:
        print(f"\n📋 参考步骤:")
        for i, step in enumerate(high_level_steps, 1):
            print(f"  {i}. {step}")
    
    # 显示其他描述
    alt_desc = test_case.get('alternative_descriptions', [])
    if alt_desc:
        print(f"\n🔄 其他描述:")
        for desc in alt_desc:
            print(f"  - {desc}")
    
    print("\n" + "-"*60)
    
    # 开始测试
    agent_history = []
    user_query = test_case.get('task_desc')
    
    try:
        logger.info(f"Sending query: {user_query}")
        
        agent_output, agent_history = agent.text(user_query, agent_history)
        
        # 提取最终答案
        final_answer_marker = "Final Answer:"
        final_answer = agent_output.rfind(final_answer_marker)
        if final_answer != -1:
            final_answer = agent_output[final_answer + len(final_answer_marker):].strip()
        else:
            final_answer = agent_output.strip()
        
        print(f"\n{'='*60}")
        print("Agent:")
        print(f"{'='*60}")
        print(final_answer)
        print(f"{'='*60}")
        
        logger.info(f"Case #{case_num} completed successfully")
        
    except KeyboardInterrupt:
        logger.warning(f"Case #{case_num} interrupted by user")
        raise
    except Exception as e:
        logger.error(f"Error testing case #{case_num}: {e}")
        print(f"\n❌ 测试出错: {e}")


def test_all_cases(agent: Agent, test_cases: list):
    """测试所有案例"""
    results = []
    total = len(test_cases)
    
    print(f"\n🚀 开始测试全部 {total} 个案例...")
    print("提示: 按 Ctrl+C 可以中断测试\n")
    
    for i, case in enumerate(test_cases, 1):
        try:
            print(f"\n{'='*60}")
            print(f"进度: {i}/{total} - {case.get('task_id')}")
            print(f"{'='*60}")
            
            test_single_case(agent, case, i)
            results.append({'case_num': i, 'status': 'success', 'error': None})
            
        except KeyboardInterrupt:
            print(f"\n\n⚠️ 测试在第 {i}/{total} 个案例时被中断")
            results.append({'case_num': i, 'status': 'interrupted', 'error': 'User interrupted'})
            break
        except Exception as e:
            logger.error(f"Case {i} failed: {e}")
            results.append({'case_num': i, 'status': 'failed', 'error': str(e)})
            print(f"\n❌ 案例 {i} 失败: {e}")
            
            # 询问是否继续
            cont = input("\n是否继续测试下一个案例? (y/n): ").strip().lower()
            if cont != 'y':
                break
    
    # 打印测试总结
    print(f"\n\n{'='*60}")
    print("📊 测试总结")
    print(f"{'='*60}")
    success = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] == 'failed')
    interrupted = sum(1 for r in results if r['status'] == 'interrupted')
    
    print(f"总测试数: {len(results)}/{total}")
    print(f"✅ 成功: {success}")
    print(f"❌ 失败: {failed}")
    print(f"⚠️  中断: {interrupted}")
    print(f"{'='*60}")


def test_range_cases(agent: Agent, test_cases: list, start: int, end: int):
    """测试指定范围的案例"""
    selected_cases = test_cases[start-1:end]
    total = len(selected_cases)
    
    print(f"\n🚀 开始测试案例 {start}-{end} (共 {total} 个)...")
    print("提示: 按 Ctrl+C 可以中断测试\n")
    
    results = []
    for i, case in enumerate(selected_cases, start):
        try:
            print(f"\n{'='*60}")
            print(f"进度: {i-start+1}/{total} - 案例 #{i}")
            print(f"{'='*60}")
            
            test_single_case(agent, case, i)
            results.append({'case_num': i, 'status': 'success', 'error': None})
            
        except KeyboardInterrupt:
            print(f"\n\n⚠️ 测试在案例 #{i} 时被中断")
            results.append({'case_num': i, 'status': 'interrupted', 'error': 'User interrupted'})
            break
        except Exception as e:
            logger.error(f"Case {i} failed: {e}")
            results.append({'case_num': i, 'status': 'failed', 'error': str(e)})
            print(f"\n❌ 案例 {i} 失败: {e}")
            
            # 询问是否继续
            cont = input("\n是否继续测试下一个案例? (y/n): ").strip().lower()
            if cont != 'y':
                break
    
    # 打印测试总结
    print(f"\n\n{'='*60}")
    print("📊 测试总结")
    print(f"{'='*60}")
    success = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] == 'failed')
    interrupted = sum(1 for r in results if r['status'] == 'interrupted')
    
    print(f"总测试数: {len(results)}/{total}")
    print(f"✅ 成功: {success}")
    print(f"❌ 失败: {failed}")
    print(f"⚠️  中断: {interrupted}")
    print(f"{'='*60}")
