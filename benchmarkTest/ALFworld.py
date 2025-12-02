"""
ALFworld Benchmark Test Module
用于测试 Agent 在 ALFworld 环境中的表现

ALFWorld 是一个交互式文本游戏环境，Agent 需要通过观察环境、执行动作来完成任务。
环境会返回观察结果和可用动作列表。
"""

import json
import os
from loguru import logger
from typing import List, Tuple, Dict, Any

from agent import Agent

# 尝试导入 ALFWorld
try:
    import alfworld.agents.environment as environment
    import alfworld.agents.modules.generic as generic
    ALFWORLD_AVAILABLE = True
except ImportError:
    ALFWORLD_AVAILABLE = False
    logger.warning("ALFWorld not installed. Install with: pip install alfworld[full]")


def testALFworld(agent: Agent, config_path: str = "configs/base_config.yaml", num_games: int = None):
    """使用 ALFworld 环境测试 Agent
    
    Args:
        agent: Agent 实例
        config_path: ALFWorld 配置文件路径
        num_games: 要测试的游戏数量，None 表示询问用户
    """
    logger.info("=" * 60)
    logger.info("AIAA3102 Agent System - ALFworld Test Mode")
    logger.info("=" * 60)
    
    # 检查 ALFWorld 是否可用
    if not ALFWORLD_AVAILABLE:
        print("❌ 错误: ALFWorld 未安装")
        print("\n请按照以下步骤安装 ALFWorld:")
        print("1. pip install alfworld[full]")
        print("2. alfworld-download  # 下载游戏数据")
        return
    
    # 加载 ALFWorld 配置
    if not os.path.exists(config_path):
        logger.error(f"配置文件不存在: {config_path}")
        print(f"❌ 配置文件不存在: {config_path}")
        return
    
    # 直接加载 YAML 配置文件（避免使用 argparse）
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    env_type = config['env']['type']  # 'AlfredTWEnv' or 'AlfredThorEnv'
    logger.info(f"环境类型: {env_type}")
    
    # 初始化环境
    try:
        print("🔄 正在初始化 ALFWorld 环境...")
        env_class = environment.get_environment(env_type)
        env = env_class(config, train_eval='eval_out_of_distribution')
        env = env.init_env(batch_size=1)
        print("✅ 环境初始化成功\n")
        logger.info("ALFWorld environment initialized successfully")
    except Exception as e:
        logger.error(f"环境初始化失败: {e}")
        print(f"❌ 环境初始化失败: {e}")
        print("\n提示: 请确保已运行 'alfworld-download' 下载游戏数据")
        return
    
    # 打印测试横幅
    print("╔════════════════════════════════════════════════════════╗")
    print("║       🧪 ALFworld Interactive Test Mode 🧪           ║")
    print("╚════════════════════════════════════════════════════════╝")
    print(f"\n📊 环境类型: {env_type}")
    print("━" * 60)
    
    # 询问测试数量
    if num_games is None:
        while True:
            try:
                num_input = input("\n请输入要测试的游戏数量 (1-30): ").strip()
                num_games = int(num_input)
                if 1 <= num_games <= 30:
                    break
                print("⚠️ 请输入 1-30 之间的数字")
            except ValueError:
                print("⚠️ 请输入有效的数字")
    
    print(f"\n准备测试 {num_games} 个游戏场景...")
    
    results = []
    
    try:
        for game_count in range(1, num_games + 1):
            print(f"\n{'='*60}")
            print(f"🎮 游戏 {game_count}/{num_games}")
            print(f"{'='*60}")
            
            # 重置环境获取新任务
            obs, info = env.reset()
            task_desc = obs[0].strip()
            
            print(f"\n📝 任务: {task_desc}\n")
            logger.info(f"Game {game_count}/{num_games} - Task: {task_desc}")
            
            # 运行单个游戏
            result = test_interactive_game(agent, env, task_desc, game_count)
            results.append(result)
    
    except KeyboardInterrupt:
        print("\n\n⚠️ 测试被用户中断")
    except Exception as e:
        logger.error(f"测试过程出错: {e}")
        print(f"\n❌ 测试出错: {e}")
    
    # 打印测试总结
    print(f"\n\n{'='*60}")
    print("📊 测试总结")
    print(f"{'='*60}")
    success = sum(1 for r in results if r['success'])
    total = len(results)
    
    print(f"总游戏数: {total}")
    print(f"✅ 成功: {success}")
    print(f"❌ 失败: {total - success}")
    if total > 0:
        print(f"📈 成功率: {success/total*100:.1f}%")
    print(f"{'='*60}")
    
    logger.info(f"ALFWorld test completed: {success}/{total} games successful")
    logger.info("=" * 60)


def test_interactive_game(agent: Agent, env: Any, task_desc: str, game_num: int, max_steps: int = 50) -> Dict[str, Any]:
    """测试单个交互式游戏
    
    Args:
        agent: Agent 实例
        env: ALFWorld 环境
        task_desc: 任务描述
        game_num: 游戏编号
        max_steps: 最大步数
        
    Returns:
        包含游戏结果的字典
    """
    agent_history = []
    step_count = 0
    done = False
    success = False
    
    print(f"\n{'─'*60}")
    print("🤖 Agent 开始执行任务...")
    print(f"{'─'*60}\n")
    
    # 构建任务提示（简洁版，环境会提供详细观察）
    full_context = f"""You are playing a text-based interactive game. 

Task: {task_desc}

Please provide a step-by-step action plan with specific commands to complete this task.
Use commands like: "go to X", "take X from Y", "put X in/on Y", "open X", "close X", "toggle X", "clean X with Y", "heat X with Y", "cool X with Y", "examine X", "inventory", "look".

Format: List each action as a numbered command.
"""

    
    try:
        # 让 Agent 生成执行计划
        logger.info(f"Game #{game_num} - Requesting action plan from agent")
        agent_output, agent_history = agent.text(full_context, agent_history)
        
        # 提取 Agent 的回答
        final_answer_marker = "Final Answer:"
        if final_answer_marker in agent_output:
            response = agent_output[agent_output.rfind(final_answer_marker) + len(final_answer_marker):].strip()
        else:
            response = agent_output.strip()
        
        print(f"🤖 Agent 回答:\n{response}\n")
        
        # 从 Agent 回答中提取动作序列
        actions = extract_actions_from_response(response)
        
        if not actions:
            print("⚠️  无法从 Agent 回答中提取有效动作")
            logger.warning(f"Game #{game_num} - No valid actions extracted")
            return {'game_num': game_num, 'success': False, 'steps': 0, 'reason': 'No actions extracted'}
        
        print(f"\n📋 提取到 {len(actions)} 个动作:")
        for i, action in enumerate(actions, 1):
            print(f"  {i}. {action}")
        print()
        
        # 执行动作序列
        for action in actions:
            if step_count >= max_steps:
                print(f"\n⚠️  达到最大步数限制 ({max_steps})")
                break
            
            step_count += 1
            print(f"步骤 {step_count}: {action}")
            
            # 执行动作
            obs, scores, dones, infos = env.step([action])
            observation = obs[0]
            done = dones[0]
            score = scores[0]
            
            print(f"  观察: {observation[:200]}{'...' if len(observation) > 200 else ''}")
            
            if done:
                success = score > 0
                if success:
                    print(f"\n🎉 任务完成! (得分: {score}, 步数: {step_count})")
                else:
                    print(f"\n❌ 任务失败 (得分: {score}, 步数: {step_count})")
                break
        
        if not done:
            print(f"\n⏱️  未完成任务 (已执行 {step_count} 步)")
            
    except KeyboardInterrupt:
        print("\n⚠️  游戏被中断")
        raise
    except Exception as e:
        logger.error(f"Game #{game_num} error: {e}")
        print(f"\n❌ 执行出错: {e}")
        return {'game_num': game_num, 'success': False, 'steps': step_count, 'error': str(e)}
    
    result = {
        'game_num': game_num,
        'success': success,
        'steps': step_count,
        'task': task_desc
    }
    
    logger.info(f"Game #{game_num} result: {result}")
    return result


def extract_actions_from_response(response: str) -> List[str]:
    """从 Agent 的回答中提取动作序列
    
    尝试识别类似以下格式的动作:
    - go to countertop 1
    - take apple 1 from countertop 1
    - etc.
    """
    import re
    actions = []
    lines = response.split('\n')
    
    action_keywords = ['go to', 'take', 'put', 'open', 'close', 'toggle', 
                      'clean', 'heat', 'cool', 'examine', 'inventory', 'look']
    
    for line in lines:
        line = line.strip()
        
        # 移除编号前缀 (如 "1. ", "1) ", "- ")
        line = re.sub(r'^\d+[\.\)]\s*', '', line)
        line = re.sub(r'^[-*]\s*', '', line)
        line = line.strip()
        
        # 检查是否包含动作关键词（不区分大小写检查，但保留原始大小写）
        if line:
            line_lower = line.lower()
            for keyword in action_keywords:
                if line_lower.startswith(keyword):
                    actions.append(line)
                    break
    
    return actions[:20]  # 限制动作数量
    