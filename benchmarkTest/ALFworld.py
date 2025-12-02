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
from benchmarkTest.result_logger import ResultLogger

# 尝试导入 ALFWorld
try:
    import alfworld.agents.environment as environment
    import alfworld.agents.modules.generic as generic
    ALFWORLD_AVAILABLE = True
except ImportError:
    ALFWORLD_AVAILABLE = False
    logger.warning("ALFWorld not installed. Install with: pip install alfworld[full]")


def testALFworld(agent: Agent, config_path: str = "configs/base_config.yaml", num_games: int = None):
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
    
    # 创建结果记录器
    result_logger = ResultLogger()
    
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
            
            # 记录到结果记录器
            result_logger.add_result(result)
    
    except KeyboardInterrupt:
        print("\n\n⚠️ 测试被用户中断")
    except Exception as e:
        logger.error(f"测试过程出错: {e}")
        print(f"\n❌ 测试出错: {e}")
    
    # 保存所有结果并生成统计报告
    result_logger.finalize()
    
    # 打印测试总结
    print(f"\n\n{'='*60}")
    print("📊 测试总结")
    print(f"{'='*60}")
    
    # 任务完成率
    success_games = sum(1 for r in results if r['success'])
    total_games = len(results)
    
    # 动作成功率
    total_steps = sum(r.get('steps', 0) for r in results)
    successful_steps = sum(r.get('successful_steps', 0) for r in results)
    
    print(f"\n【任务完成情况】")
    print(f"  总游戏数: {total_games}")
    print(f"  ✅ 完成: {success_games}")
    print(f"  ❌ 未完成: {total_games - success_games}")
    if total_games > 0:
        print(f"  📈 任务完成率: {success_games/total_games*100:.1f}%")
    
    print(f"\n【动作执行情况】")
    print(f"  总执行步数: {total_steps}")
    print(f"  ✅ 成功步数: {successful_steps}")
    print(f"  ❌ 失败步数: {total_steps - successful_steps}")
    if total_steps > 0:
        print(f"  📈 动作成功率: {successful_steps/total_steps*100:.1f}%")
    
    print(f"\n{'='*60}")
    
    logger.info(f"ALFWorld test completed: {success_games}/{total_games} games, {successful_steps}/{total_steps} steps successful")
    logger.info("=" * 60)


def test_interactive_game(agent: Agent, env: Any, task_desc: str, game_num: int, max_steps: int = 50) -> Dict[str, Any]:
    """测试单个交互式游戏（逐步交互模式）
    
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
    successful_steps = 0
    done = False
    success = False
    
    print(f"\n{'─'*60}")
    print("🤖 Agent 开始执行任务（逐步交互模式）")
    print(f"{'─'*60}\n")
    
    # 初始化
    current_observation = task_desc
    action_history = []  # 记录所有动作和结果
    
    # 第一次提示（包含完整说明 - 中英文双语）
    initial_prompt = f"""你正在玩一个基于文本的交互式游戏。You are playing a text-based interactive game.
    你将获得一个任务描述，你需要完成这个任务。You will be given a task description. You need to complete the task.

    任务 Task: {task_desc}

    【关键规则 CRITICAL Rules】
    1. 先移动到物体旁边才能操作它
    You MUST "go to [object] [number]" BEFORE you can interact with it!
    
    示例 Example: 
    - 在 "open fridge 1" 之前，必须先 "go to fridge 1"
    - Before "open fridge 1", you must first "go to fridge 1"

    2. 所有物体都有编号（如 "apple 1", "fridge 1"），必须包含编号！
    Objects have numbers (e.g., "apple 1", "fridge 1"). Always include numbers!

    3. 如果看到 "Nothing happens"，可能表示：
    If "Nothing happens", it usually means:
    - 你还没移动到那个位置
        You are not at that location yet (use "go to" first)
    - 物体不存在或无法打开
        The object doesn't exist or can't be opened
    - 文本格式错误
        Text formatting error

    4. Complete successful example 完整成功示例:
   Task: put some vase in safe
   
    go to shelf 6          
    take vase 2 from shelf 6 
    go to safe 1          
    open safe 1
    move vase 2 to safe 1
   
   Key pattern 关键模式:
   - ALWAYS "go to" before interacting 总是先移动
   - Open containers before putting things in 先打开容器

    可用命令 Available Commands:
    - go to [object] [number]
    - open/close [object] [number]
    - take [object] [number] from [location] [number] - 拿起物体
    - move [object] [number] to [location] [number] - 这个是把物体放在指定位置的命令！
    - clean/heat/cool [object] [number] with [tool] [number] - 使用工具
    - examine [object] [number] - 检查物体
    - inventory - 查看手上的状态物品
    - look       - 查看当前场景 

   move [object] [number] to [location] [number] 替代了原本 put [object] [number] in/on [location] [number]
   示例： move spatula 1 to drawer 1

    CRITICAL IMPORTANT 关键重要 !!!
    这些命令不是工具调用！不要使用 Action: 和 Action Input:！
    These are NOT tool calls! Do NOT use Action: and Action Input:!
    
    直接输出命令！Just output the command directly!
    正确格式 Correct format:
    Thought: [你的思考]
    Final Answer: go to countertop 1
    
    错误格式 Wrong format (DON'T do this):
    Action: go to countertop 1
    Action Input: {{"command": "go to countertop 1"}}
    
    示例 Example:
    Thought: 我需要先找到 spatula，可能在厨房台面上
    Final Answer: go to countertop 2

    Provide your NEXT action (only ONE command in Final Answer).
    5. 你不能同时拿两个物品！！
    """
    
    try:
        while not done and step_count < max_steps:
            step_count += 1
            
            # 构建包含历史的提示
            if step_count == 1:
                # 第一步使用完整提示
                prompt = initial_prompt
            else:
                # 后续步骤：包含任务、上一步动作和结果
                last_action = action_history[-1]['action']
                last_result = action_history[-1]['observation']
                
                # 智能失败提示
                failure_warning = ""
                if "Nothing happens" in last_result:
                    if "open" in last_action.lower() or "close" in last_action.lower():
                        failure_warning = "\n⚠️ Failed! Did you 'go to' that object first?"
                    elif "take" in last_action.lower():
                        failure_warning = "\n⚠️ Failed! Make sure you opened the container or are at the right location."
                    else:
                        failure_warning = "\n⚠️ Failed! Check your command format or location."
                
                prompt = f"""Task: {task_desc.split('Your task is to:')[-1].strip() if 'Your task is to:' in task_desc else task_desc}

    Previous action: {last_action}
    Result: {last_result}{failure_warning}

    Provide your NEXT action (ONE command with object numbers).
    Remember: Use "go to [object] [number]" to move before interacting!
    """
            
            # 让 Agent 生成下一个动作
            logger.info(f"Game #{game_num} Step {step_count} - Requesting next action")
            agent_output, agent_history = agent.text(prompt, agent_history)
            
            # 【调试】打印 Agent 的完整输出
            print(f"\n{'='*60}")
            print(f"🤖 Agent 原始输出 (步骤 {step_count}):")
            print(f"{'='*60}")
            print(agent_output)
            print(f"{'='*60}\n")
            
            # 提取 Agent 的回答
            final_answer_marker = "Final Answer:"
            if final_answer_marker in agent_output:
                response = agent_output[agent_output.rfind(final_answer_marker) + len(final_answer_marker):].strip()
                print(f"✂️  提取的 Final Answer: '{response}'\n")
            else:
                response = agent_output.strip()
                print(f"⚠️  没有 Final Answer 标记，使用全部输出\n")
            
            # 提取单个动作
            actions = extract_actions_from_response(response)
            print(f"🎯 提取的动作列表: {actions}\n")
            
            if not actions:
                print(f"步骤 {step_count}: ⚠️ 无法提取有效动作")
                logger.warning(f"Game #{game_num} Step {step_count} - No valid action extracted")
                break
            
            # 只执行第一个动作
            action = actions[0]
            print(f"\n步骤 {step_count}: {action}")
            
            # 执行动作
            obs, scores, dones, infos = env.step([action])
            current_observation = obs[0]
            done = dones[0]
            score = scores[0]
            
            # 记录到历史
            action_history.append({
                'action': action,
                'observation': current_observation
            })
            
            # 判断动作是否成功
            if "Nothing happens" not in current_observation:
                successful_steps += 1
                # 显示完整观察结果，但格式化换行
                if len(current_observation) > 120:
                    # 长输出：缩进换行显示
                    print(f"  ✅ {current_observation[:120]}")
                    remaining = current_observation[120:]
                    while remaining:
                        print(f"     {remaining[:120]}")
                        remaining = remaining[120:]
                else:
                    print(f"  ✅ {current_observation}")
            else:
                print(f"  ❌ {current_observation}")
            
            if done:
                success = score > 0
                if success:
                    print(f"\n🎉 任务完成! (得分: {score}, 执行: {step_count}步, 成功: {successful_steps}步)")
                else:
                    print(f"\n❌ 任务失败 (得分: {score}, 执行: {step_count}步, 成功: {successful_steps}步)")
                break
        
        if not done:
            print(f"\n⏱未完成任务 (执行: {step_count}步, 成功: {successful_steps}步)")
            
    except KeyboardInterrupt:
        print("\n 游戏被中断")
        raise
    except Exception as e:
        logger.error(f"Game #{game_num} error: {e}")
        print(f"\n❌ 执行出错: {e}")
        return {
            'game_num': game_num, 
            'success': False, 
            'steps': step_count, 
            'successful_steps': successful_steps,
            'error': str(e)
        }
    
    result = {
        'game_num': game_num,
        'success': success,
        'steps': step_count,
        'successful_steps': successful_steps,
        'task': task_desc
    }
    
    logger.info(f"Game #{game_num} result: {result}")
    return result


def extract_actions_from_response(response: str) -> List[str]:
    """从 Agent 的回答中提取动作序列
    
    尝试识别ALFworld命令格式
    """
    import re
    actions = []
    
    # ALFworld 有效命令关键词
    action_keywords = ['go to', 'take', 'move', 'put', 'open', 'close', 'toggle', 
                      'clean', 'heat', 'cool', 'examine', 'inventory', 'look']
    
    # 先尝试找到完整的命令行（可能在引号中）
    quoted_commands = re.findall(r'"([^"]+)"', response)
    for cmd in quoted_commands:
        cmd = cmd.strip().lower()
        for keyword in action_keywords:
            if cmd.startswith(keyword):
                actions.append(cmd)
                break
    
    # 如果没找到引号命令，按行解析
    if not actions:
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            
            # 移除各种前缀
            line = re.sub(r'^\d+[\.\)]\s*', '', line)  # 数字编号
            line = re.sub(r'^[-*•]\s*', '', line)  # 列表符号
            line = re.sub(r'^(Action|Command|Next):\s*', '', line, flags=re.IGNORECASE)  # Action: 前缀
            line = line.strip()
            
            # 检查是否是有效命令
            if line:
                line_lower = line.lower()
                for keyword in action_keywords:
                    if line_lower.startswith(keyword):
                        actions.append(line_lower)  # 统一转小写
                        break
    
    # 如果还是没找到，尝试在整个文本中找最后一个命令
    if not actions:
        response_lower = response.lower()
        for keyword in action_keywords:
            if keyword in response_lower:
                # 找到最后一次出现的位置
                idx = response_lower.rfind(keyword)
                # 提取该命令（到下一个换行或结束）
                end_idx = response_lower.find('\n', idx)
                if end_idx == -1:
                    end_idx = len(response_lower)
                cmd = response_lower[idx:end_idx].strip()
                actions.append(cmd)
                break
    
    return actions[:1]  # 只返回第一个动作（交互模式）
    