"""
ALFworld Baseline Test Module (No Agent Framework)
用于消融实验：直接使用原始模型，不使用 ReAct Agent 框架

对比实验：
- ALFworld.py: 使用 Agent + ReAct 框架
- ALFworld_base.py: 直接使用原始 LLM（本文件）
"""

import json
import os
import time
import signal
from contextlib import contextmanager
from loguru import logger
from typing import List, Tuple, Dict, Any

from llm import BaseLLM
from benchmarkTest.result_logger import ResultLogger


class TimeoutException(Exception):
    """超时异常"""
    pass


@contextmanager
def time_limit(seconds: int):
    """
    上下文管理器：设置代码块执行时间限制
    
    Args:
        seconds: 超时时间（秒）
    """
    def signal_handler(signum, frame):
        raise TimeoutException(f"Timed out after {seconds} seconds")
    
    # 设置信号处理器
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # 取消定时器
        signal.alarm(0)


# 尝试导入 ALFWorld
try:
    import alfworld.agents.environment as environment
    import alfworld.agents.modules.generic as generic
    ALFWORLD_AVAILABLE = True
except ImportError:
    ALFWORLD_AVAILABLE = False
    logger.warning("ALFWorld not installed. Install with: pip install alfworld[full]")


def testALFworld_base(llm: BaseLLM, config_path: str = "configs/base_config.yaml", num_games: int = None):
    """基线测试：直接使用原始 LLM，不使用 Agent 框架"""
    logger.info("=" * 60)
    logger.info("AIAA3102 Baseline Test - ALFworld WITHOUT Agent Framework")
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
    
    # 直接加载 YAML 配置文件
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    env_type = config['env']['type']
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
    print("║   🧪 ALFworld BASELINE Test (No Framework) 🧪       ║")
    print("╚════════════════════════════════════════════════════════╝")
    print(f"\n📊 环境类型: {env_type}")
    print("🔬 测试模式: 原始 LLM（无 Agent 框架）")
    print("━" * 60)
    
    # 询问测试数量
    if num_games is None:
        while True:
            try:
                user_input = input("\n请输入要测试的游戏数量 (1-30): ").strip()
                num_games = int(user_input) if user_input else 30
                if 1 <= num_games <= 30:
                    break
                print("⚠️ 请输入 1-30 之间的数字")
            except ValueError:
                print("⚠️ 请输入有效的数字")
    
    print(f"\n准备测试 {num_games} 个游戏场景...")
    
    # 询问是否从特定游戏开始
    start_from = 1
    try:
        start_input = input("\n从第几个游戏开始？(直接回车从1开始): ").strip()
        if start_input:
            start_from = int(start_input)
            if start_from < 1:
                start_from = 1
                print(f"⚠️ 开始序号无效，从第 1 个游戏开始")
            elif start_from > num_games:
                print(f"⚠️ 开始序号 {start_from} 超过总数 {num_games}，从第 1 个游戏开始")
                start_from = 1
    except ValueError:
        print("⚠️ 输入无效，从第 1 个游戏开始")
        start_from = 1
    
    results = []
    
    # 创建结果记录器（使用不同的子目录）
    result_logger = ResultLogger(prefix="baseline")
    
    try:
        for game_count in range(1, num_games + 1):
            # 重置环境获取新任务
            obs, info = env.reset()
            task_desc = obs[0].strip()
            
            # 如果还没到开始位置，跳过这个游戏
            if game_count < start_from:
                print(f"⏭️  跳过游戏 {game_count}/{num_games}")
                continue
            
            print(f"\n{'='*60}")
            print(f"🎮 游戏 {game_count}/{num_games}")
            print(f"{'='*60}")
            
            print(f"\n📝 任务: {task_desc}\n")
            logger.info(f"Game {game_count}/{num_games} - Task: {task_desc}")
            
            # 运行单个游戏（使用原始 LLM）
            result = test_interactive_game_base(llm, env, task_desc, game_count)
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
    print("📊 基线测试总结")
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
    
    logger.info(f"Baseline test completed: {success_games}/{total_games} games, {successful_steps}/{total_steps} steps successful")
    logger.info("=" * 60)


def test_interactive_game_base(llm: BaseLLM, env: Any, task_desc: str, game_num: int, max_steps: int = 40) -> Dict[str, Any]:
    """测试单个交互式游戏（使用原始 LLM，无 Agent 框架）
    
    Args:
        llm: LLM 实例（不是 Agent）
        env: ALFWorld 环境
        task_desc: 任务描述
        game_num: 游戏编号
        max_steps: 最大步数
        
    Returns:
        包含游戏结果的字典
    """
    step_count = 0
    successful_steps = 0
    timeout_count = 0
    parse_error_count = 0
    done = False
    success = False
    
    print(f"\n{'─'*60}")
    print("🤖 原始 LLM 开始执行任务（无 Agent 框架）")
    print(f"{'─'*60}\n")
    
    # 初始化
    current_observation = task_desc
    action_history = []
    
    # 第一次提示（简化版，直接要求输出命令）
    initial_prompt = f"""你是一个游戏玩家，正在玩文本冒险游戏。

{task_desc}

你需要执行命令来完成任务。

可用命令格式：
- go to [物体] [编号]
- take [物体] [编号] from [位置] [编号]
- open/close [物体] [编号]
- move [物体] [编号] to [位置] [编号]
- clean/heat/cool [物体] [编号] with [工具] [编号]
- look
- inventory

重要规则：
1. 操作物体前必须先 go to 那个位置
2. 所有命令必须包含物体编号（如 "apple 1", "fridge 1"）
3. 一次只能拿一个物体

请直接输出一个游戏命令（不要解释，不要多余文字）。"""
    
    try:
        while not done and step_count < max_steps:
            step_count += 1
            
            # 构建提示
            if step_count == 1:
                prompt = initial_prompt
            else:
                # 后续步骤：包含任务和上一步结果
                last_action = action_history[-1]['action']
                last_result = action_history[-1]['observation']
                
                task_simple = task_desc.split('Your task is to:')[-1].strip() if 'Your task is to:' in task_desc else task_desc
                
                prompt = f"""任务: {task_simple}

上一步: {last_action}
结果: {last_result}

下一步命令:"""
            
            # 直接调用 LLM（不使用 Agent）
            logger.info(f"Game #{game_num} Step {step_count} - Calling raw LLM")
            
            try:
                start_time = time.time()
                with time_limit(180):
                    # 直接使用 LLM 的 chat 方法（不传递 history，每次独立对话）
                    response, _ = llm.chat(
                        prompt=prompt,
                        history=None,  # 基线测试：每次独立调用，不保留历史
                        meta_instruction="",  # 不使用 meta_instruction，让提示词本身更直接
                        max_new_tokens=256,  # 减少生成长度，只需要一个命令
                        temperature=0.7
                    )
                elapsed_time = time.time() - start_time
                print(f"⏱️  LLM 响应时间: {elapsed_time:.1f}秒")
                
            except TimeoutException:
                elapsed_time = time.time() - start_time
                print(f"\n⚠️ LLM 响应超时 (>{elapsed_time:.0f}秒)！跳过此步骤...")
                logger.error(f"Game #{game_num} Step {step_count} - LLM timeout after {elapsed_time:.0f}s")
                
                timeout_count += 1
                action_history.append({
                    'action': '[TIMEOUT]',
                    'observation': f'Error: LLM response timed out after 180 seconds.'
                })
                continue
            
            # 打印 LLM 输出
            print(f"\n{'='*60}")
            print(f"🤖 LLM 原始输出 (步骤 {step_count}):")
            print(f"{'='*60}")
            print(response)
            print(f"{'='*60}\n")
            
            # 提取命令（简化版：直接从输出中提取）
            actions = extract_actions_from_response(response)
            print(f"🎯 提取的命令: {actions}\n")
            
            if not actions:
                print(f"步骤 {step_count}: ⚠️ 无法提取有效命令")
                logger.warning(f"Game #{game_num} Step {step_count} - No valid action extracted")
                
                parse_error_count += 1
                action_history.append({
                    'action': '[PARSE_ERROR]',
                    'observation': 'Error: Could not extract valid command. Please provide a clear command like "go to countertop 1".'
                })
                continue
            
            # 执行第一个命令
            action = actions[0]
            print(f"\n步骤 {step_count}: {action}")
            
            # 执行动作
            obs, scores, dones, infos = env.step([action])
            current_observation = obs[0]
            done = dones[0]
            score = scores[0]
            
            # 记录历史
            action_history.append({
                'action': action,
                'observation': current_observation
            })
            
            # 判断动作是否成功
            if "Nothing happens" not in current_observation:
                successful_steps += 1
                if len(current_observation) > 120:
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
                
                if timeout_count > 0 or parse_error_count > 0:
                    print(f"   ⚠️ 超时: {timeout_count}次, 解析错误: {parse_error_count}次")
                break
        
        if not done:
            print(f"\n⏱️  未完成任务 (执行: {step_count}步, 成功: {successful_steps}步)")
            if timeout_count > 0 or parse_error_count > 0:
                print(f"   ⚠️ 超时: {timeout_count}次, 解析错误: {parse_error_count}次")
            
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
        'task': task_desc,
        'timeout_count': timeout_count,
        'parse_error_count': parse_error_count
    }
    
    logger.info(f"Baseline Game #{game_num} result: {result}")
    return result


def extract_actions_from_response(response: str) -> List[str]:
    """从 LLM 响应中提取命令（简化版）"""
    import re
    actions = []
    
    # ALFworld 命令关键词
    action_keywords = ['go to', 'take', 'move', 'put', 'open', 'close', 'toggle',
                      'clean', 'heat', 'cool', 'examine', 'inventory', 'look']
    
    # 移除思考标记
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    
    # 尝试找引号中的命令
    quoted_commands = re.findall(r'"([^"]+)"', response)
    for cmd in quoted_commands:
        cmd = cmd.strip().lower()
        for keyword in action_keywords:
            if cmd.startswith(keyword):
                actions.append(cmd)
                break
    
    # 按行解析
    if not actions:
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            line = re.sub(r'^\d+[\.\)]\s*', '', line)
            line = re.sub(r'^[-*•]\s*', '', line)
            line = re.sub(r'^(命令|Command|Action|Next)[:：]\s*', '', line, flags=re.IGNORECASE)
            line = line.strip()
            
            if line:
                line_lower = line.lower()
                for keyword in action_keywords:
                    if line_lower.startswith(keyword):
                        actions.append(line_lower)
                        break
    
    # 在整个文本中查找
    if not actions:
        response_lower = response.lower()
        for keyword in action_keywords:
            if keyword in response_lower:
                idx = response_lower.rfind(keyword)
                end_idx = response_lower.find('\n', idx)
                if end_idx == -1:
                    end_idx = len(response_lower)
                cmd = response_lower[idx:end_idx].strip()
                actions.append(cmd)
                break
    
    return actions[:1]  # 只返回第一个命令
