"""
ALFworld 消融实验 - Baseline (无 Agent)
使用 BaselineAgent（无 ReAct 框架、无工具）

关键：使用 BaselineAgent 类，保持与 RAG/Web 消融实验一致
"""

import argparse
import os
import sys
import time
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from loguru import logger

from llm import Qwen3VL
from base import BaselineAgent  # 使用统一的 BaselineAgent
from benchmarkTest.result_logger import ResultLogger  # 使用统一的结果记录
from log_config import setup_logger

# 尝试导入 ALFWorld
try:
    import alfworld.agents.environment as environment
    ALFWORLD_AVAILABLE = True
except ImportError:
    ALFWORLD_AVAILABLE = False
    logger.warning("ALFWorld not installed")


def extract_action_from_response(response: str) -> str:
    """
    从 LLM 响应中提取动作命令
    使用与 benchmarkTest/ALFworld.py 相同的逻辑
    """
    import re
    
    action_keywords = ['go to', 'take', 'move', 'put', 'open', 'close', 'toggle', 
                      'clean', 'heat', 'cool', 'examine', 'inventory', 'look']
    
    # 1. 查找引号中的命令
    quoted_commands = re.findall(r'"([^"]+)"', response)
    for cmd in quoted_commands:
        cmd = cmd.strip().lower()
        for keyword in action_keywords:
            if cmd.startswith(keyword):
                return cmd
    
    # 2. 按行解析
    lines = response.split('\n')
    for line in lines:
        line = line.strip()
        line = re.sub(r'^\d+[\.\)]\s*', '', line)
        line = re.sub(r'^[-*•]\s*', '', line)
        line = re.sub(r'^(Action|Command|Next|Final Answer):\s*', '', line, flags=re.IGNORECASE)
        line = line.strip()
        
        if line:
            line_lower = line.lower()
            for keyword in action_keywords:
                if line_lower.startswith(keyword):
                    return line_lower
    
    # 3. 在整个文本中找最后一个命令
    response_lower = response.lower()
    for keyword in action_keywords:
        if keyword in response_lower:
            idx = response_lower.rfind(keyword)
            end_idx = response_lower.find('\n', idx)
            if end_idx == -1:
                end_idx = len(response_lower)
            cmd = response_lower[idx:end_idx].strip()
            return cmd
    
    return None


def run_baseline_game(baseline: BaselineAgent, env, game_num: int, max_steps: int = 40) -> Dict[str, Any]:
    """
    运行单个游戏 - Baseline 模式（使用 BaselineAgent）
    
    Args:
        baseline: BaselineAgent 实例（与 RAG/Web 测试一致）
        env: ALFWorld 环境
        game_num: 游戏编号
        max_steps: 最大步数
        
    Returns:
        游戏结果字典
    """
    # 重置环境
    obs, infos = env.reset()
    task_desc = obs[0]
    
    print(f"\n{'='*70}")
    print(f"🎮 游戏 {game_num} - Baseline (使用 BaselineAgent)")
    print(f"{'='*70}")
    print(f"\n📝 任务: {task_desc}\n")
    
    logger.info(f"Baseline Game {game_num} - Task: {task_desc}")
    
    # 游戏状态
    step_count = 0
    successful_steps = 0
    done = False
    success = False
    action_history = []
    baseline_history = []  # BaselineAgent 的对话历史
    
    # 第一次提示
    initial_prompt = f"""你正在玩一个基于文本的交互式游戏。

任务: {task_desc}

【关键规则】
1. 先移动到物体旁边才能操作它（例如：在 "open fridge 1" 之前，必须先 "go to fridge 1"）
2. 所有物体都有编号（如 "apple 1", "fridge 1"），命令中必须包含编号
3. 每次只给出一个动作命令

可用命令:
- go to [object] [number]
- open/close [object] [number]
- take [object] [number] from [location] [number]
- move [object] [number] to [location] [number]
- examine [object] [number]
- inventory / look

请给出你的第一个动作（只给出命令）："""
    
    game_start = time.time()
    
    try:
        while not done and step_count < max_steps:
            step_count += 1
            
            # 构建提示
            if step_count == 1:
                prompt = initial_prompt
            else:
                last_action = action_history[-1]['action']
                last_result = action_history[-1]['observation']
                
                failure_warning = ""
                if "Nothing happens" in last_result:
                    failure_warning = "\n⚠️ 上一步失败，请检查是否需要先移动到物体旁边。"
                
                prompt = f"""任务: {task_desc.split('Your task is to:')[-1].strip() if 'Your task is to:' in task_desc else task_desc}

上一步动作: {last_action}
结果: {last_result}{failure_warning}

请给出下一个动作（只给出命令）："""
            
            # 使用 BaselineAgent.text() 方法（与 RAG/Web 测试一致）
            logger.info(f"Baseline Game #{game_num} Step {step_count} - Calling BaselineAgent")
            
            step_start = time.time()
            
            # 调用 BaselineAgent（保留对话历史）
            response, baseline_history = baseline.text(
                prompt,
                history=baseline_history,
                images=None
            )
            
            step_time = time.time() - step_start
            
            print(f"\n步骤 {step_count}:")
            print(f"⏱️  LLM 响应时间: {step_time:.1f}秒")
            
            # 提取动作
            action = extract_action_from_response(response)
            
            if not action:
                print(f"⚠️ 无法提取有效动作")
                print(f"LLM 输出: {response[:200]}")
                logger.warning(f"Baseline Game #{game_num} Step {step_count} - No valid action")
                
                action_history.append({
                    'action': '[PARSE_ERROR]',
                    'observation': 'Error: Could not extract valid action',
                    'success': False
                })
                continue
            
            print(f"🎯 提取的动作: {action}")
            
            # 执行动作
            obs, scores, dones, infos = env.step([action])
            observation = obs[0]
            done = dones[0]
            score = scores[0]
            
            # 判断步骤成功
            step_success = "Nothing happens" not in observation
            if step_success:
                successful_steps += 1
            
            action_history.append({
                'action': action,
                'observation': observation,
                'success': step_success
            })
            
            # 显示结果
            if step_success:
                print(f"✅ {observation[:100]}{'...' if len(observation) > 100 else ''}")
            else:
                print(f"❌ {observation}")
            
            if done:
                success = score > 0
                print(f"\n{'='*70}")
                if success:
                    print(f"🎉 任务完成！(步数: {step_count}, 成功步数: {successful_steps})")
                else:
                    print(f"❌ 任务失败 (步数: {step_count}, 成功步数: {successful_steps})")
                print(f"{'='*70}")
                break
    
    except KeyboardInterrupt:
        print("\n⚠️ 游戏被中断")
        logger.warning(f"Baseline Game {game_num} interrupted")
    except Exception as e:
        print(f"\n❌ 游戏出错: {e}")
        logger.error(f"Baseline Game {game_num} error: {e}")
    
    game_time = time.time() - game_start
    
    # 构建结果
    result = {
        'game_num': game_num,
        'task': task_desc,
        'success': success,
        'steps': step_count,
        'successful_steps': successful_steps,
        'action_history': action_history,
        'elapsed_time': game_time,
        'max_steps_reached': step_count >= max_steps and not done
    }
    
    logger.info(f"Baseline Game {game_num} result: success={success}, steps={step_count}/{successful_steps}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="ALFworld 消融实验 - Baseline (使用 BaselineAgent)")
    parser.add_argument('--config', type=str, default='configs/base_config.yaml', help='ALFworld 配置文件')
    parser.add_argument('--num_games', type=int, default=None, help='测试游戏数量')
    parser.add_argument('--gpu_ids', type=str, default=None, help='GPU IDs (逗号分隔)')
    
    args = parser.parse_args()
    
    # 初始化日志
    setup_logger()
    
    print("=" * 70)
    print("  🧪 ALFworld 消融实验 - Baseline (BaselineAgent)")
    print("=" * 70)
    print("\n⚠️  使用 BaselineAgent（与 RAG/Web 测试一致）:")
    print("  - ❌ 无 ReAct 推理框架")
    print("  - ❌ 无工具调用")
    print("  - ❌ 无多轮优化")
    print("  - ✅ 直接调用 LLM\n")
    
    # 检查 ALFWorld
    if not ALFWORLD_AVAILABLE:
        print("❌ ALFWorld 未安装，请运行: pip install alfworld[full]")
        sys.exit(1)
    
    # 解析 GPU
    gpu_ids = [4, 5, 6, 7]
    if args.gpu_ids:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
    print(f"GPU: {gpu_ids}")
    print("=" * 70)
    
    # 加载 LLM
    print("\n🔄 正在加载模型...")
    try:
        llm = Qwen3VL(path="Qwen3-8B", gpu_ids=gpu_ids)
        print("✅ 模型加载成功\n")
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        print(f"❌ 模型加载失败: {e}")
        sys.exit(1)
    
    # 创建 BaselineAgent（与 RAG/Web 测试一致）
    print("🔄 正在初始化 BaselineAgent...")
    baseline = BaselineAgent(llm)
    print("✅ BaselineAgent 初始化成功\n")
    
    # 加载 ALFWorld 配置
    if not os.path.exists(args.config):
        print(f"❌ 配置文件不存在: {args.config}")
        sys.exit(1)
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 初始化环境
    print("🔄 正在初始化 ALFWorld 环境...")
    try:
        env_type = config['env']['type']
        env_class = environment.get_environment(env_type)
        env = env_class(config, train_eval='eval_out_of_distribution')
        env = env.init_env(batch_size=1)
        print("✅ 环境初始化成功\n")
    except Exception as e:
        print(f"❌ 环境初始化失败: {e}")
        sys.exit(1)
    
    # 创建 ResultLogger（与 Agent 测试一致）
    result_logger = ResultLogger(log_dir="benchmark_results/alfworld_baseline")
    
    # 询问测试数量
    if args.num_games is None:
        while True:
            try:
                num_input = input("\n请输入要测试的游戏数量 (1-30): ").strip()
                args.num_games = int(num_input)
                if 1 <= args.num_games <= 30:
                    break
                print("⚠️ 请输入 1-30 之间的数字")
            except ValueError:
                print("⚠️ 请输入有效的数字")
    
    print(f"\n准备测试 {args.num_games} 个游戏场景...")
    
    # 运行测试
    try:
        for game_count in range(1, args.num_games + 1):
            result = run_baseline_game(baseline, env, game_count)
            result_logger.add_result(result)
            
            if game_count < args.num_games:
                time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n\n⚠️ 测试被用户中断")
    except Exception as e:
        logger.error(f"测试过程出错: {e}")
        print(f"\n❌ 测试出错: {e}")
    
    # 保存结果并打印总结（使用 ResultLogger）
    result_logger.finalize()
    
    print(f"\n注意: 使用 BaselineAgent（与 RAG/Web 消融实验一致）")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
