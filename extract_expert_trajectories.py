"""
从 ALFworld 数据文件中提取专家轨迹（标准答案）
"""

import os
import json
import glob
from pathlib import Path


def find_alfworld_data():
    """查找 ALFworld 数据目录"""
    possible_paths = [
        os.path.expanduser("~/.cache/alfworld/json_2.1.1"),
        os.path.expanduser("~/.alfworld/json_2.1.1"),
        "data/json_2.1.1",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None


def load_game_file(game_path):
    """加载游戏文件并提取专家轨迹"""
    try:
        with open(game_path, 'r') as f:
            game_data = json.load(f)
        
        # 提取任务信息
        task_type = game_data.get('task_type', 'Unknown')
        pddl_params = game_data.get('pddl_params', {})
        
        # 提取专家动作
        expert_plan = game_data.get('plan', {})
        high_pddl = expert_plan.get('high_pddl', [])
        low_actions = expert_plan.get('low_actions', [])
        
        return {
            'task_type': task_type,
            'pddl_params': pddl_params,
            'high_pddl': high_pddl,
            'low_actions': low_actions,
            'game_data': game_data
        }
    except Exception as e:
        print(f"  ❌ 加载失败: {e}")
        return None


def extract_commands_from_plan(game_info):
    """从计划中提取可执行命令"""
    commands = []
    
    # 从 low_actions 中提取
    low_actions = game_info.get('low_actions', [])
    for action_group in low_actions:
        if isinstance(action_group, list):
            for action in action_group:
                if isinstance(action, dict):
                    api_action = action.get('api_action', {})
                    # 提取动作和对象
                    action_name = api_action.get('action', '')
                    obj = api_action.get('objectId', '')
                    
                    if action_name and obj:
                        # 构建命令
                        if action_name in ['GotoLocation', 'Navigation']:
                            cmd = f"go to {obj}"
                        elif action_name == 'PickupObject':
                            cmd = f"take {obj}"
                        elif action_name == 'PutObject':
                            receptacle = api_action.get('receptacleObjectId', '')
                            cmd = f"put {obj} in/on {receptacle}"
                        elif action_name == 'OpenObject':
                            cmd = f"open {obj}"
                        elif action_name == 'CloseObject':
                            cmd = f"close {obj}"
                        elif action_name == 'ToggleObjectOn':
                            cmd = f"toggle {obj} on"
                        elif action_name == 'ToggleObjectOff':
                            cmd = f"toggle {obj} off"
                        else:
                            cmd = f"{action_name.lower()} {obj}"
                        
                        commands.append(cmd)
    
    return commands


def show_expert_trajectories(data_path, num_games=5, split='valid_unseen'):
    """显示专家轨迹"""
    
    print("╔════════════════════════════════════════════════════════╗")
    print("║                                                        ║")
    print("║     📚 ALFworld 专家轨迹 (从数据文件提取) 📚        ║")
    print("║                                                        ║")
    print("╚════════════════════════════════════════════════════════╝\n")
    
    # 查找游戏文件
    split_path = os.path.join(data_path, split)
    if not os.path.exists(split_path):
        print(f"❌ 找不到数据目录: {split_path}")
        return
    
    # 获取所有子目录
    task_dirs = sorted([d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))])
    
    print(f"📂 数据目录: {split_path}")
    print(f"📊 找到 {len(task_dirs)} 个任务类型\n")
    
    game_count = 0
    
    for task_dir in task_dirs:
        if game_count >= num_games:
            break
        
        task_path = os.path.join(split_path, task_dir)
        
        # ALFworld 数据结构: task_dir/trial_XXX/traj_data.json
        trial_dirs = sorted([d for d in os.listdir(task_path) if os.path.isdir(os.path.join(task_path, d))])
        
        if not trial_dirs:
            continue
        
        # 获取第一个 trial 的 traj_data.json
        trial_path = os.path.join(task_path, trial_dirs[0])
        game_file = os.path.join(trial_path, 'traj_data.json')
        
        if not os.path.exists(game_file):
            continue
        
        game_count += 1
        
        print(f"\n{'='*60}")
        print(f"🎮 游戏 {game_count}/{num_games}")
        print(f"{'='*60}")
        print(f"📁 文件: {os.path.basename(game_file)}")
        print(f"📝 任务类型: {task_dir}\n")
        
        # 加载游戏数据
        game_info = load_game_file(game_file)
        
        if not game_info:
            continue
        
        # 显示任务参数
        pddl_params = game_info.get('pddl_params', {})
        print(f"🎯 任务参数:")
        for key, value in pddl_params.items():
            print(f"   {key}: {value}")
        
        # 显示高层计划
        high_pddl = game_info.get('high_pddl', [])
        if high_pddl:
            print(f"\n📋 高层计划 (PDDL):")
            for i, action in enumerate(high_pddl, 1):
                action_name = action.get('planner_action', {}).get('action', '')
                print(f"   {i}. {action_name}")
        
        # 显示低层动作
        low_actions = game_info.get('low_actions', [])
        if low_actions:
            print(f"\n✅ 专家动作序列 (共 {len(low_actions)} 个动作组):")
            
            action_num = 0
            for group_idx, action_group in enumerate(low_actions):
                if isinstance(action_group, list):
                    print(f"\n  【动作组 {group_idx + 1}】")
                    for action in action_group:
                        if isinstance(action, dict):
                            action_num += 1
                            api_action = action.get('api_action', {})
                            action_name = api_action.get('action', 'Unknown')
                            
                            # 格式化显示
                            if action_name == 'GotoLocation':
                                obj = api_action.get('objectId', '')
                                print(f"    {action_num}. go to {obj}")
                            elif action_name == 'PickupObject':
                                obj = api_action.get('objectId', '')
                                print(f"    {action_num}. take {obj}")
                            elif action_name == 'PutObject':
                                obj = api_action.get('objectId', '')
                                receptacle = api_action.get('receptacleObjectId', '')
                                print(f"    {action_num}. put {obj} in/on {receptacle}")
                            elif action_name == 'OpenObject':
                                obj = api_action.get('objectId', '')
                                print(f"    {action_num}. open {obj}")
                            elif action_name == 'CloseObject':
                                obj = api_action.get('objectId', '')
                                print(f"    {action_num}. close {obj}")
                            elif action_name == 'CleanObject':
                                obj = api_action.get('objectId', '')
                                print(f"    {action_num}. clean {obj}")
                            elif action_name == 'HeatObject':
                                obj = api_action.get('objectId', '')
                                print(f"    {action_num}. heat {obj}")
                            elif action_name == 'CoolObject':
                                obj = api_action.get('objectId', '')
                                print(f"    {action_num}. cool {obj}")
                            elif action_name == 'ToggleObjectOn':
                                obj = api_action.get('objectId', '')
                                print(f"    {action_num}. toggle {obj} on")
                            elif action_name == 'ToggleObjectOff':
                                obj = api_action.get('objectId', '')
                                print(f"    {action_num}. toggle {obj} off")
                            else:
                                print(f"    {action_num}. {action_name}: {api_action}")
        
        print(f"\n{'─'*60}")
        
        # 询问是否继续
        if game_count < num_games:
            try:
                choice = input(f"\n继续查看下一个? (y/n) > ").strip().lower()
                if choice not in ['y', 'yes', '']:
                    break
            except (EOFError, KeyboardInterrupt):
                break


def main():
    # 查找数据目录
    data_path = find_alfworld_data()
    
    if not data_path:
        print("❌ 找不到 ALFworld 数据目录")
        print("\n可能的位置:")
        print("  - ~/.cache/alfworld/json_2.1.1")
        print("  - ~/.alfworld/json_2.1.1")
        print("\n💡 请确保已运行: alfworld-download")
        return
    
    print(f"✅ 找到数据目录: {data_path}\n")
    
    # 询问查看数量
    while True:
        try:
            num_input = input("请输入要查看的游戏数量 (1-20): ").strip()
            if not num_input:
                num_games = 5
                break
            num_games = int(num_input)
            if 1 <= num_games <= 20:
                break
            print("⚠️ 请输入 1-20 之间的数字")
        except ValueError:
            print("⚠️ 请输入有效的数字")
    
    # 显示专家轨迹
    show_expert_trajectories(data_path, num_games)
    
    print("\n" + "="*60)
    print("👋 完成！")
    print("="*60)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ 程序被中断")
        print("👋 再见！")
