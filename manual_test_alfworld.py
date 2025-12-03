"""
手动测试 ALFworld - 交互式游戏模式
用于理解 ALFworld 的命令规则和环境反馈
"""

import os
import yaml
from loguru import logger

# 配置简单的日志
logger.remove()
logger.add(lambda msg: None)  # 禁用日志输出

try:
    import alfworld.agents.environment as environment
    ALFWORLD_AVAILABLE = True
except ImportError:
    ALFWORLD_AVAILABLE = False
    print("❌ ALFWorld 未安装，请先运行: pip install alfworld[full]")
    exit(1)


def print_banner():
    banner = """
╔════════════════════════════════════════════════════════╗
║                                                        ║
║         🎮 ALFworld 手动测试工具 🎮                   ║
║                                                        ║
║         Manual Test & Command Explorer                 ║
║                                                        ║
╚════════════════════════════════════════════════════════╝
    """
    print(banner)
    print("💡 这是一个交互式测试工具，让你手动输入命令来理解 ALFworld")
    print("💡 输入 'help' 查看可用命令")
    print("💡 输入 'quit' 或 'exit' 退出游戏")
    print("💡 输入 'reset' 重置到新任务")
    print("━" * 60)


def print_help():
    help_text = """
╔════════════════ 📖 ALFworld 命令指南 ════════════════╗

【移动命令】
  go to [object] [number]
  示例: go to fridge 1, go to desk 2

【查看命令】
  look                        - 查看当前房间
  examine [object] [number]   - 检查特定物体
  inventory                   - 查看手中物品

【交互命令】
  open [object] [number]      - 打开容器
  close [object] [number]     - 关闭容器
  
【拿取/放置命令】⚠️ 注意介词！
  take [object] [number] from [location] [number]
  
  put [object] [number] IN [container] [number]
  ↑ 用于容器: drawer, fridge, microwave, cabinet, safe, garbagecan
  示例: put apple 1 in fridge 1
  
  put [object] [number] ON [surface] [number]
  ↑ 用于平面: shelf, desk, countertop, bed, table
  示例: put pencil 1 on shelf 1

【特殊命令】
  clean [object] [number] with [tool] [number]
  heat [object] [number] with [tool] [number]
  cool [object] [number] with [tool] [number]
  toggle [object] [number]    - 开关灯等

【调试命令】
  help     - 显示此帮助
  reset    - 重置到新任务
  quit     - 退出程序

╚════════════════════════════════════════════════════════╝
"""
    print(help_text)


def play_game(env):
    """运行交互式游戏循环"""
    obs, info = env.reset()
    task_desc = obs[0].strip()
    
    print(f"\n{'='*60}")
    print("🎮 新游戏开始")
    print(f"{'='*60}\n")
    print(f"📝 任务描述:\n{task_desc}\n")
    print("━" * 60)
    print("💡 提示: 先用 'look' 查看房间，再决定行动")
    print("━" * 60)
    
    step_count = 0
    done = False
    
    while not done:
        step_count += 1
        
        # 获取用户输入
        try:
            user_input = input(f"\n[步骤 {step_count}] 你的命令 > ").strip()
        except EOFError:
            print("\n\n👋 再见！")
            break
        
        if not user_input:
            continue
        
        # 处理特殊命令
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 退出游戏...")
            return 'quit'
        
        if user_input.lower() == 'help':
            print_help()
            continue
        
        if user_input.lower() == 'reset':
            print("\n🔄 重置游戏...")
            return 'reset'
        
        # 执行命令
        try:
            obs, scores, dones, infos = env.step([user_input])
            observation = obs[0]
            score = scores[0]
            done = dones[0]
            
            # 显示结果
            print(f"\n{'─'*60}")
            if "Nothing happens" in observation:
                print("❌ 执行失败:")
                print(f"   {observation}")
                print("\n💡 可能的原因:")
                if "put" in user_input.lower():
                    if " in " in user_input.lower():
                        print("   - 可能用错了介词？shelf/desk 等表面应该用 'on' 而不是 'in'")
                    elif " on " in user_input.lower():
                        print("   - 可能用错了介词？drawer/fridge 等容器应该用 'in' 而不是 'on'")
                    else:
                        print("   - 你在正确的位置吗？需要先 'go to' 目标位置")
                elif "open" in user_input.lower() or "take" in user_input.lower():
                    print("   - 你移动到物体旁边了吗？需要先 'go to [object] [number]'")
                else:
                    print("   - 命令格式可能不正确")
                    print("   - 物体可能不存在或已经在目标状态")
            else:
                print("✅ 执行成功:")
                print(f"   {observation}")
            
            if done:
                print(f"\n{'='*60}")
                if score > 0:
                    print(f"🎉 任务完成！得分: {score}")
                    print(f"📊 用了 {step_count} 步")
                else:
                    print(f"❌ 任务失败，得分: {score}")
                print(f"{'='*60}")
                return 'done'
                
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            print("💡 输入 'help' 查看命令格式")
    
    return 'done'


def main():
    print_banner()
    
    # 加载配置
    config_path = "configs/base_config.yaml"
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 初始化环境
    print("🔄 正在初始化 ALFworld 环境...")
    try:
        env_type = config['env']['type']
        env_class = environment.get_environment(env_type)
        env = env_class(config, train_eval='eval_out_of_distribution')
        env = env.init_env(batch_size=1)
        print("✅ 环境初始化成功\n")
    except Exception as e:
        print(f"❌ 环境初始化失败: {e}")
        print("\n💡 提示: 请确保已运行 'alfworld-download' 下载游戏数据")
        return
    
    # 游戏循环
    while True:
        result = play_game(env)
        
        if result == 'quit':
            break
        elif result == 'reset':
            continue
        elif result == 'done':
            # 询问是否继续
            try:
                choice = input("\n🎮 再来一局？(y/n) > ").strip().lower()
                if choice not in ['y', 'yes']:
                    break
            except (EOFError, KeyboardInterrupt):
                break
    
    print("\n" + "="*60)
    print("👋 感谢使用 ALFworld 手动测试工具！")
    print("="*60)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ 程序被中断")
        print("👋 再见！")
