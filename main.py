import json
import os
import sys
from datetime import datetime
from typing import List, Optional

import yaml
from loguru import logger

from llm import Qwen3, Qwen3VL
from agent import Agent
from log_config import setup_logger

# 修复中文输入问题并配置 readline
try:
    import readline
    import glob
    
    # 配置 readline 的路径补全功能
    def path_completer(text, state):
        """路径补全函数"""
        # 如果输入以 @image: 开头，补全路径部分
        if text.startswith('@image:'):
            path_part = text[7:]  # 去掉 @image: 前缀
            matches = glob.glob(path_part + '*')
            matches = ['@image:' + m for m in matches]
        else:
            # 普通路径补全
            matches = glob.glob(text + '*')
        
        # 为目录添加斜杠
        matches = [m + '/' if os.path.isdir(m.replace('@image:', '')) else m for m in matches]
        
        try:
            return matches[state]
        except IndexError:
            return None
    
    # 设置 Tab 补全
    readline.set_completer(path_completer)
    readline.parse_and_bind("tab: complete")
    
    # 设置补全时的分隔符（包含 @ 以支持 @image: 命令）
    readline.set_completer_delims(' \t\n;')
    
    READLINE_AVAILABLE = True
except ImportError:
    READLINE_AVAILABLE = False
    print("system", "请使用 Python 3.9 或更高版本")

# 尝试导入PIL用于剪贴板支持
try:
    from PIL import ImageGrab, Image
    CLIPBOARD_SUPPORTED = True
except ImportError:
    CLIPBOARD_SUPPORTED = False
    logger.warning("PIL not available, clipboard image support disabled. Install with: pip install Pillow")

DEFAULT_CONFIG = {
    "security": {
        "enable_prompt_guard": True,
        "lock_on_violation": True,
    },
    "google": {
        "credentials_path": "jsons/goole_search.json",
    },
}


def _merge_dict(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str = "config.yaml") -> dict:
    if not os.path.exists(path):
        return DEFAULT_CONFIG.copy()
    with open(path, "r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    return _merge_dict(DEFAULT_CONFIG, data)


def save_clipboard_image() -> Optional[str]:
    """从剪贴板保存图片，返回保存的文件路径"""
    if not CLIPBOARD_SUPPORTED:
        return None
    
    try:
        img = ImageGrab.grabclipboard()
        if img is None or not isinstance(img, Image.Image):
            return None
        
        # 创建临时目录
        temp_dir = "temp_images"
        os.makedirs(temp_dir, exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"clipboard_{timestamp}.png"
        filepath = os.path.join(temp_dir, filename)
        
        # 保存图片
        img.save(filepath)
        logger.info(f"Clipboard image saved to: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Failed to save clipboard image: {e}")
        return None


def print_banner(multimodal_support: bool = False):
    """打印欢迎横幅"""
    banner = """
╔════════════════════════════════════════════════════════╗
║                                                        ║
║            🤖 AIAA3102 Agent System 🤖                ║
║                                                        ║
║              Powered by Qwen3-8B Model                 ║
║                                                        ║
╚════════════════════════════════════════════════════════╝
    """
    print(banner)
    print("💡 提示: 输入 'exit' 或 'quit' 退出程序")
    print("💡 提示: 按 Ctrl+C 也可以随时退出")
    if READLINE_AVAILABLE:
        print("💡 提示: 按 Tab 键可自动补全文件路径")
    
    if multimodal_support:
        print("\n📷 图片输入功能已启用:")
        print("   • @image:<路径>  - 添加图片文件 (支持Tab补全)")
        print("   • @paste         - 从剪贴板添加图片" + (" (不可用)" if not CLIPBOARD_SUPPORTED else ""))
        print("   • @clear         - 清空图片列表")
        print("   • @show          - 显示当前图片")
    
    print("━" * 60)

def print_message(role: str, content: str):
    """格式化打印消息"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    if role == "user":
        print(f"\n┌─ 👤 用户 [{timestamp}]")
        print(f"│ {content}")
        print("└" + "─" * 58)
    elif role == "agent":
        print(f"\n┌─ 🤖 Agent [{timestamp}]")
        for line in content.split('\n'):
            print(f"│ {line}")
        print("└" + "─" * 58)
    elif role == "system":
        print(f"\n⚙️  {content}")
    elif role == "error":
        print(f"\n❌ 错误: {content}")

def main():
    # 初始化日志系统
    setup_logger()
    logger.info("=" * 60)
    logger.info("AIAA3102 Agent System started")
    logger.info("=" * 60)

    config = load_config()
    security_config = config.get("security", {})
    google_config = config.get("google", {})
    logger.info(
        "Security config loaded | prompt_guard={}, lock_on_violation={}",
        security_config.get("enable_prompt_guard"),
        security_config.get("lock_on_violation"),
    )
    
    credentials_path = google_config.get("credentials_path")
    if credentials_path:
        os.environ.setdefault("GOOGLE_SEARCH_CREDENTIALS", credentials_path)
        if os.path.exists(credentials_path):
            try:
                with open(credentials_path, "r", encoding="utf-8") as fp:
                    google_credentials = json.load(fp) or {}
                if not google_credentials.get("api_key"):
                    logger.warning(
                        "Google 搜索 API key 未配置 (文件: %s)，google_search 工具将不可用。",
                        credentials_path,
                    )
            except Exception as exc:
                logger.warning(
                    "无法读取 Google 搜索配置文件 %s: %s",
                    credentials_path,
                    exc,
                )

    # model_path = "Qwen/Qwen3-8B"
    
    try:
        logger.info("正在加载模型...")
        llm = Qwen3VL()
        logger.success("Model loaded successfully")
        
        # llm = Qwen3(model_path, gpu_ids=[0, 5, 8])
    except Exception as e:
        logger.error(f"模型初始化失败: {e}")
        return

    # 检查是否有RAG数据库(优先使用wiki_vector_db)
    rag_db_path = None
    if os.path.exists("rag/wiki_vector_db"):
        rag_db_path = "rag/wiki_vector_db"
        logger.info(f"发现AI维基知识库: {rag_db_path}")
    elif os.path.exists("rag/vector_db"):
        rag_db_path = "rag/vector_db"
        logger.info(f"发现知识库: {rag_db_path}")
    else:
        logger.warning("未找到知识库,将不启用RAG功能")
    
    agent = Agent(llm, rag_db_path=rag_db_path, security_config=security_config)
    
    agent_history = []
    current_images: List[str] = []  # 当前会话的图片列表
    
    # 打印欢迎信息
    print_banner(multimodal_support=agent.supports_multimodal)
    
    logger.info("Agent ready, chat session started")

    while True:
        try:
            # 构建提示符
            cwd = os.getcwd()
            if agent.supports_multimodal and current_images:
                prompt = f"[📷 {len(current_images)}张 | {cwd}]\n💬 您: "
            elif agent.supports_multimodal:
                prompt = f"[无图片 | {cwd}]\n💬 您: "
            else:
                prompt = f"[{cwd}]\n💬 您: "
            
            user_input = input(prompt).strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['exit', 'quit']:
                logger.info("User requested exit")
                break

            if user_input.lower() in ['reset guard', 'reset_guard', '/reset_guard']:
                logger.info("Operator requested prompt guard reset")
                was_blocked = agent.reset_security()
                if was_blocked:
                    print("\n🔓 Prompt Guard: 已清除锁定，可以继续输入。")
                else:
                    print("\nℹ️ Prompt Guard: 当前未锁定，无需重置。")
                continue
            
            # 处理图片相关命令
            if user_input.startswith('@'):
                if user_input.startswith('@image:'):
                    img_path = user_input[7:].strip()
                    if os.path.exists(img_path):
                        abs_path = os.path.abspath(img_path)
                        current_images.append(abs_path)
                        print(f"✅ 已添加图片: {abs_path}")
                        logger.info(f"Image added: {abs_path}")
                    else:
                        print(f"❌ 图片不存在: {img_path}")
                        logger.warning(f"Image not found: {img_path}")
                    continue
                
                elif user_input == '@paste':
                    if not CLIPBOARD_SUPPORTED:
                        print("❌ 剪贴板功能不可用，请安装 Pillow: pip install Pillow")
                        continue
                    
                    saved_path = save_clipboard_image()
                    if saved_path:
                        current_images.append(os.path.abspath(saved_path))
                        print(f"✅ 已从剪贴板添加图片: {saved_path}")
                        logger.info(f"Image added from clipboard: {saved_path}")
                    else:
                        print("❌ 剪贴板中没有图片")
                    continue
                
                elif user_input == '@clear':
                    count = len(current_images)
                    current_images.clear()
                    print(f"✅ 已清空 {count} 张图片")
                    logger.info(f"Cleared {count} images")
                    continue
                
                elif user_input == '@show':
                    if current_images:
                        print(f"\n📷 当前图片列表 ({len(current_images)}张):")
                        for i, img in enumerate(current_images, 1):
                            print(f"   {i}. {img}")
                    else:
                        print("📷 当前没有图片")
                    continue
            
            logger.info(f"User input: {user_input} | images: {len(current_images)}")
            
            # 调用 agent，传入图片
            agent_output, agent_history = agent.text(
                user_input, 
                agent_history, 
                images=current_images if current_images else None
            )
            
            # 处理后清空图片（仅用于首次提问）
            if current_images:
                logger.info(f"Images used in this query, clearing for next turn")
                current_images.clear()

            final_answer_marker = "Final Answer:"
            final_answer = agent_output.rfind(final_answer_marker)
            if final_answer != -1:
                final_answer = agent_output[final_answer + len(final_answer_marker):].strip()
            else:
                final_answer = agent_output.strip()

            print(f"\n🤖 Agent: {final_answer}")

        except KeyboardInterrupt:
            logger.info("User interrupted (Ctrl+C)")
            break
        except Exception as e:
            logger.exception(f"Unexpected error in main loop: {e}")
            continue
    
    logger.info("AIAA3102 Agent System stopped")
    logger.info("=" * 60)


def testALFworld():
    """使用 ALFworld 测试案例测试 Agent"""
    # 初始化日志系统
    setup_logger()
    logger.info("=" * 60)
    logger.info("AIAA3102 Agent System - ALFworld Test Mode")
    logger.info("=" * 60)

    # 加载配置
    config = load_config()
    security_config = config.get("security", {})
    google_config = config.get("google", {})
    
    # 设置 Google 搜索凭证
    credentials_path = google_config.get("credentials_path")
    if credentials_path and os.path.exists(credentials_path):
        os.environ.setdefault("GOOGLE_SEARCH_CREDENTIALS", credentials_path)
    
    # 加载测试数据
    test_file = "benchmark/ALFworld/test_cases_valid_unseen_30.json"
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

    # 初始化模型
    try:
        logger.info("正在加载模型...")
        print("🔄 正在加载 Qwen3-VL 模型...")
        llm = Qwen3VL()
        logger.success("Model loaded successfully")
        print("✅ 模型加载成功\n")
    except Exception as e:
        logger.error(f"模型初始化失败: {e}")
        print(f"❌ 模型加载失败: {e}")
        return

    # 检查 RAG 数据库
    rag_db_path = None
    if os.path.exists("rag/wiki_vector_db"):
        rag_db_path = "rag/wiki_vector_db"
        logger.info(f"发现AI维基知识库: {rag_db_path}")
    elif os.path.exists("rag/vector_db"):
        rag_db_path = "rag/vector_db"
        logger.info(f"发现知识库: {rag_db_path}")
    
    # 创建 Agent
    agent = Agent(llm, rag_db_path=rag_db_path, security_config=security_config)
    
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
        print("  4. 查看案例列表")
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
            # 测试所有案例
            confirm = input(f"\n确认要测试全部 {len(test_cases)} 个案例吗? (y/n): ").strip().lower()
            if confirm == 'y':
                test_all_cases(agent, test_cases)
            else:
                print("已取消")
        
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
        
        elif choice == '4':
            # 查看案例列表
            print(f"\n{'='*60}")
            print(f"{'编号':<6} {'任务类型':<30} {'任务描述':<20}")
            print(f"{'='*60}")
            for i, case in enumerate(test_cases, 1):
                task_type = case.get('task_type', 'unknown')
                task_desc = case.get('task_desc', 'No description')
                # 截断过长的描述
                if len(task_desc) > 40:
                    task_desc = task_desc[:37] + "..."
                print(f"{i:<6} {task_type:<30} {task_desc}")
            print(f"{'='*60}")
        
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
        print(f"\n💬 发送给 Agent: {user_query}")
        print("\n🤖 Agent 思考中...")
        
        agent_output, agent_history = agent.text(user_query, agent_history)
        
        # 提取最终答案
        final_answer_marker = "Final Answer:"
        final_answer = agent_output.rfind(final_answer_marker)
        if final_answer != -1:
            final_answer = agent_output[final_answer + len(final_answer_marker):].strip()
        else:
            final_answer = agent_output.strip()
        
        print(f"\n{'='*60}")
        print("🤖 Agent 回答:")
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


if __name__ == '__main__':
    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        testALFworld()
    else:
        main()  