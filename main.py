import json
import os
import sys
from datetime import datetime

import yaml
from loguru import logger

from llm import Qwen3
from agent import Agent
from log_config import setup_logger

# 修复中文输入问题
try:
    import readline
except ImportError:
    print("system", "请使用 Python 3.9 或更高版本")

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


def print_banner():
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

    model_path = "Qwen/Qwen3-8B"
    
    try:
        logger.info("正在加载模型...")
        llm = Qwen3(model_path, gpu_ids=[0])
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
    
    logger.info("Agent ready, chat session started")

    while True:
        try:
            user_input = input("\n💬 您: ").strip()
            
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
            
            logger.info(f"User input: {user_input}")
            
            agent_output, agent_history = agent.text(user_input, agent_history)

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

if __name__ == '__main__':
    main()  