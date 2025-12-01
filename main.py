import os
import sys
from llm import Qwen3
from agent import Agent
from datetime import datetime

# 修复中文输入问题
try:
    import readline
except ImportError:
    print("system", "请使用 Python 3.9 或更高版本")

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
    print_banner()
    
    model_path = "Qwen/Qwen3-8B"
    
    try:
        print_message("system", "正在加载模型...")
        llm = Qwen3(model_path)
        
        # llm = Qwen3(model_path, gpu_ids=[0, 5, 8])
    except Exception as e:
        print_message("error", f"模型初始化失败: {e}")
        return

    # 检查是否有RAG数据库(优先使用wiki_vector_db)
    rag_db_path = None
    if os.path.exists("rag/wiki_vector_db"):
        rag_db_path = "rag/wiki_vector_db"
        print_message("system", f"发现AI维基知识库: {rag_db_path}")
    elif os.path.exists("rag/vector_db"):
        rag_db_path = "rag/vector_db"
        print_message("system", f"发现知识库: {rag_db_path}")
    else:
        print_message("system", "未找到知识库,将不启用RAG功能")
    
    agent = Agent(llm, rag_db_path=rag_db_path)
    
    agent_history = []
    
    print("\n" + "━" * 60)
    print("🚀 Agent 已就绪,开始对话吧!")
    print("━" * 60)

    while True:
        try:
            user_input = input("\n💬 您: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['exit', 'quit']:
                print_message("system", "👋 再见!")
                break
            
            print_message("system", "🤔 思考中...")
            agent_output, agent_history = agent.text(user_input, agent_history)

            final_answer_marker = "Final Answer:"
            final_answer = agent_output.rfind(final_answer_marker)
            if final_answer != -1:
                final_answer = agent_output[final_answer + len(final_answer_marker):].strip()
            else:
                final_answer = agent_output.strip()

            print_message("agent", final_answer)

        except KeyboardInterrupt:
            print_message("system", "\n👋 再见!")
            break
        except Exception as e:
            print_message("error", str(e))
            continue

if __name__ == '__main__':
    main()  