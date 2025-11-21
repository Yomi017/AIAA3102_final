from llm import Qwen3
from agent import Agent

def main():
    
    model_path = "Qwen3-8B"
    try:
        llm = Qwen3(model_path)
    except Exception as e:
        print(f"Error initializing model: {e}")
        return

    agent = Agent(llm)
    print("Agent initialized. Input 'exit' or 'quit' to quit.")

    agent_history = []
    while True:
        try:
            user_input = input("input: ")
            if user_input.lower() in ['exit', 'quit']:
                print("Exiting...")
                break
            agent_output, agent_history = agent.text(user_input, agent_history)

            final_answer_marker = "Final Answer:"
            final_answer = agent_output.rfind(final_answer_marker)
            if final_answer != -1:
                final_answer = agent_output[final_answer + len(final_answer_marker):].strip()
            else:
                final_answer = agent_output.strip()

            print("\nAgent:", final_answer)  

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

if __name__ == '__main__':
    main()  