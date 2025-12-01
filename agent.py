from typing import Dict, List, Optional, Tuple, Union
import json5

from llm import BaseLLM
from tool import Tools

# define tool description template, to be used to introduce available tools to the model
# name_for_model: tool name, to be called by model
# name_for_human: tool name, to be read by human
# description_for_model: tool function description
# parameters: tool parameter description
TOOL_DESC = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters} Format the arguments as a JSON object."""

# ReAct
# tool_description: all tool description
# tool_names: all tool names
REACT_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

{tool_description}

CRITICAL RULES:
1. You can ONLY use ONE tool at a time in each response
2. Each field (Thought, Action, Action Input, Final Answer) must appear EXACTLY ONCE
3. After using a tool, the system will show you the result and let you continue
4. NEVER include multiple "Action:" or "Action Input:" lines in one response
5. If you need multiple tools, use them one by one across multiple turns

Response Format Requirements:
- When you need to use a tool:
  Thought: [your reasoning about what to do next]
  Action: [exactly ONE tool name from: {tool_names}]
  Action Input: [the input for that ONE tool only]

- When you have the final answer:
  Thought: [your reasoning about why this is the final answer]
  Final Answer: [your complete response to the user]

Remember: ONE tool per response. The system will give you another chance to act after each tool use.

"""

class Agent:
    def __init__(self, model, rag_db_path=None, max_step=10) -> None:
        self.tool = Tools(rag_db_path=rag_db_path)
        self.system_prompt = self.build_system_input()  
        self.model = model
        self.max_step = max_step
    
    def build_system_input(self):
        tool_description, tool_names = [], []

        for tool in self.tool.toolConfig:
            tool_description.append(TOOL_DESC.format(**tool))
            tool_names.append(tool['name_for_model'])
        tool_description = '\n\n'.join(tool_description)
        tool_names = ', '.join(tool_names)
        system_prompt = REACT_PROMPT.format(tool_description=tool_description, tool_names=tool_names)
        return system_prompt

    def parse_latest_plugin_call(self, text: str) -> Tuple[str, str, str]:
        """
        Parse the latest tool calls from the model's output. 
        Args: text (str): The output text of the model. 
        Returns: Tuple[str, str, str]: (Plugin name, plugin parameter, warning)
        """
        plugin_name, plugin_args, warning = '', '', ''
        
        # 查找所有 Action: 出现的位置
        action_positions = []
        pos = 0
        while True:
            pos = text.find('\nAction:', pos)
            if pos == -1:
                break
            action_positions.append(pos)
            pos += 1
        
        # 如果检测到多个Action,发出警告并只使用第一个
        if len(action_positions) > 1:
            # 提取所有被忽略的工具名
            ignored_tools = []
            for i in range(1, len(action_positions)):
                pos = action_positions[i]
                action_start = pos + len('\nAction:')
                action_end = text.find('\n', action_start)
                if action_end == -1:
                    action_end = len(text)
                ignored_tool = text[action_start:action_end].strip()
                if ignored_tool:
                    ignored_tools.append(ignored_tool)
            
            ignored_list = ', '.join(ignored_tools) if ignored_tools else 'unknown tools'
            warning = f"⚠️ Warning: Multiple actions detected. You can only use ONE tool at a time. Using the first action only. Ignored tools: {ignored_list}"
            print(warning)
        
        # 使用第一个Action
        if action_positions:
            action_pos = action_positions[0]
            action_input_pos = text.find('\nAction Input:', action_pos)
            
            if action_input_pos != -1 and action_pos < action_input_pos:
                # 提取 Action: 后面那一行的内容作为工具名
                action_start = action_pos + len('\nAction:')
                action_end = text.find('\n', action_start)
                if action_end == -1 or action_end > action_input_pos:
                    action_end = action_input_pos
                plugin_name = text[action_start:action_end].strip()
                
                # 提取 Action Input: 后面那一行的内容作为参数
                input_start = action_input_pos + len('\nAction Input:')
                input_end = text.find('\n', input_start)
                if input_end == -1:
                    input_end = len(text)
                plugin_args = text[input_start:input_end].strip()
        
        return plugin_name, plugin_args, warning

    def call_plugin(self, plugin_name: str, plugin_args: str) -> str:
        """
        Call the specified plugin (tool). 
        Args: 
            plugin_name (str): The name of the plugin to be called. 
            plugin_args (str): The parameter of the plugin, which is a string in JSON format. 
        Returns: str: The observation result after the tool is executed. 
        """
        plugin_args = json5.loads(plugin_args) if plugin_args else {}
        
        if plugin_name == 'google_search':
            return '\nObservation:' + self.tool.google_search(**plugin_args)
        elif plugin_name == 'query_weather':
            return '\nObservation:' + self.tool.query_weather(**plugin_args)
        elif plugin_name == 'query_time':
            return '\nObservation:' + self.tool.query_time(**plugin_args)
        elif plugin_name == 'basic_calculator':
            return '\nObservation:' + self.tool.basic_calculator(**plugin_args)
        elif plugin_name == 'trig_calculator':
            return '\nObservation:' + self.tool.trig_calculator(**plugin_args)
        elif plugin_name == 'matrix_calculator':
            return '\nObservation:' + self.tool.matrix_calculator(**plugin_args)
        elif plugin_name == 'integral_calculator':
            return '\nObservation:' + self.tool.integral_calculator(**plugin_args)
        elif plugin_name == 'knowledge_base_query':
            return '\nObservation:' + self.tool.knowledge_base_query(**plugin_args)
        else:
            print(f"✗ unknown tool: {plugin_name}")
            return f'\nObservation: unknown tool: {plugin_name}'
        
    def check_response(self, response: str) -> bool:
        """
        Check if the model's response contains the required fields.
        Args:
            response (str): The model's response text.
        Returns:
            bool: True if the response contains the required fields, False otherwise.
        """
        # 计算各字段出现次数
        thought_count = response.count('Thought:')
        action_count = response.count('Action:')
        action_input_count = response.count('Action Input:')
        final_answer_count = response.count('Final Answer:')
        
        # 检查Thought出现次数(应该正好1次)
        if thought_count == 0:
            print("❌ Format Error: Missing 'Thought:'")
            return False
        elif thought_count > 1:
            print(f"❌ Format Error: 'Thought:' appears {thought_count} times, should appear exactly once")
            return False
        
        # 如果有Action,检查出现次数和Action Input
        if action_count > 0:
            if action_count > 1:
                print(f"❌ Format Error: 'Action:' appears {action_count} times, should appear exactly once")
                return False
            
            if action_input_count == 0:
                print("❌ Format Error: 'Action:' found but 'Action Input:' is missing")
                return False
            elif action_input_count > 1:
                print(f"❌ Format Error: 'Action Input:' appears {action_input_count} times, should appear exactly once")
                return False
        
        # 检查Final Answer出现次数
        if final_answer_count > 1:
            print(f"❌ Format Error: 'Final Answer:' appears {final_answer_count} times, should appear exactly once")
            return False
        
        return True
    
    def split_response(self, response: str) -> Tuple[str, str]:
        """
        Split the model's response into thinking part and final answer part.
        Args:
            response (str): The model's response text.
        Returns:
            Tuple[str, str]: (response without thinking tags, thinking content)
        """
        thinking_content = ''
        clean_response = response
        
        # 查找 <think> 和 </think> 标签
        think_start = response.find('<think>')
        think_end = response.find('</think>')
        
        if think_start != -1 and think_end != -1 and think_start < think_end:
            # 提取思考内容
            thinking_content = response[think_start + len('<think>'):think_end].strip()
            
            # 移除思考标签和内容,保留其他部分
            clean_response = response[:think_start] + response[think_end + len('</think>'):]
            clean_response = clean_response.strip()
        
        return clean_response, thinking_content

        
        
    def text(self, text: str, history: List = []) -> Tuple[str, List]:
        response = "\nQuestion:" + text

        # 'his' is the updated history
        step_count = 0
        while step_count < self.max_step:
            new_response, history = self.model.chat(response, history=history, meta_instruction=self.system_prompt)
            response = ""
            no_thinking_response ,thinking_response = self.split_response(new_response)

            print(f"======= history: {history}")
            print(f"======= Agent Think: {thinking_response}")
            print(f"======= Agent Response: {no_thinking_response}")

            if not self.check_response(no_thinking_response):
                response = "Your previous response was not in the correct format. Please follow the specified format strictly."
                continue

            plugin_name, plugin_args, warning = self.parse_latest_plugin_call(no_thinking_response)
            print(f"======= Parsed tool call - Name: {plugin_name}, Args: {plugin_args}")  # Debug: print parsed tool call
            if plugin_name:
                try:
                    tool_observation = self.call_plugin(plugin_name, plugin_args)
                    # 如果有警告,添加到observation中
                    if warning:
                        tool_observation += f"\n{warning}"
                except Exception as e:
                    print("❌ Error", f"Error calling tools {plugin_name}: {e}")
                    tool_observation = f"\nObservation: Error calling tools {plugin_name}: {e}. Please try another tool or correct the arguments."
                response += tool_observation
                print(f"======= Agent tool_observation: {tool_observation}")  # Debug: print the observation from the tool
            else:
                break
            step_count += 1

        return no_thinking_response, history