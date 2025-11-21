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

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
Thought: I now know the final answer
Final Answer: the final answer to the original input question
"""

class Agent:
    def __init__(self, model) -> None:
        self.tool = Tools()
        self.system_prompt = self.build_system_input()  
        self.model = model
    
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
        Returns: Tuple[str, str, str]: (Plugin name, plugin parameter, parsed text) ""
        """
        plugin_name, plugin_args = '', ''
        i = text.rfind('\nAction:')
        j = text.rfind('\nAction Input:')
        k = text.rfind('\nObservation:')
        if 0 <= i < j: 
            if k < j: 
                text = text.rstrip() + '\nObservation:'  
            k = text.rfind('\nObservation:')
            plugin_name = text[i + len('\nAction:') : j].strip()
            plugin_args = text[j + len('\nAction Input:') : k].strip()
            text = text[:k]
        return plugin_name, plugin_args, text

    def call_plugin(self, plugin_name: str, plugin_args: str) -> str:
        """
        Call the specified plugin (tool). 
        Args: 
            plugin_name (str): The name of the plugin to be called. 
            plugin_args (str): The parameter of the plugin, which is a string in JSON format. 
        Returns: str: The observation result after the tool is executed. 
        """
        plugin_args = json5.loads(plugin_args)
        if plugin_name == 'google_search':
            return '\nObservation:' + self.tool.google_search(**plugin_args)
        
    def text(self, text: str, history: List = []) -> Tuple[str, List]:
        text = "\nQuestion:" + text

        # 'his' is the updated history
        response, his = self.model.chat(text, history=history, meta_instruction=self.system_prompt)

        plugin_name, plugin_args, response = self.parse_latest_plugin_call(response)
        if plugin_name:
            tool_observation = self.call_plugin(plugin_name, plugin_args)
            response += tool_observation
            response, his = self.model.chat(response, history=his, meta_instruction=self.system_prompt)
        return response, his