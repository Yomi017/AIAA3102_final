from typing import Any, Dict, List, Optional, Tuple, Union
import json5
from loguru import logger

from llm import BaseLLM
from tool import ToolsManager
from security.prompt_guard import has_high_risk, scan_prompt

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

Important: When you need current time or date information, use the get_cur_time tool to query it. DO NOT assume or guess the current time.

CRITICAL RULES:
1. You can ONLY use ONE tool at a time in each response
2. Each field (Thought, Action, Action Input, Final Answer) must appear EXACTLY ONCE
3. After using a tool, the system will show you the result and let you continue
4. NEVER include multiple "Action:" or "Action Input:" lines in one response
5. If you need multiple tools, use them one by one across multiple turns
6. NEVER make up information - ALL answers MUST be based on tool results
7. If you don't have information from tools, admit it and use appropriate tools to find it
8. Always cite the source of your information (which tool provided it)
9. Distinguish between casual conversation and factual queries that need verification:
   - Casual conversation (greetings, chitchat, general discussion): Can answer directly with Final Answer
   - Factual queries (specific data, current events, calculations, etc.): MUST use tools to verify
10. Even for casual conversation, you MUST follow the response format with Thought and Final Answer

Information Accuracy Requirements:
- DO NOT fabricate any facts, numbers, dates, or details
- DO NOT guess or assume information you don't have
- ONLY provide information that comes directly from tool observations
- If a tool doesn't return results, say so clearly - don't make up alternatives
- When uncertain, use tools to verify before answering

Response Format Requirements:
- For casual conversation (greetings, chitchat, opinions):
  Thought: [recognize this is casual conversation that doesn't need verification]
  Final Answer: [your response]

- When you need to use a tool for factual queries:
  Thought: [your reasoning about what information needs verification and which tool to use]
  Action: [exactly ONE tool name from: {tool_names}]
  Action Input: [the input for that ONE tool only]

- After getting tool results:
  Thought: [analyze the tool results and decide if you have enough information]
  Final Answer: [your complete response based ONLY on verified information from tools]

Remember: 
- ONE tool per response
- NEVER make up factual information - every fact must come from a tool result
- Casual conversation can be answered directly but MUST follow the format
- Factual queries MUST be verified with tools

---

请尽你所能回答以下问题。你可以使用以下工具:

{tool_description}

重要提示: 当你需要当前时间或日期信息时,使用get_cur_time工具查询。不要假设或猜测当前时间。

关键规则:
1. 每次回复中只能使用一个工具
2. 每个字段(Thought、Action、Action Input、Final Answer)必须恰好出现一次
3. 使用工具后,系统会显示结果并让你继续
4. 绝对不要在一次回复中包含多个"Action:"或"Action Input:"行
5. 如果需要多个工具,请在多个回合中逐一使用
6. 绝对不要编造信息 - 所有答案必须基于工具结果
7. 如果没有从工具获得信息,请承认并使用合适的工具去查找
8. 始终标注信息来源(哪个工具提供的)
9. 区分日常对话和需要验证的事实性查询:
   - 日常对话(问候、闲聊、观点讨论): 可以直接用Final Answer回答
   - 事实性查询(具体数据、时事、计算等): 必须使用工具验证
10. 即使是日常对话,也必须遵循回复格式,包含Thought和Final Answer

信息准确性要求:
- 不要编造任何事实、数字、日期或细节
- 不要猜测或假设你没有的信息
- 只提供直接来自工具观察结果的信息
- 如果工具没有返回结果,请明确说明 - 不要编造替代答案
- 不确定时,使用工具验证后再回答

回复格式要求:
- 对于日常对话(问候、闲聊、观点):
  Thought: [识别这是不需要验证的日常对话]
  Final Answer: [你的回复]

- 当需要使用工具查询事实时:
  Thought: [分析需要验证什么信息以及使用哪个工具]
  Action: [从以下工具中选择恰好一个: {tool_names}]
  Action Input: [仅针对该一个工具的输入]

- 获得工具结果后:
  Thought: [分析工具结果并判断是否有足够信息]
  Final Answer: [仅基于工具验证信息的完整回复]

记住: 
- 每次回复只用一个工具
- 绝不编造事实信息 - 每个事实都必须来自工具结果
- 日常对话可以直接回答但必须遵循格式
- 事实性查询必须使用工具验证

"""

class Agent:
    def __init__(
        self,
        model,
        rag_db_path=None,
        max_step=10,
        security_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.tool = ToolsManager(rag_db_path=rag_db_path)
        self.system_prompt = self.build_system_input()  
        self.model = model
        self.max_step = max_step
        self.security_config = security_config or {}
        self.prompt_guard_enabled = bool(self.security_config.get("enable_prompt_guard", False))
        self.lock_on_violation = bool(self.security_config.get("lock_on_violation", True))
        self.prompt_guard_blocked = False
        
        # 检测多模态支持
        self.supports_multimodal = getattr(model, 'supports_multimodal', False)
        
        logger.info(
            "Agent initialized | max_step={}, rag_db_path={}, prompt_guard={}, multimodal={}",
            max_step,
            rag_db_path,
            "ON" if self.prompt_guard_enabled else "OFF",
            "YES" if self.supports_multimodal else "NO",
        )
    
    def build_system_input(self):
        tool_description, tool_names = [], []

        for tool in self.tool.get_all_tools_info():
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
            logger.warning(f"Multiple actions detected. Using first only. Ignored: {ignored_list}")
        
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
        Call the specified plugin (tool) using unified interface.
        Args: 
            plugin_name (str): The name of the plugin to be called. 
            plugin_args (str): The parameter of the plugin, which is a string in JSON format. 
        Returns: str: The observation result after the tool is executed. 
        """
        plugin_args = json5.loads(plugin_args) if plugin_args else {}
        
        # 使用统一的工具调用接口
        result = self.tool.call_tool(plugin_name, **plugin_args)
        return '\nObservation:' + result
        
    def check_response(self, response: str) -> Tuple[bool, str]:
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
        

        if thought_count > 1:
            logger.warning(f"Format error: 'Thought:' appears {thought_count} times")
            return False, "Format error: 'Thought:' appears more than once"
        
        # 检查混合使用Action和Final Answer
        if action_count > 0 and final_answer_count > 0:
            logger.warning("Format error: Both 'Action:' and 'Final Answer:' found")
            return False, "Format error: Both 'Action:' and 'Final Answer:' found in the same response. Please use either Action or Final Answer, not both."

        # 如果有Action,检查出现次数和Action Input
        if action_count > 0:
            if action_count > 1:
                logger.warning(f"Format error: 'Action:' appears {action_count} times")
                return False, "Format error: 'Action:' appears more than once"
            
            if action_input_count == 0:
                logger.warning("Format error: 'Action:' found but 'Action Input:' missing")
                return False, "Format error: 'Action:' found but 'Action Input:' missing. If you don't need to use Action Input, please remove 'Action' from your response."
            elif action_input_count > 1:
                logger.warning(f"Format error: 'Action Input:' appears {action_input_count} times")
                return False, f"Format error: 'Action Input:' appears {action_input_count} times. Please check your response."

        # 检查Final Answer出现次数
        if final_answer_count > 1:
            logger.warning(f"Format error: 'Final Answer:' appears {final_answer_count} times")
            return False , "Format error: 'Final Answer:' appears more than once"
        
        return True , ""
    
    def have_final_answer(self, response: str) -> bool:
        """
        判断response中是否包含Final Answer
        """
        return "Final Answer:" in response
    
    def split_response(self, response: str) -> Tuple[str, str]:
        """
        Split the model's response by removing <think> tags and extracting their content.
        Only removes <think></think> tags, keeps Thought: and other ReAct format parts.
        
        Args:
            response (str): The model's response text (may contain <think> tags from Qwen3 thinking mode)
        
        Returns:
            Tuple[str, str]: (response with think tags removed, thinking content inside tags)
        """
        thinking_content = ''
        clean_response = response
        
        # 查找 <think> 和 </think> 标签
        think_start = response.find('<think>')
        think_end = response.find('</think>')

        if think_start == -1 and think_end != -1:
            think_start = 0  # 只有结束标签，视为从开头开始
        
        if think_start != -1 and think_end != -1 and think_start < think_end:
            # 提取思考内容
            thinking_content = response[think_start + len('<think>'):think_end].strip()
            
            # 移除思考标签和内容,保留其他部分(包括Thought:和ReAct格式)
            clean_response = response[:think_start] + response[think_end + len('</think>'):]
            clean_response = clean_response.strip()
        
        return clean_response, thinking_content

        
        
    def text(self, text: str, history: List = None, images: List[str] = None) -> Tuple[str, List]:
        # 处理 None 历史记录
        if history is None:
            history = []
        
        logger.info(f"New query received: {text}... | images: {len(images) if images else 0}")
        
        # 验证图片输入
        if images:
            if not self.supports_multimodal:
                error_msg = "Final Answer: ⚠️ 当前模型不支持多模态输入，无法处理图片。请使用支持视觉的模型（如 Qwen3VL）。"
                logger.warning("Image input rejected: model does not support multimodal")
                return error_msg, history
            
            # 验证图片路径
            import os
            invalid_images = []
            for img_path in images:
                if not os.path.exists(img_path):
                    invalid_images.append(img_path)
            
            if invalid_images:
                error_msg = f"Final Answer: ⚠️ 以下图片路径不存在:\n" + "\n".join(f"  - {p}" for p in invalid_images)
                logger.error(f"Invalid image paths: {invalid_images}")
                return error_msg, history
            
            logger.info(f"Valid images loaded: {images}")
        
        if self.prompt_guard_enabled:
            guard_response = self._apply_prompt_guard(text)
            if guard_response is not None:
                logger.info("Prompt guard response returned to user")
                return guard_response, history
        response = text

        # 'his' is the updated history
        step_count = 0
        # 只在第一步传入图片，后续轮次依赖历史
        current_images = images if step_count == 0 else None
        
        while step_count < self.max_step:
            logger.debug(f"Step {step_count + 1}/{self.max_step} started")
            new_response, history = self.model.chat(
                response, 
                history=history, 
                meta_instruction=self.system_prompt,
                images=current_images
            )
            # 后续轮次不再传图片
            current_images = None
            response = ""
            no_thinking_response ,thinking_response = self.split_response(new_response)
            
            logger.debug(f"Thinking: {thinking_response}..." if thinking_response else "Thinking: None")
            logger.debug(f"Response: {no_thinking_response}...")

            response_formate_error, error_info =  self.check_response(no_thinking_response)

            if not response_formate_error:
                
                # 弹出最后一个历史记录
                if len(history) > 0:
                    history.pop()
                
                logger.warning("Response format check failed, asking model to retry")
                error_message = "⚠️ System Error: Your previous response was not in the correct format. Please follow the specified format strictly:\n- Use exactly ONE 'Thought:', ONE 'Action:' (if needed), ONE 'Action Input:' (if Action present), or ONE 'Final Answer:'\n- Do NOT include multiple actions in one response"
                history.append({"role": "system", "content": error_info + error_message})
                continue

            plugin_name, plugin_args, warning = self.parse_latest_plugin_call(no_thinking_response)
            logger.info(f"Parsed tool call - Name: {plugin_name}, Args: {plugin_args if plugin_args else 'None'}...")
            if plugin_name:
                try:
                    logger.info(f"Calling tool: {plugin_name}")
                    tool_observation = self.call_plugin(plugin_name, plugin_args)
                    logger.success(f"Tool {plugin_name} executed successfully")
                    
                    # 使用tool角色传递工具执行结果
                    history.append({
                        "role": "tool",
                        "content": tool_observation,
                        "name": plugin_name
                    })
                    
                    # 如果有警告,追加system消息
                    if warning:
                        history.append({"role": "system", "content": warning})
                    
                    logger.debug(f"Tool observation added to history: {tool_observation}...")
                except Exception as e:
                    logger.error(f"Tool execution error: {plugin_name}, {str(e)}")
                    # 使用system角色返回工具错误
                    error_message = f"⚠️ Tool Execution Error: Failed to call tool '{plugin_name}'. Error: {e}\nPlease try another tool or correct the arguments."
                    history.append({"role": "system", "content": error_message})
            else:
                if not self.have_final_answer(no_thinking_response):
                    logger.info("No final answer detected, continuing query")

                    # 弹出最后一个历史记录
                    if len(history) > 0:
                        history.pop()
                    
                    history.append({"role": "system", "content": "⚠️ Warning: No tool call detected and no Final Answer provided. Please either use a tool to get information or provide a Final Answer."})
                else:
                    logger.info("Final answer detected, stopping query")
                    break
            step_count += 1

        logger.info(f"Query completed in {step_count} steps")
        return no_thinking_response, history

    def _apply_prompt_guard(self, user_text: str) -> Optional[str]:
        """Run heuristic guard; return final answer string if blocked."""
        if self.prompt_guard_blocked:
            logger.warning("Prompt guard is locked; rejecting new query")
            return "Final Answer: ⚠️ 当前会话因检测到提示词攻击而被锁定，请重启程序或联系管理员。"

        guard_result = scan_prompt(user_text)
        if not guard_result.findings:
            return None

        categories = ", ".join(sorted({finding.category for finding in guard_result.findings}))
        logger.warning(
            "Prompt guard triggered | categories={} | normalized_prompt={}",
            categories,
            guard_result.normalized_prompt,
        )

        if self.lock_on_violation and has_high_risk(guard_result.findings):
            self.prompt_guard_blocked = True
            logger.error("Prompt guard locked the session due to high risk input")

        user_message = (
            "Final Answer: ⚠️ 检测到潜在的危险指令 ({}), 请求已被拒绝。".format(categories)
        )
        return user_message

    def reset_security(self) -> bool:
        """Clear prompt guard lock state; return True if a lock was lifted."""
        was_blocked = self.prompt_guard_blocked
        self.prompt_guard_blocked = False
        logger.info(
            "Prompt guard reset | previously_blocked={} | guard_enabled={}",
            was_blocked,
            self.prompt_guard_enabled,
        )
        return was_blocked