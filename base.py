"""
Baseline Agent - 消融实验用
直接调用 LLM，不使用 Agent 框架、ReAct、工具等
保持与 Agent 相同的接口以便于对比测试
"""

from typing import List, Tuple, Optional
from loguru import logger


class BaselineAgent:
    """
    消融实验基线 - 无 Agent 框架
    
    特点：
    - 直接调用 LLM
    - 无 ReAct 推理框架
    - 无工具调用
    - 无多轮优化
    - 保持与 Agent 相同的接口（.text() 方法）
    """
    
    def __init__(self, model, rag_db_path=None, max_step=10, security_config=None):
        """
        初始化 Baseline Agent
        
        Args:
            model: LLM 模型实例（与 Agent 使用相同的模型）
            rag_db_path: 忽略（Baseline 不使用 RAG）
            max_step: 忽略（Baseline 不需要多轮推理）
            security_config: 忽略（Baseline 不使用安全检查）
        """
        self.model = model
        self.supports_multimodal = getattr(model, 'supports_multimodal', False)
        
        # 简单的系统提示（无 ReAct 框架）
        self.system_prompt = """你是一个 AI 助手。请直接回答问题或执行指令。"""
        
        logger.info(
            "BaselineAgent initialized (ablation mode) | multimodal={}",
            "YES" if self.supports_multimodal else "NO"
        )
    
    def text(self, text: str, history: List = None, images: List[str] = None) -> Tuple[str, List]:
        """
        处理用户输入（Baseline 模式）
        
        Args:
            text: 用户输入文本
            history: 对话历史
            images: 图片列表（如果支持多模态）
            
        Returns:
            (LLM 响应, 更新后的历史)
        """
        if history is None:
            history = []
        
        logger.info(f"Baseline query: {text[:50]}... | images: {len(images) if images else 0}")
        
        # 验证图片输入
        if images and not self.supports_multimodal:
            error_msg = "⚠️ 当前模型不支持多模态输入"
            logger.warning("Image input rejected: model does not support multimodal")
            return error_msg, history
        
        try:
            # 直接调用 LLM（无 Agent 框架）
            response, new_history = self.model.chat(
                text,
                history=history,
                meta_instruction=self.system_prompt,
                images=images
            )
            
            logger.info(f"Baseline response generated: {len(response)} chars")
            return response, new_history
            
        except Exception as e:
            logger.error(f"Baseline error: {e}")
            error_msg = f"Error: {str(e)}"
            return error_msg, history
