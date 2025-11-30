"""
RAG查询引擎 - 负责检索和生成答案
"""
from typing import List, Dict, Optional
from .vector_store import VectorStore


class RAGEngine:
    """RAG查询引擎 - 结合检索和生成"""
    
    def __init__(self, vector_db_path: str):
        """
        初始化RAG引擎
        
        Args:
            vector_db_path: 向量数据库路径
        """
        self.vector_store = VectorStore()
        self.vector_store.load(vector_db_path)
        print("✓ RAG引擎初始化完成")
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        搜索相关文档
        
        Args:
            query: 用户查询
            top_k: 返回前k个结果
            
        Returns:
            搜索结果列表
        """
        results = self.vector_store.search(query, top_k=top_k)
        return results
    
    def format_context(self, search_results: List[Dict]) -> str:
        """
        格式化检索到的上下文
        
        Args:
            search_results: 搜索结果列表
            
        Returns:
            格式化的上下文字符串
        """
        if not search_results:
            return "未找到相关信息"
        
        context_parts = []
        for i, result in enumerate(search_results, 1):
            content = result['content']
            metadata = result['metadata']
            score = result['score']
            
            source = metadata.get('source', '未知')
            chunk_id = metadata.get('chunk_id', 0)
            
            context_parts.append(
                f"[参考资料{i}] (来源: {source}, 相关度: {score:.3f})\n{content}"
            )
        
        return "\n\n".join(context_parts)
    
    def query(self, question: str, top_k: int = 3) -> Dict[str, any]:
        """
        执行RAG查询
        
        Args:
            question: 用户问题
            top_k: 检索文档数量
            
        Returns:
            包含上下文和元信息的字典
        """
        # 检索相关文档
        search_results = self.search(question, top_k=top_k)
        
        # 格式化上下文
        context = self.format_context(search_results)
        
        return {
            'question': question,
            'context': context,
            'search_results': search_results
        }
