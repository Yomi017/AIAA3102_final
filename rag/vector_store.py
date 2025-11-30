"""
向量存储模块 - 使用FAISS构建和管理向量数据库
"""
import os
import pickle
from typing import List, Dict, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class VectorStore:
    """向量数据库 - 使用FAISS进行向量检索"""
    
    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        """
        初始化向量存储
        
        Args:
            model_name: 嵌入模型名称
        """
        print(f"正在加载嵌入模型: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        
        # FAISS索引
        self.index = None
        
        # 存储文档内容和元数据
        self.documents = []
        self.metadatas = []
        
        print(f"✓ 嵌入模型加载完成,向量维度: {self.dimension}")
    
    def create_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        为文本列表创建嵌入向量
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            
        Returns:
            嵌入向量数组
        """
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True  # 归一化向量
        )
        return embeddings
    
    def build_index(self, documents: List[Dict[str, str]]):
        """
        从文档列表构建FAISS索引
        
        Args:
            documents: 文档列表,每个文档包含 content 和 metadata
        """
        if not documents:
            raise ValueError("文档列表为空")
        
        print(f"\n开始构建向量索引,共 {len(documents)} 个文档块...")
        
        # 提取文本内容
        texts = [doc['content'] for doc in documents]
        self.documents = texts
        self.metadatas = [doc['metadata'] for doc in documents]
        
        # 创建嵌入
        print("正在生成嵌入向量...")
        embeddings = self.create_embeddings(texts)
        
        # 构建FAISS索引 - 使用Inner Product (点积)搜索
        # 因为向量已归一化,点积等价于余弦相似度
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        
        print(f"✓ 向量索引构建完成,共 {self.index.ntotal} 个向量")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        搜索最相关的文档
        
        Args:
            query: 查询文本
            top_k: 返回前k个结果
            
        Returns:
            搜索结果列表,每个结果包含 content, metadata 和 score
        """
        if self.index is None:
            raise ValueError("向量索引未构建,请先调用 build_index()")
        
        # 为查询创建嵌入
        query_embedding = self.create_embeddings([query])
        
        # 搜索
        scores, indices = self.index.search(query_embedding, top_k)
        
        # 构建结果
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):  # 确保索引有效
                results.append({
                    'content': self.documents[idx],
                    'metadata': self.metadatas[idx],
                    'score': float(score)
                })
        
        return results
    
    def save(self, save_dir: str):
        """
        保存向量数据库到磁盘
        
        Args:
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存FAISS索引
        index_path = os.path.join(save_dir, 'faiss.index')
        faiss.write_index(self.index, index_path)
        
        # 保存文档和元数据
        data_path = os.path.join(save_dir, 'documents.pkl')
        with open(data_path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadatas': self.metadatas
            }, f)
        
        print(f"✓ 向量数据库已保存到: {save_dir}")
    
    def load(self, load_dir: str):
        """
        从磁盘加载向量数据库
        
        Args:
            load_dir: 加载目录
        """
        # 加载FAISS索引
        index_path = os.path.join(load_dir, 'faiss.index')
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"索引文件不存在: {index_path}")
        
        self.index = faiss.read_index(index_path)
        
        # 加载文档和元数据
        data_path = os.path.join(load_dir, 'documents.pkl')
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.metadatas = data['metadatas']
        
        print(f"✓ 向量数据库已加载,共 {len(self.documents)} 个文档块")
