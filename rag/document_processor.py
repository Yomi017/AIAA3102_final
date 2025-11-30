"""
文档处理模块 - 负责文件读取、清洗和分块
"""
import os
import re
from typing import List, Dict
from pathlib import Path


class DocumentProcessor:
    """文档处理器 - 处理文本文件的读取、清洗和分块"""
    
    def __init__(self, chunk_size: int = 400, overlap: int = 80):
        """
        初始化文档处理器
        
        Args:
            chunk_size: 每个块的字符数
            overlap: 块之间重叠的字符数
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.supported_extensions = {'.txt', '.md', '.markdown'}
    
    def load_documents_from_folder(self, folder_path: str) -> List[Dict[str, str]]:
        """
        从文件夹加载所有支持的文档
        
        Args:
            folder_path: 文件夹路径
            
        Returns:
            文档列表,每个文档包含 content 和 metadata
        """
        folder = Path(folder_path)
        if not folder.exists():
            raise ValueError(f"文件夹不存在: {folder_path}")
        
        documents = []
        for file_path in folder.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                try:
                    doc = self.load_single_document(str(file_path))
                    documents.extend(doc)
                    print(f"✓ 已加载: {file_path.name}")
                except Exception as e:
                    print(f"✗ 加载失败 {file_path.name}: {e}")
        
        return documents
    
    def load_single_document(self, file_path: str) -> List[Dict[str, str]]:
        """
        加载单个文档并进行分块
        
        Args:
            file_path: 文件路径
            
        Returns:
            分块后的文档列表
        """
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 清洗文本
        cleaned_content = self._clean_text(content)
        
        # 分块
        chunks = self._chunk_text(cleaned_content)
        
        # 构建文档列表
        documents = []
        file_name = Path(file_path).name
        for i, chunk in enumerate(chunks):
            documents.append({
                'content': chunk,
                'metadata': {
                    'source': file_name,
                    'chunk_id': i,
                    'total_chunks': len(chunks)
                }
            })
        
        return documents
    
    def _clean_text(self, text: str) -> str:
        """
        清洗文本 - 去除多余空格、特殊字符等
        
        Args:
            text: 原始文本
            
        Returns:
            清洗后的文本
        """
        # 统一换行符
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # 去除多余的空白行(保留单个换行)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # 去除行首行尾空格
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        # 去除特殊控制字符
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]', '', text)
        
        return text.strip()
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        使用滑动窗口策略分块文本
        
        Args:
            text: 要分块的文本
            
        Returns:
            分块列表
        """
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # 如果不是最后一块,尝试在句子边界处切分
            if end < len(text):
                # 查找最近的句子结束符
                for sep in ['。', '!\n', '?\n', '\n\n', '!', '?', '\n', '。']:
                    pos = text.rfind(sep, start, end)
                    if pos != -1 and pos > start + self.chunk_size // 2:
                        end = pos + len(sep)
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # 计算下一个起始位置
            next_start = end - self.overlap
            
            # 避免无限循环 - 确保前进
            if next_start <= start:
                next_start = end
            
            start = next_start
        
        return chunks
