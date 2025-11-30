"""
向量数据库生成工具 - 用于从文件夹生成向量数据库
"""
import argparse
import os
from document_processor import DocumentProcessor
from vector_store import VectorStore


def build_vector_db(folder_path: str, output_dir: str, chunk_size: int = 400, overlap: int = 80):
    """
    从文件夹构建向量数据库
    
    Args:
        folder_path: 包含文本文件的文件夹路径
        output_dir: 输出向量数据库的目录
        chunk_size: 每个块的字符数
        overlap: 块之间重叠的字符数
    """
    print("=" * 60)
    print("📚 向量数据库构建工具")
    print("=" * 60)
    
    # 1. 加载和处理文档
    print(f"\n[步骤 1/3] 加载文档从: {folder_path}")
    processor = DocumentProcessor(chunk_size=chunk_size, overlap=overlap)
    documents = processor.load_documents_from_folder(folder_path)
    
    if not documents:
        print("❌ 错误: 未找到任何支持的文档文件")
        print(f"   支持的格式: {processor.supported_extensions}")
        return
    
    print(f"✓ 共加载 {len(documents)} 个文档块")
    
    # 统计来源文件
    sources = set(doc['metadata']['source'] for doc in documents)
    print(f"✓ 来自 {len(sources)} 个文件: {', '.join(sources)}")
    
    # 2. 构建向量索引
    print(f"\n[步骤 2/3] 构建向量索引")
    vector_store = VectorStore()
    vector_store.build_index(documents)
    
    # 3. 保存向量数据库
    print(f"\n[步骤 3/3] 保存向量数据库")
    vector_store.save(output_dir)
    
    print("\n" + "=" * 60)
    print("✅ 向量数据库构建完成!")
    print(f"   保存位置: {output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='从文件夹构建向量数据库')
    parser.add_argument('--input', '-i', required=True, help='输入文件夹路径')
    parser.add_argument('--output', '-o', required=True, help='输出向量数据库目录')
    parser.add_argument('--chunk-size', type=int, default=400, help='分块大小(默认: 400)')
    parser.add_argument('--overlap', type=int, default=80, help='块重叠大小(默认: 80)')
    
    args = parser.parse_args()
    
    # 验证输入路径
    if not os.path.exists(args.input):
        print(f"❌ 错误: 输入路径不存在: {args.input}")
        return
    
    # 构建向量数据库
    build_vector_db(
        folder_path=args.input,
        output_dir=args.output,
        chunk_size=args.chunk_size,
        overlap=args.overlap
    )


if __name__ == '__main__':
    main()
