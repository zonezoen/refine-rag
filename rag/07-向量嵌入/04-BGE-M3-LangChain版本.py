"""
BGE-M3 使用 LangChain 集成版本

如果 FlagEmbedding 安装有问题，可以使用这个版本。
使用 LangChain 的 HuggingFace 集成来加载 BGE-M3 模型。

注意：这个版本只支持密集嵌入，不支持稀疏和多向量嵌入。

安装依赖：
pip install langchain-huggingface sentence-transformers
"""

from langchain_huggingface import HuggingFaceEmbeddings

def main():
    print("正在加载 BGE-M3 模型（LangChain 版本）...")
    print("注意：此版本只支持密集嵌入\n")
    
    # 使用 LangChain 加载 BGE-M3
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    print("模型加载完成！\n")
    
    # 测试文本
    texts = [
        "猢狲施展烈焰拳，击退妖怪；随后开启金刚体，抵挡神兵攻击。",
        "孙悟空使用金箍棒战斗。",
        "妖怪被烈焰拳击败。"
    ]
    
    print("原始文本:")
    for i, text in enumerate(texts, 1):
        print(f"{i}. {text}")
    
    print("\n" + "="*60)
    print("生成密集嵌入...")
    print("="*60)
    
    # 生成嵌入
    doc_embeddings = embeddings.embed_documents(texts)
    
    print(f"\n嵌入维度: {len(doc_embeddings[0])}")
    print(f"文档数量: {len(doc_embeddings)}")
    print(f"\n第一个文档的前10维:")
    print(doc_embeddings[0][:10])
    
    # 测试查询
    print("\n" + "="*60)
    print("测试语义搜索")
    print("="*60)
    
    query = "孙悟空的技能"
    print(f"\n查询: {query}")
    
    query_embedding = embeddings.embed_query(query)
    
    # 计算相似度
    import numpy as np
    similarities = np.dot(doc_embeddings, query_embedding)
    
    print("\n相似度排名:")
    for i, sim in sorted(enumerate(similarities), key=lambda x: x[1], reverse=True):
        print(f"{i+1}. 文档 {i+1}: {sim:.4f} - {texts[i]}")
    
    print("\n" + "="*60)
    print("说明")
    print("="*60)
    print("""
    LangChain 版本的限制：
    - ✅ 支持密集嵌入（Dense Embedding）
    - ❌ 不支持稀疏嵌入（Sparse Embedding）
    - ❌ 不支持多向量嵌入（ColBERT）
    
    如果需要完整功能，请修复 FlagEmbedding 安装：
    pip install --upgrade transformers FlagEmbedding
    
    对于大多数应用，密集嵌入已经足够。
    """)

if __name__ == '__main__':
    main()
