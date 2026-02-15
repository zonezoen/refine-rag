"""
BGE-M3 多向量嵌入模型演示

BGE-M3 是什么？
- M3 = Multi-Functionality（多功能）、Multi-Linguality（多语言）、Multi-Granularity（多粒度）
- 由北京智源人工智能研究院（BAAI）开发
- 支持 100+ 种语言
- 同时支持三种嵌入方式：密集嵌入、稀疏嵌入、多向量嵌入

三种嵌入方式的区别：
1. 密集嵌入（Dense）：传统的向量表示，适合语义搜索
2. 稀疏嵌入（Sparse）：类似 BM25，适合关键词匹配
3. 多向量嵌入（ColBERT）：每个 token 一个向量，适合精确匹配

安装依赖：
pip install FlagEmbedding
"""

from FlagEmbedding import BGEM3FlagModel

def main():
    # 1. 加载 BGE-M3 模型
    # use_fp16=False: 使用 float32 精度（更准确但更慢，适合 CPU）
    # use_fp16=True: 使用 float16 精度（更快但稍微不准确，适合 GPU）
    print("正在加载 BGE-M3 模型...")
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)
    print("模型加载完成！\n")
    
    # 2. 准备测试文本
    # 注意：passage 必须是列表格式，即使只有一个文本
    passage = ["猢狲施展烈焰拳，击退妖怪；随后开启金刚体，抵挡神兵攻击。"]
    
    print(f"原始文本: {passage[0]}\n")
    print("="*60)

    # 3. 编码文本，同时获取三种嵌入
    print("正在生成三种嵌入...\n")
    passage_embeddings = model.encode(
        passage,
        return_sparse=True,      # 返回稀疏嵌入（类似 BM25 的词频权重）
        return_dense=True,       # 返回密集嵌入（传统的语义向量）
        return_colbert_vecs=True # 返回多向量嵌入（每个 token 一个向量）
    )
    
    # 4. 提取三种嵌入结果
    # dense_vecs: 密集向量，形状为 (1024,)，一个文本对应一个向量
    dense_vecs = passage_embeddings["dense_vecs"]
    
    # sparse_vecs: 稀疏向量，字典格式 {token_id: weight}
    # 只存储非零权重的 token，类似 BM25 的词频统计
    sparse_vecs = passage_embeddings["lexical_weights"]
    
    # colbert_vecs: 多向量嵌入，形状为 (token_count, 1024)
    # 每个 token 都有一个独立的向量
    colbert_vecs = passage_embeddings["colbert_vecs"]
    
    # 5. 展示密集嵌入（Dense Embedding）
    print("="*60)
    print("【1. 密集嵌入 (Dense Embedding)】")
    print("="*60)
    print(f"维度: {dense_vecs[0].shape}")
    print(f"说明: 整个文本被压缩成一个 {dense_vecs[0].shape[0]} 维的向量")
    print(f"用途: 语义搜索、相似度计算")
    print(f"前10维示例: {dense_vecs[0][:10]}")
    print()
    
    # 6. 展示稀疏嵌入（Sparse Embedding）
    print("="*60)
    print("【2. 稀疏嵌入 (Sparse Embedding)】")
    print("="*60)
    print(f"非零元素数量: {len(sparse_vecs[0])}")
    print(f"说明: 只存储重要的 token 及其权重（类似 BM25）")
    print(f"用途: 关键词匹配、精确检索")
    print(f"前10个非零值示例:")
    for token_id, weight in list(sparse_vecs[0].items())[:10]:
        print(f"  Token ID {token_id}: 权重 {weight:.4f}")
    print()
    
    # 7. 展示多向量嵌入（ColBERT-style Multi-Vector）
    print("="*60)
    print("【3. 多向量嵌入 (ColBERT Multi-Vector)】")
    print("="*60)
    print(f"维度: {colbert_vecs[0].shape}")
    print(f"说明: 文本被分成 {colbert_vecs[0].shape[0]} 个 token，每个 token 有一个 {colbert_vecs[0].shape[1]} 维向量")
    print(f"用途: 精确匹配、细粒度检索")
    print(f"前2个 token 的向量示例:")
    for i, vec in enumerate(colbert_vecs[0][:2]):
        print(f"  Token {i}: {vec[:10]}... (仅显示前10维)")
    print()
    
    # 8. 三种嵌入的对比总结
    print("="*60)
    print("【三种嵌入方式对比】")
    print("="*60)
    print("""
    | 嵌入类型 | 维度 | 特点 | 适用场景 |
    |---------|------|------|---------|
    | 密集嵌入 | (1024,) | 整个文本一个向量 | 语义搜索、相似度计算 |
    | 稀疏嵌入 | 字典 | 只存储重要 token | 关键词匹配、BM25 风格 |
    | 多向量嵌入 | (tokens, 1024) | 每个 token 一个向量 | 精确匹配、细粒度检索 |
    
    推荐使用场景：
    - 问答系统：密集嵌入（理解语义）
    - 关键词搜索：稀疏嵌入（精确匹配）
    - 混合检索：密集 + 稀疏（结合两者优势）
    - 高精度检索：多向量嵌入（最精确但最慢）
    """)

if __name__ == '__main__':
    main()
