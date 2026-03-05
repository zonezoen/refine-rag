"""
BM25 检索演示 - 修复版

BM25 是一种基于词频的检索算法，不需要 embedding 模型。
适合关键词精确匹配的场景。

与向量检索的区别：
- BM25: 基于词频统计（TF-IDF 改进版）
- 向量检索: 基于语义相似度

安装依赖:
pip install langchain langchain-community
pip install rank-bm25  # BM25 算法实现
"""

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

# 1. 准备测试数据
docs = [
    Document(page_content="猢狲在无回谷遭遇了妖怪，妖怪开始攻击，猢狲使用铜云棒抵挡。"),
    Document(page_content="妖怪使用寒冰箭攻击猢狲但被烈焰拳反击击溃。"),
    Document(page_content="猢狲施展烈焰拳击退妖怪随后开启金刚体抵挡神兵攻击。"),
    Document(page_content="猢狲召唤烈焰拳与毁灭咆哮击败妖怪随后收集妖怪精华。"),
    Document(page_content="在战斗中猢狲使用了多种技能包括烈焰拳金刚体和铜云棒。"),
]

print("文档数量:", len(docs))
print("\n文档内容:")
for i, doc in enumerate(docs, 1):
    print(f"{i}. {doc.page_content}")

# 2. 创建 BM25 检索器
print("\n" + "="*60)
print("创建 BM25 检索器...")
print("="*60)

bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 3  # 返回前3个最相关的文档

# 3. 测试检索
queries = [
    "烈焰拳",
    "妖怪攻击",
    "铜云棒",
    "战斗技能"
]

print("\n" + "="*60)
print("BM25 检索测试")
print("="*60)

for query in queries:
    print(f"\n查询: {query}")
    results = bm25_retriever.invoke(query)  # 使用 invoke 方法
    
    print(f"检索到 {len(results)} 个相关文档:")
    for i, doc in enumerate(results, 1):
        print(f"  {i}. {doc.page_content}")
    print("-" * 50)

# 4. 对比：使用向量检索（可选）
print("\n" + "="*60)
print("对比：向量检索 vs BM25")
print("="*60)

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.vectorstores import InMemoryVectorStore
    
    print("\n正在加载 Embedding 模型...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # 创建向量数据库
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(docs)
    
    # 对比检索结果
    query = "烈焰拳"
    
    print(f"\n查询: {query}")
    
    # BM25 检索
    bm25_results = bm25_retriever.invoke(query)  # 使用 invoke 方法
    print("\nBM25 检索结果:")
    for i, doc in enumerate(bm25_results, 1):
        print(f"  {i}. {doc.page_content}")
    
    # 向量检索
    vector_results = vector_store.similarity_search(query, k=3)
    print("\n向量检索结果:")
    for i, doc in enumerate(vector_results, 1):
        print(f"  {i}. {doc.page_content}")
    
    print("\n" + "="*60)
    print("对比分析")
    print("="*60)
    print("""
BM25 检索特点：
- 基于关键词匹配
- 精确匹配效果好
- 不需要 embedding 模型
- 速度快
- 适合：关键词搜索、精确匹配

向量检索特点：
- 基于语义相似度
- 能理解同义词和相关概念
- 需要 embedding 模型
- 速度较慢
- 适合：语义搜索、问答系统
    """)
    
except ImportError:
    print("\n未安装向量检索依赖，跳过对比")
    print("安装命令: pip install langchain-huggingface sentence-transformers")

# 5. BM25 参数说明
print("\n" + "="*60)
print("BM25 参数说明")
print("="*60)
print("""
k: 返回的文档数量（默认 4）
  bm25_retriever.k = 3  # 返回前3个

BM25 算法参数（高级）:
- k1: 词频饱和度参数（默认 1.5）
  - 值越大，词频影响越大
  - 推荐范围: 1.2 - 2.0

- b: 文档长度归一化参数（默认 0.75）
  - 0: 不考虑文档长度
  - 1: 完全归一化
  - 推荐值: 0.75

使用场景：
1. 关键词搜索：用户输入精确关键词
2. 代码搜索：搜索函数名、变量名
3. 专业术语搜索：医学、法律等领域
4. 混合检索：BM25 + 向量检索结合
""")
