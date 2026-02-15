"""
LangChain 语义分块演示 - 使用 DeepSeek

语义分块 vs 传统分块的区别：
- 传统分块：按固定字符数分割，可能在句子中间断开
- 语义分块：基于语义相似度分割，保持语义连贯性

工作原理：
1. 将文本按句子分割
2. 计算相邻句子的语义相似度（使用 embedding）
3. 当相似度低于阈值时，创建新的分块

安装依赖:
pip install langchain langchain-community langchain-experimental
pip install langchain-huggingface sentence-transformers
pip install langchain-deepseek python-dotenv
"""

import os
from dotenv import load_dotenv

os.environ['USER_AGENT'] = 'my-rag-app/1.0'
load_dotenv()

# 1. 加载文档
from langchain_community.document_loaders import TextLoader

file_path = "../99-doc-data/黑悟空/黑悟空wiki.txt"

print(f"正在加载文档: {file_path}")
loader = TextLoader(file_path=file_path, encoding="utf-8")
docs = loader.load()

print(f"文档加载成功")
print(f"文档数量: {len(docs)}")
print(f"文档长度: {len(docs[0].page_content)} 字符")

# 2. 设置嵌入模型（用于计算语义相似度）
from langchain_huggingface import HuggingFaceEmbeddings

print("\n正在加载 Embedding 模型...")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",  # 中文模型
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 3. 创建语义分块器
from langchain_experimental.text_splitter import SemanticChunker

print("\n创建语义分块器...")
semantic_splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile",  # 使用百分位数阈值
    breakpoint_threshold_amount=90,  # 90% 不相似度时分割
)

# 4. 创建传统分块器（对比）
from langchain_text_splitters import RecursiveCharacterTextSplitter

traditional_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# 5. 执行语义分块
print("\n" + "="*60)
print("执行语义分块...")
print("="*60)

semantic_chunks = semantic_splitter.split_documents(docs)

print(f"\n=== 语义分块结果 ===")
print(f"生成的块数: {len(semantic_chunks)}")
print(f"参数设置:")
print(f"  - breakpoint_threshold_type: percentile")
print(f"  - breakpoint_threshold_amount: 90 (90%不相似度阈值)")

for i, chunk in enumerate(semantic_chunks, 1):
    print(f"\n--- 第 {i} 个语义块 ---")
    print(f"长度: {len(chunk.page_content)} 字符")
    print(f"内容: {chunk.page_content[:200]}...")  # 只显示前200字符
    print("-" * 50)
    
    if i >= 5:
        print(f"... 还有 {len(semantic_chunks) - 5} 个语义块 ...")
        break

# 6. 执行传统分块（对比）
print("\n" + "="*60)
print("执行传统分块...")
print("="*60)

traditional_chunks = traditional_splitter.split_documents(docs)

print(f"\n=== 传统分块结果 ===")
print(f"生成的块数: {len(traditional_chunks)}")
print(f"参数设置:")
print(f"  - chunk_size: 500 字符")
print(f"  - chunk_overlap: 50 字符")

for i, chunk in enumerate(traditional_chunks, 1):
    print(f"\n--- 第 {i} 个传统块 ---")
    print(f"长度: {len(chunk.page_content)} 字符")
    print(f"内容: {chunk.page_content[:200]}...")
    print("-" * 50)
    
    if i >= 5:
        print(f"... 还有 {len(traditional_chunks) - 5} 个传统块 ...")
        break

# 7. 对比分析
print("\n" + "="*80)
print("分块方法对比分析")
print("="*80)

semantic_avg_length = sum(len(chunk.page_content) for chunk in semantic_chunks) / len(semantic_chunks)
traditional_avg_length = sum(len(chunk.page_content) for chunk in traditional_chunks) / len(traditional_chunks)

print(f"\n语义分块:")
print(f"  - 块数量: {len(semantic_chunks)}")
print(f"  - 平均长度: {semantic_avg_length:.1f} 字符")
print(f"  - 优点: 保持语义连贯性，块边界更自然")
print(f"  - 缺点: 需要 embedding 计算，速度较慢")

print(f"\n传统分块:")
print(f"  - 块数量: {len(traditional_chunks)}")
print(f"  - 平均长度: {traditional_avg_length:.1f} 字符")
print(f"  - 优点: 快速，块大小可控")
print(f"  - 缺点: 可能在句子中间断开")

# 8. 使用语义分块构建 RAG 系统
print("\n" + "="*80)
print("使用语义分块构建 RAG 系统")
print("="*80)

from langchain_core.vectorstores import InMemoryVectorStore

print("\n正在构建向量索引...")
vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(semantic_chunks)

# 9. 测试问答
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek

questions = [
    "黑神话悟空的主角是谁？",
    "游戏的战斗系统有什么特点？",
    "游戏的故事背景是什么？"
]

prompt = ChatPromptTemplate.from_template("""
基于以下上下文回答问题。如果没有结果，就说没有找到对应信息。

上下文: {context}

问题: {question}

回答:
""")

llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0.7,
    max_tokens=2048,
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

print("\n开始问答测试...")
for question in questions:
    print(f"\n问题: {question}")
    
    # 检索相关文档
    retrieved_docs = vector_store.similarity_search(question, k=2)
    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
    
    # 生成答案
    answer = llm.invoke(prompt.format(question=question, context=docs_content))
    print(f"回答: {answer.content}")
    print("-" * 50)

# 10. 使用建议
print("\n" + "="*80)
print("使用建议")
print("="*80)
print("""
1. 语义分块适用场景：
   - 问答系统：保持上下文完整性
   - 长文档处理：自动识别主题边界
   - 高质量检索：语义相关性更强

2. 参数调优：
   - breakpoint_threshold_amount 越小 → 块越多（更细粒度）
   - breakpoint_threshold_amount 越大 → 块越少（更粗粒度）
   - 推荐值：85-95 之间

3. 性能考虑：
   - 首次运行会下载 embedding 模型（约 100MB）
   - 语义分块比传统分块慢 5-10 倍
   - 适合离线处理，不适合实时场景

4. 其他分块类型：
   - breakpoint_threshold_type="percentile"：百分位数（推荐）
   - breakpoint_threshold_type="standard_deviation"：标准差
   - breakpoint_threshold_type="interquartile"：四分位数
""")
