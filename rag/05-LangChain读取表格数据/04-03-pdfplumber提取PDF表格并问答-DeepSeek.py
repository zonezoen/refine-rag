import os
from dotenv import load_dotenv
import pdfplumber
import pandas as pd

os.environ['USER_AGENT'] = 'my-rag-app/1.0'
load_dotenv()

# 1. 使用 pdfplumber 提取 PDF 表格
pdf_path = "../99-doc-data/复杂PDF/billionaires_page-1-5.pdf"

print("正在提取 PDF 表格...")
with pdfplumber.open(pdf_path) as pdf:
    tables = []
    for page in pdf.pages:
        table = page.extract_table()
        if table:
            tables.append(table)

# 2. 将表格转换为文本文档
from langchain_core.documents import Document

documents = []
if tables:
    for i, table in enumerate(tables, 1):
        # 转换为 DataFrame
        df = pd.DataFrame(table)
        
        # 转换为文本（更易读的格式）
        text = df.to_string(index=False)
        
        # 创建 Document 对象
        doc = Document(
            page_content=text,
            metadata={"source": f"表格{i}", "page": i}
        )
        documents.append(doc)

print(f"成功提取 {len(documents)} 个表格")

# 3. 文档分块（表格通常不需要分块，但如果表格很大可以分块）
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,  # 表格可能较大，增加 chunk_size
    chunk_overlap=200
)
all_splits = text_splitter.split_documents(documents)

print(f"分块后共 {len(all_splits)} 个文档片段")

# 4. 设置嵌入模型（使用本地 HuggingFace 模型）
from langchain_huggingface import HuggingFaceEmbeddings

print("正在加载 Embedding 模型...")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",  # 中文模型
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 5. 存储到向量数据库
from langchain_core.vectorstores import InMemoryVectorStore

print("正在构建向量索引...")
vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(all_splits)

# 6. 定义问题
questions = [
    "2023年谁是最富有的人?",
    "最年轻的富豪是谁?",
    "有哪些科技行业的富豪?"
]

# 7. 构建提示模板
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""
基于以下表格数据回答问题。如果表格中没有相关信息，就说没有找到对应信息。

表格数据:
{context}

问题: {question}

回答:
""")

# 8. 配置 DeepSeek LLM
from langchain_deepseek import ChatDeepSeek

llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0.7,
    max_tokens=2048,
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

# 9. 问答循环
print("\n" + "="*50)
print("开始问答")
print("="*50)

for question in questions:
    print(f"\n问题: {question}")
    
    # 检索相关文档
    retrieved_docs = vector_store.similarity_search(question, k=2)
    
    # 合并文档内容
    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
    
    # 生成答案
    answer = llm.invoke(prompt.format(question=question, context=docs_content))
    
    print(f"回答: {answer.content}")
    print("-"*50)
