import pdfplumber
import pandas as pd
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core import Document
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from typing import List
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

# 配置 DeepSeek LLM（使用 OpenAI 兼容接口）
Settings.llm = OpenAILike(
    model="deepseek-chat",
    api_base="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0.7,
    max_tokens=2048,
)

# 配置本地 Embedding 模型（避免使用 OpenAI）
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-zh-v1.5"  # 中文 embedding 模型
)

pdf_path = "../99-doc-data/复杂PDF/billionaires_page-1-5.pdf"

# 打开 PDF 并解析表格
with pdfplumber.open(pdf_path) as pdf:
    tables = []
    for page in pdf.pages:
        table = page.extract_table()
        if table:
            tables.append(table)

# 转换所有表格为 DataFrame 并构建文档
documents: List[Document] = []
if tables:
    # 遍历所有表格
    for i, table in enumerate(tables, 1):
        # 将表格转换为 DataFrame
        df = pd.DataFrame(table)
        
        # 保存到CSV文件
        # csv_filename = f"billionaires_table_{i}.csv"
        # df.to_csv(csv_filename, index=False)
        # print(f"\n表格 {i} 数据已保存到 {csv_filename}")
        
        # 将DataFrame转换为文本
        text = df.to_string()
        
        # 创建Document对象
        doc = Document(text=text, metadata={"source": f"表格{i}"})
        documents.append(doc)

# 构建索引
index = VectorStoreIndex.from_documents(documents)

# 创建查询引擎
query_engine = index.as_query_engine()

# 示例问答
questions = [
    "2023年谁是最富有的人?",
    "最年轻的富豪是谁?"
]

print("\n===== 问答演示 =====")
for question in questions:
    response = query_engine.query(question)
    print(f"\n问题: {question}")
    print(f"回答: {response}")