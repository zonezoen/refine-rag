# 导入必要的库
import os  # 用于访问环境变量
from dotenv import load_dotenv  # 用于加载.env文件中的环境变量
from langchain_openai import ChatOpenAI  # OpenAI兼容的聊天模型（这里用于DeepSeek）
from langchain_community.embeddings import HuggingFaceEmbeddings  # 开源的文本嵌入模型
from langchain_core.documents import Document  # LangChain的文档对象
from langchain_core.prompts import ChatPromptTemplate  # 聊天提示词模板
from langchain_community.vectorstores import Chroma  # 向量数据库，需要安装: pip install chromadb
from langchain_community.retrievers import BM25Retriever  # BM25检索器，需要安装: pip install rank_bm25

# 准备测试数据：一些关于猢狲（孙悟空）战斗的文本
battle_logs = [
    "猢狲身披锁子甲。",  # 装备信息
    "猢狲在无回谷遭遇了妖怪，妖怪开始攻击，猢狲使用铜云棒抵挡。",  # 武器和战斗场景
    "猢狲施展烈焰拳击退妖怪随后开启金刚体抵挡神兵攻击。",  # 技能信息
    "妖怪使用寒冰箭攻击猢狲但被烈焰拳反击击溃。",  # 技能对战
    "猢狲召唤烈焰拳与毁灭咆哮击败妖怪随后收集妖怪精华。"  # 更多技能
]

# 用户的查询问题
request = "猢狲有什么装备和招数？"

# === 第一步：BM25检索（基于关键词匹配） ===
# BM25是一种基于词频的检索算法，擅长精确匹配关键词
bm25_retriever = BM25Retriever.from_texts(battle_logs)  # 从文本列表创建BM25检索器
bm25_response = bm25_retriever.invoke(request)  # 执行检索，返回最相关的文档
print(f"BM25检索结果：\n{bm25_response}")

# === 第二步：向量检索（基于语义相似度） ===
# 将文本转换为LangChain的Document对象
docs = [Document(page_content=log) for log in battle_logs]

# 加载.env文件中的环境变量
load_dotenv()

# 初始化开源的文本嵌入模型
# 这个模型支持多语言（包括中文），可以将文本转换为向量
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # 多语言模型
    model_kwargs={'device': 'cpu'}  # 使用CPU运行，如果有GPU可以改为'cuda'
)

# 创建Chroma向量数据库
# Chroma会自动将文档转换为向量并存储
chroma_vs = Chroma.from_documents(
    docs,  # 要存储的文档
    embedding=embeddings  # 使用的嵌入模型
)

# 创建向量检索器
chroma_retriever = chroma_vs.as_retriever()
# 执行语义检索，找到语义上最相似的文档
chroma_response = chroma_retriever.invoke(request)
print(f"Chroma检索结果：\n{chroma_response}")

# === 第三步：混合检索（结合BM25和向量检索的结果） ===
# 将两种检索方法的结果合并，去除重复内容
# 使用set去重，然后转换为list
hybrid_response = list({doc.page_content for doc in bm25_response + chroma_response}) 
print(f"混合检索结果：\n{hybrid_response}")

# === 第四步：使用大语言模型生成最终答案 ===

# 创建提示词模板
# 这个模板告诉AI如何基于检索到的上下文来回答问题
prompt = ChatPromptTemplate.from_template("""
                基于以下上下文，回答问题。如果上下文中没有相关信息，
                请说"我无法从提供的上下文中找到相关信息"。
                上下文: {context}
                问题: {question}
                回答:"""
                                          )

# 初始化DeepSeek聊天模型
# 使用ChatOpenAI是因为DeepSeek API兼容OpenAI的接口格式
llm = ChatOpenAI(
    model="deepseek-chat",  # DeepSeek的聊天模型
    api_key=os.getenv("DEEPSEEK_API_KEY"),  # 从环境变量获取API密钥（更安全）
    base_url="https://api.deepseek.com"  # DeepSeek的API地址
)

# 将检索到的文档内容合并成一个字符串
# 用双换行符分隔不同的文档，让AI更容易理解
doc_content = "\n\n".join(hybrid_response)

# 使用LLM生成最终答案
# 将问题和上下文填入提示词模板，然后发送给AI
answer = llm.invoke(prompt.format(question=request, context=doc_content))
print(f"LLM回答：\n{answer.content}")

# === 总结 ===
# 这个程序展示了混合检索的完整流程：
# 1. BM25检索：基于关键词匹配，擅长精确匹配
# 2. 向量检索：基于语义相似度，擅长理解语义
# 3. 混合检索：结合两种方法的优势
# 4. LLM生成：基于检索结果生成自然语言答案