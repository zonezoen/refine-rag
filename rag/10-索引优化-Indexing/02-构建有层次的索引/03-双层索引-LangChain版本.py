# 双层检索-富豪榜 - LangChain版本 - 需要pip install openpyxl
import os
from dotenv import load_dotenv
import pandas as pd
import logging
from langchain_core.documents import Document
from langchain_community.vectorstores import Milvus
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.agents import create_pandas_dataframe_agent

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 加载环境变量
load_dotenv()

# 初始化模型
llm = ChatDeepSeek(model="deepseek-chat", temperature=0)
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh")

# Milvus 连接配置
MILVUS_URI = "http://localhost:19530"

# 加载Excel文件并准备数据
excel_file = "../../99-doc-data/复杂PDF/十大富豪/世界十大富豪.xlsx"

# 存储所有表格的DataFrame和Agent
table_info = {}  # {sheet_name: {"df": df, "agent": agent}}
summary_documents = []

# 读取Excel文件中的所有sheet
with pd.ExcelFile(excel_file) as xls:
    for sheet_name in xls.sheet_names:
        try:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            logging.info(f"正在处理sheet: {sheet_name}")
            
            # 创建该表格的Pandas Agent
            # 关键：使用 agent_executor_kwargs 传递参数
            agent = create_pandas_dataframe_agent(
                llm,
                df,
                verbose=False,
                allow_dangerous_code=True,
                agent_executor_kwargs={
                    "handle_parsing_errors": True,  # 处理解析错误
                    "max_iterations": 10,  # 增加最大迭代次数
                    "max_execution_time": 60,  # 最大执行时间（秒）
                }
            )
            
            # 存储DataFrame和Agent
            table_info[sheet_name] = {
                "df": df,
                "agent": agent
            }
            
            # 创建摘要文档（用于第一层检索）
            summary = f"这是{sheet_name}的世界十大富豪排行榜数据，包含排名、姓名、财富、行业等信息。"
            summary_doc = Document(
                page_content=summary,
                metadata={"sheet_name": sheet_name}
            )
            summary_documents.append(summary_doc)
            
            logging.info(f"成功处理sheet: {sheet_name}")
            
        except Exception as e:
            logging.error(f"处理sheet {sheet_name} 时出错: {str(e)}")
            logging.error(f"错误详情: {e.__class__.__name__}")
            continue

# 创建第一层索引：摘要向量存储
summary_vectorstore = Milvus.from_documents(
    summary_documents, 
    embeddings,
    collection_name="billionaires_summary_langchain",
    connection_args={"uri": MILVUS_URI},
    drop_old=True  # 删除旧集合
)
summary_retriever = summary_vectorstore.as_retriever(search_kwargs={"k": 1})

def generate_answer(question):
    """
    两层检索流程：
    1. 第一层：在摘要中检索，找到最相关的sheet
    2. 第二层：使用该sheet的Pandas Agent回答问题
    """
    try:
        # 第一层检索：找到相关的sheet
        relevant_docs = summary_retriever.invoke(question)
        
        if not relevant_docs:
            return "抱歉，没有找到相关信息。"
        
        # 获取最相关的sheet名称
        sheet_name = relevant_docs[0].metadata["sheet_name"]
        logging.info(f"第一层检索结果：匹配到 {sheet_name}")
        
        # 第二层检索：使用该sheet的Pandas Agent
        agent = table_info[sheet_name]["agent"]
        
        # 构建简洁的提示
        detailed_question = f"基于数据回答：{question}"
        
        # 使用 invoke 方法
        result = agent.invoke(detailed_question)
        answer = result['output']
        
        return f"数据来源：{sheet_name}\n\n{answer}"
        
    except Exception as e:
        logging.error(f"生成答案时出错: {str(e)}")
        return f"抱歉，处理问题时出错: {str(e)}"

def generate_answer_with_details(question):
    """
    带详细信息的答案生成（显示检索过程）
    """
    try:
        # 第一层检索
        relevant_docs = summary_retriever.invoke(question)
        
        if not relevant_docs:
            return {
                "answer": "抱歉，没有找到相关信息。",
                "matched_sheet": None,
                "summary": None,
                "dataframe": None
            }
        
        # 获取匹配信息
        sheet_name = relevant_docs[0].metadata["sheet_name"]
        summary = relevant_docs[0].page_content
        
        # 第二层检索
        agent = table_info[sheet_name]["agent"]
        detailed_question = f"基于数据回答：{question}"
        result = agent.invoke(detailed_question)
        answer = result['output']
        
        return {
            "answer": answer,
            "matched_sheet": sheet_name,
            "summary": summary,
            "dataframe": table_info[sheet_name]["df"]
        }
        
    except Exception as e:
        logging.error(f"生成答案时出错: {str(e)}")
        return {
            "answer": f"抱歉，处理问题时出错: {str(e)}",
            "matched_sheet": None,
            "summary": None,
            "dataframe": None
        }

# 测试示例
if __name__ == "__main__":
    print("=" * 60)
    print("双层检索系统 - LangChain版本（使用Pandas Agent）")
    print("=" * 60)
    
    test_questions = [
        "2020年世界首富是谁？他的财富是多少？",
        "2023年排名前三的富豪分别是谁？",
        "哪一年的首富财富最多？"
    ]
    
    for question in test_questions:
        print(f"\n问题：{question}")
        print("-" * 60)
        
        # 简单版本
        answer = generate_answer(question)
        print(f"答案：\n{answer}")
        print("=" * 60)
    
    # 详细版本示例
    print("\n\n" + "=" * 60)
    print("详细检索过程示例")
    print("=" * 60)
    
    question = "2020年世界首富是谁？"
    result = generate_answer_with_details(question)
    
    print(f"\n问题：{question}")
    print(f"\n第一层检索结果：")
    print(f"  匹配的表格：{result['matched_sheet']}")
    print(f"  表格摘要：{result['summary']}")
    print(f"\n第二层检索结果：")
    print(f"  答案：{result['answer']}")
    
    if result['dataframe'] is not None:
        print(f"\n数据预览：")
        print(result['dataframe'].head())
