# 粗中有细的示例 - LangChain版本
import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import Milvus
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from typing import List

# 加载环境变量
load_dotenv()

# 初始化模型
llm = ChatDeepSeek(model="deepseek-chat", temperature=0.1)
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh")

# Milvus 连接配置
MILVUS_URI = "http://localhost:19530"

# 创建游戏场景的描述（粗粒度文档）
scene_descriptions = [
    """
    花果山：这里是齐天大圣孙悟空的出生地。山上常年缭绕着仙气，瀑布从千米高空倾泻而下，
    形成"天河飞瀑"。山中生长着各种仙草灵药，还有不少修炼成精的动物。
    """,
    """
    水帘洞：位于花果山之巅，洞前有一道天然形成的水帘，既是天然屏障，也是修炼圣地。
    """,
    """
    东海龙宫：位于东海海底的宏伟宫殿，由珊瑚和夜明珠装饰。这里是孙悟空取定海神针的地方。
    """
]

# 创建场景详细信息（细粒度文档）
scene_details = {
    "花果山": """
    花果山详细设定
    1. 地理位置：东胜神洲傲来国境内
    2. 自然环境：终年不谢的奇花异草，清澈的山泉和瀑布，茂密的古树森林
    3. 特殊区域：仙果园，种植各种灵果；练功场，平坦开阔的修炼区域；休憩区，供猴族休息的场所
    """,
    "水帘洞": """
    水帘洞详细设定
    1. 建筑结构：外部，巨大的天然岩石洞窟；入口，高30丈的水帘瀑布；内部，错综复杂的洞穴系统
    2. 功能分区：修炼大厅，配备各类修炼器具；藏宝室，存放各种法宝和丹药，有强大的防护阵法；议事厅，可容纳数百猴族，商讨重要事务的地方。
    """,
    "东海龙宫": """
    东海龙宫详细设定
    1. 建筑特征：材质，珊瑚、珍珠、夜明珠；规模，占地数十里；风格，海底宫殿建筑群。
    2. 重要场所：龙王宝库，储存着无数珍宝，如夜明珠，也存放镇海神针等神器；兵器库，各式水系法器，各种神兵利器；大殿，会见宾客的正殿，可召开水族会议。
    """
}

# 第一层：粗粒度索引（场景概述）
coarse_documents = []
for idx, desc in enumerate(scene_descriptions):
    # 提取场景名称
    scene_name = desc.split("：")[0].strip()
    doc = Document(
        page_content=desc,
        metadata={"scene_name": scene_name, "level": "coarse"}
    )
    coarse_documents.append(doc)

# 创建粗粒度向量存储
coarse_vectorstore = Milvus.from_documents(
    coarse_documents, 
    embeddings,
    collection_name="coarse_scenes_langchain",
    connection_args={"uri": MILVUS_URI},
    drop_old=True  # 删除旧集合
)
coarse_retriever = coarse_vectorstore.as_retriever(search_kwargs={"k": 1})

# 第二层：细粒度索引（场景详情）
fine_documents = []
for scene_name, detail in scene_details.items():
    doc = Document(
        page_content=detail,
        metadata={"scene_name": scene_name, "level": "fine"}
    )
    fine_documents.append(doc)

# 创建细粒度向量存储
fine_vectorstore = Milvus.from_documents(
    fine_documents, 
    embeddings,
    collection_name="fine_scenes_langchain",
    connection_args={"uri": MILVUS_URI},
    drop_old=True  # 删除旧集合
)

def query_scene_two_stage(question: str):
    """
    两阶段检索：
    1. 粗粒度检索：找到相关场景
    2. 细粒度检索：在该场景的详细信息中查找
    """
    print(f"问题：{question}\n")
    
    # 第一阶段：粗粒度检索
    coarse_results = coarse_retriever.invoke(question)
    
    if not coarse_results:
        print("未找到相关场景")
        return
    
    # 获取匹配的场景名称
    matched_scene = coarse_results[0].metadata["scene_name"]
    print(f"【第一阶段】粗粒度检索结果：匹配到场景 - {matched_scene}")
    print(f"场景概述：{coarse_results[0].page_content.strip()}\n")
    
    # 第二阶段：细粒度检索
    # 在细粒度文档中过滤出匹配场景的详细信息
    fine_retriever = fine_vectorstore.as_retriever(
        search_kwargs={
            "k": 1,
            "expr": f'scene_name == "{matched_scene}"'  # Milvus 使用 expr 进行过滤
        }
    )
    
    # 获取详细信息
    fine_results = fine_retriever.invoke(question)
    
    if not fine_results:
        print("未找到详细信息")
        return
    
    # 使用 LLM 生成答案
    context = fine_results[0].page_content
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个游戏场景专家。基于提供的上下文信息回答问题。"),
        ("human", "上下文：\n{context}\n\n问题：{question}\n\n请提供详细的回答：")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    
    print(f"【第二阶段】细粒度检索结果：")
    print(f"回答：{response.content}\n")
    
    # 显示使用的源文档
    print(f"使用的详细信息来源：{matched_scene}")
    print(f"详细信息片段：")
    print(context[:200] + "...")
    
    print("-" * 80)

def query_scene_direct(question: str):
    """
    直接在所有细粒度文档中检索（对比用）
    """
    print(f"问题：{question}\n")
    
    # 直接在细粒度文档中检索
    fine_retriever = fine_vectorstore.as_retriever(search_kwargs={"k": 1})
    fine_results = fine_retriever.invoke(question)
    
    if not fine_results:
        print("未找到相关信息")
        return
    
    # 使用 LLM 生成答案
    context = fine_results[0].page_content
    matched_scene = fine_results[0].metadata["scene_name"]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个游戏场景专家。基于提供的上下文信息回答问题。"),
        ("human", "上下文：\n{context}\n\n问题：{question}\n\n请提供详细的回答：")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    
    print(f"【直接检索】结果：")
    print(f"回答：{response.content}\n")
    print(f"匹配的场景：{matched_scene}")
    
    print("-" * 80)

# 示例查询
if __name__ == "__main__":
    questions = [
        "花果山里有什么特别的地方？",
        "详细描述一下水帘洞的内部结构。",
        "东海龙宫存放了哪些宝物？",
    ]
    
    print("=" * 80)
    print("方法1：两阶段检索（粗中有细）")
    print("=" * 80)
    for q in questions:
        query_scene_two_stage(q)
    
    print("\n\n" + "=" * 80)
    print("方法2：直接检索（对比）")
    print("=" * 80)
    for q in questions:
        query_scene_direct(q)
    
    print("\n\n" + "=" * 80)
    print("总结：")
    print("=" * 80)
    print("""
两阶段检索的优势：
1. 先用粗粒度快速定位到相关场景（减少搜索范围）
2. 再在该场景的详细信息中精确查找（提高精度）
3. 适合大规模数据，可以显著提升检索速度

直接检索的特点：
1. 一次性在所有详细信息中检索
2. 实现简单，但在数据量大时效率较低
3. 适合小规模数据
    """)
