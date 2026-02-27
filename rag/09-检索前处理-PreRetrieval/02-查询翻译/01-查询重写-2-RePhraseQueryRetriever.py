import os
import logging
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 加载环境变量
load_dotenv()

# 设置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# 建议通过环境变量设置：export DEEPSEEK_API_KEY='你的key'
# 或者在此处临时手动设置（不建议在生产环境硬编码）
# os.environ["DEEPSEEK_API_KEY"] = "sk-xxxxxxxxxxxxxxxx"

def run_rag_query():
    # 3. 加载游戏文档数据
    file_path = "../../99-doc-data/黑悟空/设定.txt"
    if not os.path.exists(file_path):
        print(f"错误：找不到文件 {file_path}")
        return

    loader = TextLoader(file_path, encoding='utf-8')
    data = loader.load()

    # 4. 文本分块 (增加了一些 overlap 以保持上下文连贯)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    all_splits = text_splitter.split_documents(data)

    # 5. 创建向量存储 (首次运行会自动下载 BAAI/bge-small-zh 模型)
    embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh")
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=embed_model)

    # 6. 设置 DeepSeek 作为重写查询的推理大脑
    llm = ChatDeepSeek(
        model="deepseek-chat",
        temperature=0,
        # api_key="如果环境变量没设，可在此传入"
    )

    # 7. 构建 MultiQueryRetriever
    # 它的作用：把“那个...嗯...怎么也过不去”这种口语变成“普陀山 关卡攻略”、“新手技能推荐”等精准词
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(),
        llm=llm
    )

    # 8. 示例输入：处理口语化、含糊的问题
    query = "那个，我刚开始玩这个游戏，感觉很难，在普陀山那一关，嗯，怎么也过不去。先学什么技能比较好？新手求指导！"

    # 9. 执行检索
    print("\n[系统] 正在分析问题并检索知识库...\n")
    docs = retriever_from_llm.invoke(query)

    # 10. 打印结果
    print(f"--- 检索到 {len(docs)} 个相关的知识片段 ---\n")
    for i, doc in enumerate(docs):
        print(f"片段 {i+1}:\n{doc.page_content}\n{'-'*20}")

if __name__ == "__main__":
    run_rag_query()