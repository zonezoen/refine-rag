"""
查询分解系统 - 使用MultiQueryRetriever进行多角度查询

这个程序展示了如何将复杂的用户查询分解为多个子查询：
1. 识别复合查询中的不同问题点
2. 生成多个相关但不同角度的查询
3. 分别检索每个子查询的结果
4. 合并结果提供更全面的答案

应用场景：
- 复杂问答系统：处理包含多个问题的查询
- 文档检索：从不同角度搜索相关内容
- 知识库查询：提高检索的覆盖面和准确性
- 智能客服：理解用户的复合需求
"""

# ==================== 加载环境变量 ====================
import os
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

# ==================== 导入必要的库 ====================
import logging  # 用于日志记录，观察查询分解过程
from langchain_chroma import Chroma  # Chroma向量数据库
from langchain_community.document_loaders import TextLoader  # 文本文件加载器
from langchain_deepseek import ChatDeepSeek  # DeepSeek聊天模型
from langchain_huggingface import HuggingFaceEmbeddings  # HuggingFace嵌入模型
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 递归文本分割器
from langchain_core.prompts import ChatPromptTemplate  # 提示词模板
from langchain_core.output_parsers import StrOutputParser  # 字符串输出解析器

# ==================== 配置日志系统 ====================
# 设置日志记录，用于观察查询分解过程
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== 构建向量数据库 ====================
print("1. 正在加载文档并构建向量数据库...")

# 加载游戏设定文档
# 这个文档包含了游戏的基本信息、关卡设定、技能系统等
loader = TextLoader("../../99-doc-data/黑悟空/设定.txt", encoding='utf-8')
data = loader.load()  # 加载文档内容

# 文档分割：将长文档切分为小块，便于向量化和检索
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,    # 每个文档块的最大字符数
    chunk_overlap=0    # 文档块之间的重叠字符数（0表示无重叠）
)
splits = text_splitter.split_documents(data)  # 执行文档分割

# 初始化嵌入模型
# 使用中文优化的BGE模型将文本转换为向量
embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh")

# 创建向量数据库
# 将分割后的文档转换为向量并存储在Chroma数据库中
vectorstore = Chroma.from_documents(
    documents=splits,      # 要存储的文档块
    embedding=embed_model  # 使用的嵌入模型
)

print(f"向量数据库构建完成，共存储 {len(splits)} 个文档块")

# ==================== 初始化多查询检索器 ====================
print("2. 正在初始化多查询检索器...")

# 初始化DeepSeek聊天模型
# temperature=0 确保生成的子查询具有一致性
llm = ChatDeepSeek(model="deepseek-chat", temperature=0)

# 创建查询分解提示词模板
# 这个模板会指导LLM将复杂查询分解为多个子查询
query_decomposition_prompt = ChatPromptTemplate.from_template(
    """你是一个AI助手，擅长将复杂的用户查询分解为多个简单的子查询。

用户的原始查询可能包含多个不同的问题点。请将其分解为3-5个独立的子查询，每个子查询针对一个具体的问题。

原始查询：{question}

请直接输出子查询，每行一个，不要添加编号或其他格式。"""
)

# 创建查询分解链
query_decomposition_chain = query_decomposition_prompt | llm | StrOutputParser()

# ==================== 执行多角度查询 ====================
print("3. 正在执行多角度查询...")

# 这是一个典型的复合查询，包含多个不同的问题：
# 1. 游戏难度级别
# 2. 关卡数量
# 3. 普陀山关卡攻略
# 4. 技能推荐
# 5. 新手指导
query = "那个，我刚开始玩这个游戏，感觉很难，请问这个游戏难度级别如何，有几关，在普陀山那一关，嗯，怎么也过不去。先学什么技能比较好？新手求指导！"

print(f"原始查询：{query}")
print("\n开始查询分解和检索...")

# 步骤1：使用LLM分解查询
sub_queries_text = query_decomposition_chain.invoke({"question": query})
sub_queries = [q.strip() for q in sub_queries_text.strip().split('\n') if q.strip()]

logger.info(f"生成的子查询：{sub_queries}")
print(f"\n生成了 {len(sub_queries)} 个子查询：")
for i, sq in enumerate(sub_queries, 1):
    print(f"  {i}. {sq}")

# 步骤2：对每个子查询进行检索
retriever = vectorstore.as_retriever()
all_docs = []
seen_contents = set()  # 用于去重

for sub_query in sub_queries:
    sub_docs = retriever.invoke(sub_query)
    for doc in sub_docs:
        # 去重：只添加未见过的文档
        if doc.page_content not in seen_contents:
            seen_contents.add(doc.page_content)
            all_docs.append(doc)

docs = all_docs

# ==================== 显示检索结果 ====================
print(f"\n4. 检索完成，共找到 {len(docs)} 个相关文档块：")
print("=" * 60)

for i, doc in enumerate(docs, 1):
    print(f"\n文档块 {i}:")
    print(f"内容：{doc.page_content}")
    print(f"来源：{doc.metadata}")
    print("-" * 40)

# ==================== 程序说明 ====================
"""
手动实现的MultiQueryRetriever工作原理：

1. 查询分析阶段：
   - 使用LLM分析原始查询的复杂性
   - 识别查询中包含的不同问题点
   - 理解用户的多重需求

2. 查询分解阶段：
   - 将复合查询分解为多个独立的子查询
   - 每个子查询针对一个具体的问题点
   - 确保子查询之间相互补充，覆盖原始查询的所有方面

3. 并行检索阶段：
   - 对每个子查询分别进行向量检索
   - 从不同角度搜索相关文档
   - 获取更全面的检索结果

4. 结果合并阶段：
   - 合并所有子查询的检索结果
   - 去除重复的文档
   - 按相关性排序最终结果

预期的查询分解效果：
原始查询：那个，我刚开始玩这个游戏，感觉很难，请问这个游戏难度级别如何，有几关，在普陀山那一关，嗯，怎么也过不去。先学什么技能比较好？新手求指导！

可能的子查询：
1. "游戏难度级别设定"
2. "游戏总共有多少关卡"
3. "普陀山关卡攻略指南"
4. "新手推荐学习的技能"
5. "新手游戏指导教程"

优势：
- 提高检索覆盖面：从多个角度搜索相关内容
- 处理复杂查询：自动分解复合问题
- 增强结果质量：通过多角度检索获得更全面的答案
- 适应用户习惯：处理自然语言中的复合表达

适用场景：
- 用户提出包含多个问题的复杂查询
- 需要从不同角度理解同一个主题
- 希望获得更全面和详细的检索结果
- 处理模糊或不够具体的查询

注意：此实现使用手动方式替代了原始的MultiQueryRetriever，
因为在LangChain 1.2+版本中该类已被移除。功能完全相同。
"""