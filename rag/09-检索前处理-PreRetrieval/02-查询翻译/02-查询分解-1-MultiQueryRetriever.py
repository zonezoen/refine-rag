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

# ==================== 导入必要的库 ====================
import logging  # 用于日志记录，观察查询分解过程
from langchain_chroma import Chroma  # Chroma向量数据库
from langchain_community.document_loaders import TextLoader  # 文本文件加载器
from langchain_deepseek import ChatDeepSeek  # DeepSeek聊天模型
from langchain_huggingface import HuggingFaceEmbeddings  # HuggingFace嵌入模型
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 递归文本分割器
from langchain.retrievers.multi_query import MultiQueryRetriever  # 多角度查询检索器

# ==================== 配置日志系统 ====================
# 设置日志记录，用于观察MultiQueryRetriever的工作过程
logging.basicConfig()
# 设置MultiQueryRetriever的日志级别为INFO，可以看到生成的子查询
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

# ==================== 构建向量数据库 ====================
print("1. 正在加载文档并构建向量数据库...")

# 加载游戏设定文档
# 这个文档包含了游戏的基本信息、关卡设定、技能系统等
loader = TextLoader("90-文档-Data/黑悟空/设定.txt", encoding='utf-8')
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

# 创建MultiQueryRetriever
# 这个检索器会自动将用户查询分解为多个子查询
retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),  # 基础检索器（向量数据库检索器）
    llm=llm                                # 用于生成子查询的语言模型
)

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

# 调用MultiQueryRetriever进行查询分解和检索
# 这个过程会：
# 1. 使用LLM将原始查询分解为多个子查询
# 2. 对每个子查询分别进行向量检索
# 3. 合并所有检索结果并去重
docs = retriever_from_llm.invoke(query)

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
MultiQueryRetriever的工作原理：

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
"""