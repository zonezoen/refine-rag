"""
语义路由系统 - 基于向量相似度的智能路由

这个程序展示了如何使用向量嵌入和余弦相似度进行语义路由：
1. 将不同的提示模板转换为向量表示
2. 将用户查询也转换为向量
3. 通过计算相似度选择最匹配的模板
4. 动态路由到最合适的处理流程

应用场景：
- 多专家系统：根据问题类型选择合适的专家模板
- 智能对话：根据用户意图选择不同的回答风格
- 内容分类：将内容路由到不同的处理管道
- 个性化服务：根据用户偏好选择合适的服务模式

注意：这个示例使用本地 HuggingFace 嵌入模型和 DeepSeek API
"""

# ==================== 导入必要的库 ====================
import os
from dotenv import load_dotenv

# 加载环境变量 - 必须在导入其他模块之前
load_dotenv()

from langchain_community.utils.math import cosine_similarity  # 余弦相似度计算工具
from langchain_core.output_parsers import StrOutputParser  # 字符串输出解析器
from langchain_core.prompts import PromptTemplate  # 提示模板
from langchain_core.runnables import RunnableLambda, RunnablePassthrough  # 可运行组件
from langchain_huggingface import HuggingFaceEmbeddings  # HuggingFace 嵌入模型
from langchain_deepseek import ChatDeepSeek  # DeepSeek 聊天模型

# ==================== 定义专家提示模板 ====================
# 这些模板代表不同的专家角色，每个都有特定的专业领域和回答风格

# 战斗技巧专家模板
# 专注于游戏战斗机制、技能使用、战术策略等问题
combat_template = """你是一位精通黑悟空战斗技巧的专家。
你擅长以简洁易懂的方式回答关于黑悟空战斗的问题。
你的专业领域包括：
- 战斗技能和连招
- 敌人弱点分析
- 装备选择和升级
- 战斗策略和技巧

当你不知道问题的答案时，你会坦诚相告。

以下是一个问题：
{query}"""

# 故事情节专家模板
# 专注于游戏剧情、角色背景、世界观设定等问题
story_template = """你是一位熟悉黑悟空故事情节的专家。
你擅长将复杂的情节分解并详细解释。
你的专业领域包括：
- 游戏主线剧情
- 角色背景故事
- 世界观和设定
- 隐藏剧情和彩蛋

当你不知道问题的答案时，你会坦诚相告。

以下是一个问题：
{query}"""

# ==================== 初始化嵌入系统 ====================
print("正在初始化语义路由系统...")

# 使用本地的 HuggingFace 嵌入模型
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh")

# 准备提示模板列表
prompt_templates = [combat_template, story_template]

# 将所有提示模板转换为向量表示
# 这些向量将作为路由决策的基准
print("正在生成提示模板的向量表示...")
prompt_embeddings = embeddings.embed_documents(prompt_templates)

print(f"成功生成 {len(prompt_embeddings)} 个模板向量")

# ==================== 定义语义路由函数 ====================
def prompt_router(input):
    """
    基于语义相似度的提示模板路由器
    
    工作原理：
    1. 将用户查询转换为向量
    2. 计算查询向量与所有模板向量的余弦相似度
    3. 选择相似度最高的模板
    4. 返回对应的提示模板对象
    
    参数:
        input: 包含用户查询的字典，格式为 {"query": "用户问题"}
        
    返回:
        最匹配的PromptTemplate对象
    """
    # 步骤1：将用户问题转换为向量表示
    query_embedding = embeddings.embed_query(input["query"])
    
    # 步骤2：计算查询向量与所有模板向量的余弦相似度
    # cosine_similarity返回一个相似度矩阵，[0]取第一行（查询向量的相似度）
    similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
    
    # 步骤3：找到相似度最高的模板索引
    max_similarity_index = similarity.argmax()
    most_similar = prompt_templates[max_similarity_index]
    
    # 步骤4：输出路由决策信息（用于调试和监控）
    if most_similar == combat_template:
        print(f"🗡️  路由决策：使用战斗技巧模板 (相似度: {similarity[max_similarity_index]:.4f})")
    else:
        print(f"📖 路由决策：使用故事情节模板 (相似度: {similarity[max_similarity_index]:.4f})")
    
    # 步骤5：返回选定的提示模板
    return PromptTemplate.from_template(most_similar)

# ==================== 构建处理链 ====================
print("正在构建语义路由处理链...")

# 初始化 DeepSeek 模型（在环境变量加载之后）
llm = ChatDeepSeek(model="deepseek-chat", temperature=0.7)

# 使用LangChain的管道操作符构建完整的处理流程
chain = (
    # 步骤1：准备输入数据，将查询包装为字典格式
    {"query": RunnablePassthrough()}
    
    # 步骤2：执行语义路由，选择合适的提示模板
    | RunnableLambda(prompt_router)
    
    # 步骤3：使用选定的模板调用 DeepSeek 生成回答
    | llm  # 适度的创造性，保持回答的多样性
    
    # 步骤4：解析输出为字符串格式
    | StrOutputParser()
)

print("语义路由系统初始化完成！\n")

# ==================== 测试和演示 ====================
if __name__ == "__main__":
    print("=== 语义路由系统测试 ===\n")
    
    # 定义测试问题，涵盖不同的语义类别
    test_questions = [
        "黑悟空是如何打败敌人的？",           # 战斗相关
        "游戏的主要剧情是什么？",             # 故事相关
        "最强的技能组合是什么？",             # 战斗相关
        "悟空的身世背景是怎样的？",           # 故事相关
        "如何提升战斗力？",                   # 战斗相关
        "游戏中有哪些重要的角色？"            # 故事相关
    ]
    
    # 对每个问题进行测试
    for i, question in enumerate(test_questions, 1):
        print(f"测试 {i}: {question}")
        print("-" * 50)
        
        try:
            # 执行语义路由和回答生成
            answer = chain.invoke(question)
            print(f"回答: {answer}")
            
        except Exception as e:
            print(f"处理失败: {e}")
        
        print("\n" + "=" * 60 + "\n")

# ==================== 程序说明 ====================
"""
语义路由的工作原理：

1. 向量化阶段：
   - 将所有提示模板转换为向量表示
   - 每个向量捕获了模板的语义特征
   - 向量维度通常为1536（OpenAI嵌入模型）

2. 查询处理阶段：
   - 将用户查询转换为同样维度的向量
   - 确保查询和模板在同一向量空间中

3. 相似度计算阶段：
   - 使用余弦相似度衡量向量间的语义相似性
   - 余弦相似度范围为[-1, 1]，值越大表示越相似
   - 不受向量长度影响，只关注方向

4. 路由决策阶段：
   - 选择相似度最高的模板
   - 动态构建对应的提示模板对象
   - 将查询路由到最合适的处理流程

优势：
- 语义理解：基于真实的语义相似性，而非关键词匹配
- 自动化：无需手动定义复杂的路由规则
- 灵活性：可以轻松添加新的专家模板
- 准确性：向量相似度提供量化的匹配度量

扩展可能：
- 添加更多专家模板（装备专家、探索专家等）
- 引入阈值机制，处理模糊查询
- 集成多模板融合，处理跨领域问题
- 添加用户反馈机制，优化路由准确性

实际应用：
- 游戏客服系统：根据问题类型分配专业客服
- 知识库问答：在不同专业领域间智能路由
- 内容推荐：根据用户兴趣选择推荐策略
- 多模态对话：根据对话上下文调整回答风格
"""
