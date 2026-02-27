"""
逻辑路由系统 - 基于规则的查询路由

这个程序展示了如何使用大语言模型进行智能查询路由：
1. 分析用户查询的内容和意图
2. 根据预定义的逻辑规则选择合适的数据源
3. 使用结构化输出确保路由结果的准确性
4. 支持多种数据源的动态路由

应用场景：
- 多数据源问答系统：根据问题类型选择最合适的知识库
- 智能客服：将用户问题路由到不同的专业团队
- 文档检索：在多个文档集合中选择最相关的进行搜索
- 微服务架构：将请求路由到合适的服务模块
"""

# ==================== 导入必要的库 ====================
import os
from typing import Literal  # 用于定义字面量类型，限制可选值
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate  # 聊天提示模板
from pydantic import BaseModel, Field  # Pydantic数据模型
from langchain_deepseek import ChatDeepSeek  # DeepSeek聊天模型

# 加载环境变量
load_dotenv()

# ==================== 定义路由数据模型 ====================
class RouteQuery(BaseModel):
    """
    查询路由的数据模型
    
    使用Pydantic定义结构化的输出格式，确保路由结果的类型安全：
    1. 限制可选的数据源类型
    2. 提供清晰的字段描述
    3. 支持自动验证和序列化
    """
    datasource: Literal["python_docs", "js_docs", "golang_docs"] = Field(
        ...,  # 必填字段
        description="根据用户问题，选择最适合回答问题的数据源",
    )
    
    # 可以扩展更多字段，例如：
    # confidence: float = Field(description="路由决策的置信度")
    # reasoning: str = Field(description="路由决策的理由")

# ==================== 创建路由器函数 ====================
def create_router():
    """
    创建并返回智能路由模型
    
    这个函数构建了一个完整的路由链：
    1. 初始化大语言模型
    2. 配置结构化输出
    3. 设计路由提示模板
    4. 组装路由处理链
    
    返回:
        配置好的路由处理链
    """
    # 初始化DeepSeek聊天模型
    # temperature=0 确保路由决策的一致性和可预测性
    llm = ChatDeepSeek(model="deepseek-chat", temperature=0)
    
    # 配置结构化输出
    # 这确保模型的输出严格遵循RouteQuery的格式
    structured_llm = llm.with_structured_output(RouteQuery)
    
    # 设计系统提示词
    # 这个提示词定义了路由器的角色和工作方式
    system = """你是将用户问题路由到合适数据源的专家。

你的任务是分析用户的问题，并根据以下规则选择最适合的数据源：

1. python_docs: 
   - Python语言相关的问题
   - Python库和框架的使用
   - Python语法、特性、最佳实践
   
2. js_docs:
   - JavaScript语言相关的问题
   - 前端开发、Node.js相关问题
   - JavaScript框架和库的使用
   
3. golang_docs:
   - Go语言相关的问题
   - Go语法、并发编程、性能优化
   - Go生态系统和工具链

分析步骤：
1. 识别问题中提到的编程语言关键词
2. 理解问题的技术领域和上下文
3. 选择最匹配的数据源
4. 如果问题涉及多种语言，选择主要关注的语言"""

    # 创建聊天提示模板
    # 结合系统提示和用户问题，形成完整的输入
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),      # 系统角色和指令
        ("human", "{question}"), # 用户问题占位符
    ])
    
    # 构建路由处理链
    # 使用LangChain的管道操作符连接提示模板和模型
    return prompt | structured_llm

# ==================== 路由执行函数 ====================
def route_question(question: str) -> str:
    """
    路由用户问题到合适的数据源
    
    这个函数是路由系统的主要接口：
    1. 接收用户的自然语言问题
    2. 调用路由器进行智能分析
    3. 返回选定的数据源标识
    
    参数:
        question: 用户的问题文本
        
    返回:
        选定的数据源名称（python_docs/js_docs/golang_docs）
    """
    # 创建路由器实例
    router = create_router()
    
    # 执行路由决策
    # invoke方法会将问题传递给路由链进行处理
    result = router.invoke({"question": question})
    
    # 返回路由结果
    return result.datasource

# ==================== 测试和演示 ====================
if __name__ == "__main__":
    print("=== 逻辑路由系统测试 ===\n")
    
    # 定义多个测试问题，覆盖不同的编程语言
    test_questions = [
        "Python中的列表和元组有什么区别？",
        "JavaScript中如何实现异步编程？",
        "Go语言的goroutine是如何工作的？",
        "如何在Python中处理JSON数据？",
        "React和Vue.js有什么区别？",
        "Go语言中的channel有什么用途？"
    ]
    
    # 对每个问题进行路由测试
    for i, question in enumerate(test_questions, 1):
        print(f"测试 {i}:")
        print(f"问题: {question}")
        
        try:
            result = route_question(question)
            print(f"路由结果: {result}")
            
            # 根据路由结果提供说明
            source_descriptions = {
                "python_docs": "Python文档库 - 适合Python相关问题",
                "js_docs": "JavaScript文档库 - 适合前端和JS相关问题", 
                "golang_docs": "Go语言文档库 - 适合Go语言相关问题"
            }
            print(f"说明: {source_descriptions.get(result, '未知数据源')}")
            
        except Exception as e:
            print(f"路由失败: {e}")
        
        print("-" * 50)

# ==================== 程序说明 ====================
"""
逻辑路由的工作原理：

1. 问题分析阶段：
   - 使用大语言模型理解用户问题的语义
   - 识别问题中的关键技术词汇和上下文
   - 分析问题所属的技术领域

2. 规则匹配阶段：
   - 根据预定义的路由规则进行匹配
   - 考虑编程语言关键词、技术概念、应用场景
   - 处理模糊或跨领域的问题

3. 决策输出阶段：
   - 使用结构化输出确保结果格式正确
   - 返回明确的数据源标识
   - 支持后续的检索和处理流程

4. 扩展能力：
   - 可以轻松添加新的数据源类型
   - 支持更复杂的路由逻辑和条件
   - 可以集成置信度评估和多路由支持

优势：
- 智能化：使用AI理解问题语义，不依赖简单的关键词匹配
- 准确性：结构化输出确保路由结果的类型安全
- 可扩展：易于添加新的数据源和路由规则
- 可维护：清晰的代码结构和完整的文档说明

实际应用：
- 技术文档问答系统：根据问题选择合适的技术文档库
- 多语言代码助手：为不同编程语言提供专门的帮助
- 企业知识管理：将问题路由到相关部门的知识库
- 智能客服系统：根据问题类型分配给合适的客服团队
"""

