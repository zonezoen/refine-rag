import os
from dotenv import load_dotenv
load_dotenv()  # 加载.env文件中的环境变量

from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import DeepEvalBaseLLM
from langchain_community.chat_models.tongyi import ChatTongyi

# 自定义 DeepEval 模型包装器，适配千问并支持中文输出
class QwenModel(DeepEvalBaseLLM):
    """
    将阿里云千问模型包装成 DeepEval 可用的格式
    DeepEval 需要模型实现 generate 和 get_model_name 方法
    
    关键改进：在 generate 方法中添加中文指令，确保输出中文理由
    """
    def __init__(self):
        # 初始化千问模型
        self.model = ChatTongyi(
            model_name="qwen-max",
            temperature=0,
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
        )
    
    def load_model(self):
        """加载模型（千问通过API调用，无需加载）"""
        return self.model
    
    def generate(self, prompt: str) -> str:
        """
        生成响应的核心方法
        DeepEval 会调用这个方法来获取模型输出
        
        关键：在原始 prompt 前添加中文指令，强制模型用中文回复
        """
        chat_model = self.load_model()
        
        # 在原始 prompt 前添加中文指令
        chinese_instruction = "请用中文回答以下问题。所有的解释、理由和分析都必须使用中文。\n\n"
        modified_prompt = chinese_instruction + prompt
        
        response = chat_model.invoke(modified_prompt)
        return response.content
    
    async def a_generate(self, prompt: str) -> str:
        """异步生成方法"""
        return self.generate(prompt)
    
    def get_model_name(self) -> str:
        """返回模型名称"""
        return "Qwen-Max"

print("\n=== DeepEval 评估框架介绍 ===")
print("DeepEval 是一个像写单元测试一样的 RAG 评估框架")
print("特点：基于 Pytest、指标清晰、易于集成 CI/CD")
print("\n本示例将评估《黑神话：悟空》相关的 RAG 系统")

# 初始化千问模型
qwen_model = QwenModel()

# 准备测试用例
# DeepEval 使用 LLMTestCase 来组织测试数据
test_cases = [
    LLMTestCase(
        input="《黑神话：悟空》的主角是谁？",
        actual_output="《黑神话：悟空》的主角是孙悟空，基于中国古典名著《西游记》，但进行了全新的诠释。这个版本的孙悟空更加成熟和沉稳，展现出与传统形象不同的性格特点。",
        retrieval_context=[
            "《黑神话：悟空》是游戏科学开发的动作角色扮演游戏，以《西游记》中的孙悟空为主角，但进行了创新性的诠释。游戏中的悟空性格更加沉稳，肩负着特殊的使命。",
            "游戏设定在神话世界中，讲述了一个全新的故事，呈现出与传统孙悟空形象不同的角色塑造。"
        ]
    ),
    LLMTestCase(
        input="《黑神话：悟空》的战斗系统有什么特色？",
        actual_output="《黑神话：悟空》的战斗系统融合了中国武术与魂类游戏特色，包括轻重攻击组合、法术变化和法术系统。值得注意的是，悟空可以在战斗中变换不同的武器形态，如标志性的金箍棒和双节棍，并使用各种神通法术。",
        retrieval_context=[
            "游戏的战斗系统深受魂类游戏影响，同时融入了中国传统武术元素。玩家可以使用不同的武器形态，包括标志性的金箍棒和其他变化武器。",
            "战斗中玩家可以施展各种神通法术，配合轻重攻击和连招系统，打造流畅而独特的战斗体验。游戏还具有独特的变身系统。"
        ]
    ),
    LLMTestCase(
        input="《黑神话：悟空》的画面质量如何？",
        actual_output="《黑神话：悟空》采用虚幻引擎5开发，展现出惊艳的画面质量。游戏的场景建模、光影效果和角色细节都达到了顶级水准，特别是对中国传统建筑和神话场景的细致还原。",
        retrieval_context=[
            "《黑神话：悟空》展现出卓越的视觉质量，采用虚幻引擎5打造，实现了极高的画面保真度。游戏的环境和角色模型都经过精心制作。",
            "光影效果、材质渲染和环境细节都达到了3A级标准，完美捕捉了东方神话世界的氛围。"
        ]
    ),
]

print("\n" + "="*60)
print("评估指标 1: 答案相关性 (Answer Relevancy)")
print("="*60)
print("评估生成的答案是否直接回答了用户的问题")
print("阈值设置: 0.7 (70分以上算通过)")


# 创建答案相关性评估指标
answer_relevancy_metric = AnswerRelevancyMetric(
    threshold=0.7,  # 设置阈值，低于0.7视为不通过
    model=qwen_model,  # 使用千问作为评判模型
    include_reason=True  # 包含评分理由
)

# 评估答案相关性
print("\n开始评估答案相关性...")
for i, test_case in enumerate(test_cases, 1):
    print(f"\n--- 测试用例 {i} ---")
    print(f"问题: {test_case.input}")
    
    # 使用 measure 方法进行评估
    answer_relevancy_metric.measure(test_case)
    
    print(f"答案相关性得分: {answer_relevancy_metric.score:.4f}")
    print(f"是否通过: {'✓ 通过' if answer_relevancy_metric.is_successful() else '✗ 未通过'}")
    if answer_relevancy_metric.reason:
        print(f"评分理由: {answer_relevancy_metric.reason}")

print("\n" + "="*60)
print("评估指标 2: 忠实度 (Faithfulness)")
print("="*60)
print("评估生成的答案是否忠实于检索到的上下文")
print("检查是否存在幻觉（模型自己编造的内容）")
print("阈值设置: 0.7")

# 创建忠实度评估指标
faithfulness_metric = FaithfulnessMetric(
    threshold=0.7,
    model=qwen_model,
    include_reason=True
)

# 评估忠实度
print("\n开始评估忠实度...")
for i, test_case in enumerate(test_cases, 1):
    print(f"\n--- 测试用例 {i} ---")
    print(f"问题: {test_case.input}")
    
    faithfulness_metric.measure(test_case)
    
    print(f"忠实度得分: {faithfulness_metric.score:.4f}")
    print(f"是否通过: {'✓ 通过' if faithfulness_metric.is_successful() else '✗ 未通过'}")
    if faithfulness_metric.reason:
        print(f"评分理由: {faithfulness_metric.reason}")

print("\n" + "="*60)
print("评估指标 3: 上下文相关性 (Contextual Relevancy)")
print("="*60)
print("评估检索到的上下文是否与问题相关")
print("检查检索系统是否找到了正确的文档")
print("阈值设置: 0.7")

# 创建上下文相关性评估指标
contextual_relevancy_metric = ContextualRelevancyMetric(
    threshold=0.7,
    model=qwen_model,
    include_reason=True
)

# 评估上下文相关性
print("\n开始评估上下文相关性...")
for i, test_case in enumerate(test_cases, 1):
    print(f"\n--- 测试用例 {i} ---")
    print(f"问题: {test_case.input}")
    
    contextual_relevancy_metric.measure(test_case)
    
    print(f"上下文相关性得分: {contextual_relevancy_metric.score:.4f}")
    print(f"是否通过: {'✓ 通过' if contextual_relevancy_metric.is_successful() else '✗ 未通过'}")
    if contextual_relevancy_metric.reason:
        print(f"评分理由: {contextual_relevancy_metric.reason}")

print("\n" + "="*60)
print("综合评估：使用 evaluate 函数批量评估")
print("="*60)
print("DeepEval 提供了 evaluate 函数，可以一次性评估多个指标")

# 使用 evaluate 函数进行综合评估
print("\n开始综合评估...")
evaluation_result = evaluate(
    test_cases=test_cases,
    metrics=[
        AnswerRelevancyMetric(threshold=0.7, model=qwen_model),
        FaithfulnessMetric(threshold=0.7, model=qwen_model),
        ContextualRelevancyMetric(threshold=0.7, model=qwen_model)
    ]
)

print("\n" + "="*60)
print("评估总结")
print("="*60)
print(f"总测试用例数: {len(test_cases)}")
print(f"评估指标数: 3 (答案相关性、忠实度、上下文相关性)")
print("\n评估完成！")
print("\nDeepEval 的优势:")
print("1. 像写单元测试一样简单，基于 Pytest 框架")
print("2. 可以设置阈值，自动判断通过/失败")
print("3. 提供详细的评分理由，便于调试")
print("4. 易于集成到 CI/CD 流程中")
print("5. 支持多种评估指标，覆盖 RAG 三元组")


'''
DeepEval 使用说明：

1. 核心概念：
   - LLMTestCase: 测试用例，包含输入、输出、上下文等
   - Metric: 评估指标，如答案相关性、忠实度等
   - evaluate: 批量评估函数

2. 主要评估指标：
   - AnswerRelevancyMetric: 答案相关性
   - FaithfulnessMetric: 忠实度（无幻觉）
   - ContextualRelevancyMetric: 上下文相关性
   - BiasMetric: 偏见检测
   - ToxicityMetric: 毒性检测
   - HallucinationMetric: 幻觉检测

3. 与 RAGAS 的对比：
   DeepEval:
   - 基于 Pytest，更像单元测试
   - 可以设置阈值，自动判断通过/失败
   - 提供详细的评分理由
   - 易于集成 CI/CD
   
   RAGAS:
   - 专注于 RAG 评估
   - 指标更丰富
   - 社区更大

4. 环境要求：
   pip install deepeval langchain-community dashscope python-dotenv
   
   需要在 .env 文件中配置:
   DASHSCOPE_API_KEY=your_api_key

5. 使用 Pytest 运行（可选）：
   可以将评估代码改写成 pytest 测试函数：
   
   def test_answer_relevancy():
       assert answer_relevancy_metric.measure(test_case)
       assert answer_relevancy_metric.is_successful()
   
   然后使用命令运行：
   deepeval test run test_file.py

6. 预期输出示例：
   
   === DeepEval 评估框架介绍 ===
   DeepEval 是一个像写单元测试一样的 RAG 评估框架
   特点：基于 Pytest、指标清晰、易于集成 CI/CD
   
   ============================================================
   评估指标 1: 答案相关性 (Answer Relevancy)
   ============================================================
   
   --- 测试用例 1 ---
   问题: 《黑神话：悟空》的主角是谁？
   答案相关性得分: 0.9234
   是否通过: ✓ 通过
   评分理由: 答案直接回答了问题，明确指出主角是孙悟空...
   
   ============================================================
   评估指标 2: 忠实度 (Faithfulness)
   ============================================================
   
   --- 测试用例 1 ---
   问题: 《黑神话：悟空》的主角是谁？
   忠实度得分: 0.8876
   是否通过: ✓ 通过
   评分理由: 答案内容完全基于提供的上下文...
   
   ============================================================
   评估总结
   ============================================================
   总测试用例数: 3
   评估指标数: 3 (答案相关性、忠实度、上下文相关性)
   
   评估完成！

7. 进阶用法：
   - 自定义评估指标
   - 集成到 CI/CD 流程
   - 生成评估报告
   - 对比不同版本的 RAG 系统

8. 与 RAGAS 代码的主要区别：
   - DeepEval 使用 LLMTestCase 组织数据
   - DeepEval 每个指标可以单独设置阈值
   - DeepEval 提供 is_successful() 方法判断是否通过
   - DeepEval 更强调测试驱动的评估方式
'''
