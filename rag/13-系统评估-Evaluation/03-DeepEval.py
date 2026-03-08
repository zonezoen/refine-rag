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

# 初始化千问模型
qwen_model = QwenModel()

# 定义测试案例
test_case = LLMTestCase(
    input="如果这双鞋不合脚怎么办？",
    actual_output="我们提供30天无理由全额退款服务。",
    expected_output="顾客可以在30天内退货并获得全额退款。",
    retrieval_context=["所有顾客都有资格享受30天无理由全额退款服务。"]
)

# 定义评估指标
contextual_precision = ContextualRelevancyMetric(
    threshold=0.7,
    model=qwen_model,
    include_reason=True
)
answer_relevancy = AnswerRelevancyMetric(
    threshold=0.7,  # 设置阈值，低于0.7视为不通过
    model=qwen_model,  # 使用千问作为评判模型
    include_reason=True  # 包含评分理由
)

# 运行评估
contextual_precision.measure(test_case)
answer_relevancy.measure(test_case)

print("上下文精确度得分: ", contextual_precision.score)
print("理由：", contextual_precision.reason)
print("答案相关性得分: ", answer_relevancy.score)
print("理由：", answer_relevancy.reason)
