import os
from dotenv import load_dotenv
load_dotenv() # 加载.env文件中的环境变量
import numpy as np
from datasets import Dataset
from ragas.metrics import Faithfulness, AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from typing import Any, List, Optional
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.callbacks.manager import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun

# 创建一个自定义的 ChatDeepSeek 包装器，强制 n=1
class ChatDeepSeekForRAGAS(ChatDeepSeek):
    """DeepSeek 兼容 RAGAS 的包装器，强制 n=1"""
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # 强制设置 n=1，DeepSeek API 不支持 n>1
        if 'n' in kwargs:
            kwargs['n'] = 1
        return super()._generate(messages, stop, run_manager, **kwargs)
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # 强制设置 n=1，DeepSeek API 不支持 n>1
        if 'n' in kwargs:
            kwargs['n'] = 1
        return await super()._agenerate(messages, stop, run_manager, **kwargs)

# 准备评估用的LLM（使用DeepSeek）
# 使用自定义的 ChatDeepSeekForRAGAS 来强制 n=1
llm = LangchainLLMWrapper(ChatDeepSeekForRAGAS(
    model="deepseek-chat",
    temperature=0
))

# 准备数据集
# 这个数据集包含了问题、生成的答案以及相关的上下文信息
data = {
    "question": [
        "Who is the main character in Black Myth: Wukong?",
        "What are the special features of the combat system in Black Myth: Wukong?",
        "How is the visual quality of Black Myth: Wukong?",
    ],
    "answer": [
        "The main character in Black Myth: Wukong is Sun Wukong, based on the Chinese classic 'Journey to the West' but with a new interpretation. This version of Sun Wukong is more mature and brooding, showing a different personality from the traditional character.",
        "Black Myth: Wukong's combat system combines Chinese martial arts with Soulslike game features, including light and heavy attack combinations, technique transformations, and magic systems. Notably, Wukong can transform between different weapon forms during combat, such as his iconic staff and nunchucks, and use various mystical abilities.",
        "Black Myth: Wukong is developed using Unreal Engine 5, showcasing stunning visual quality. The game's scene modeling, lighting effects, and character details are all top-tier, particularly in its detailed recreation of traditional Chinese architecture and mythological settings.",
    ],
    "contexts": [
        [
            "Black Myth: Wukong is an action RPG developed by Game Science, featuring Sun Wukong as the protagonist based on 'Journey to the West' but with innovative interpretations. In the game, Wukong has a more composed personality and carries a special mission.",
            "The game is set in a mythological world, telling a new story that presents a different take on the traditional Sun Wukong character."
        ],
        [
            "The game's combat system is heavily influenced by Soulslike games while incorporating traditional Chinese martial arts elements. Players can utilize different weapon forms, including the iconic staff and other transforming weapons.",
            "During combat, players can unleash various mystical abilities, combined with light and heavy attacks and combo systems, creating a fluid and distinctive combat experience. The game also features a unique transformation system."
        ],
        [
            "Black Myth: Wukong demonstrates exceptional visual quality, built with Unreal Engine 5, achieving extremely high graphical fidelity. The game's environments and character models are meticulously crafted.",
            "The lighting effects, material rendering, and environmental details all reach AAA-level standards, perfectly capturing the atmosphere of an Eastern mythological world."
        ]
    ]
}

# 将字典转换为Hugging Face的Dataset对象，方便Ragas处理
dataset = Dataset.from_dict(data)

print("\n=== Ragas评估指标说明 ===")
print("\n1. Faithfulness（忠实度）")
print("- 评估生成的答案是否忠实于上下文内容")
print("- 通过将答案分解为简单陈述，然后验证每个陈述是否可以从上下文中推断得出")
print("- 该指标仅依赖LLM，不需要embedding模型")

# 评估Faithfulness
# 创建Faithfulness评估指标，它只需要一个LLM来进行评估
faithfulness_metric = [Faithfulness(llm=llm)] # 只需要提供生成模型
print("\n正在评估忠实度...")
# 使用evaluate函数对数据集进行评估
faithfulness_result = evaluate(dataset, faithfulness_metric)
# 提取忠实度分数
scores = faithfulness_result['faithfulness']
# 计算平均分
mean_score = np.mean(scores) if isinstance(scores, (list, np.ndarray)) else scores
print(f"忠实度评分: {mean_score:.4f}")

print("\n2. AnswerRelevancy（答案相关性）")
print("- 评估生成的答案与问题的相关程度")
print("- 使用embedding模型计算语义相似度")
print("- 我们将比较两种开源embedding模型")

# 设置两种embedding模型
# 使用Ragas的LangchainEmbeddingsWrapper来包装LangChain的嵌入模型
# 1. 开源的 all-MiniLM-L6-v2 模型（英文优化）
embedding_en = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
)
# 2. 开源的 bge-small-zh 模型（中文优化）
embedding_zh = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh")
)

# 创建答案相关性评估指标
# 分别为两种embedding模型创建AnswerRelevancy评估指标
relevancy_en = [AnswerRelevancy(llm=llm, embeddings=embedding_en)]
relevancy_zh = [AnswerRelevancy(llm=llm, embeddings=embedding_zh)]

print("\n正在评估答案相关性...")
print("\n使用英文优化Embedding模型评估 (all-MiniLM-L6-v2):")
# 使用英文优化embedding模型进行评估
result_en = evaluate(dataset, relevancy_en)
scores = result_en['answer_relevancy']
mean_en = np.mean(scores) if isinstance(scores, (list, np.ndarray)) else scores
print(f"相关性评分: {mean_en:.4f}")

print("\n使用中文优化Embedding模型评估 (bge-small-zh):")
# 使用中文优化embedding模型进行评估
result_zh = evaluate(dataset, relevancy_zh)
scores = result_zh['answer_relevancy']
mean_zh = np.mean(scores) if isinstance(scores, (list, np.ndarray)) else scores
print(f"相关性评分: {mean_zh:.4f}")

# 比较两种embedding模型的结果
print("\n=== Embedding模型比较 ===")
diff = mean_zh - mean_en
print(f"英文模型评分: {mean_en:.4f}")
print(f"中文模型评分: {mean_zh:.4f}")
print(f"差异: {diff:.4f} ({'中文模型更好' if diff > 0 else '英文模型更好' if diff < 0 else '相当'})")


'''
修改说明：
1. 将OpenAI的GPT-3.5替换为DeepSeek API（兼容OpenAI接口）
2. 移除OpenAI的embedding模型，改用两种开源HuggingFace模型：
   - all-MiniLM-L6-v2: 英文优化的轻量级模型
   - bge-small-zh: 中文优化的BGE模型
3. 使用LangchainEmbeddingsWrapper包装LangChain的HuggingFaceEmbeddings
4. 所有API调用都使用项目中已配置的DeepSeek密钥

技术要点：
- LangChain的HuggingFaceEmbeddings是完整实现，包含所有必要方法
- LangchainEmbeddingsWrapper将LangChain嵌入模型适配到RAGAS接口
- 包装器自动处理同步和异步方法转换
- DeepSeek API完全兼容OpenAI的接口格式

预期输出示例：
=== Ragas评估指标说明 ===

1. Faithfulness（忠实度）
- 评估生成的答案是否忠实于上下文内容
- 通过将答案分解为简单陈述，然后验证每个陈述是否可以从上下文中推断得出
- 该指标仅依赖LLM，不需要embedding模型

正在评估忠实度...
Evaluating: 100%|███████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:05<00:00,  1.87s/it]
忠实度评分: 0.6071

2. AnswerRelevancy（答案相关性）
- 评估生成的答案与问题的相关程度
- 使用embedding模型计算语义相似度
- 我们将比较两种开源embedding模型

正在评估答案相关性...

使用英文优化Embedding模型评估 (all-MiniLM-L6-v2):
Evaluating: 100%|███████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:01<00:00,  1.54it/s]
相关性评分: 0.8565

使用中文优化Embedding模型评估 (bge-small-zh):
Evaluating: 100%|███████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:06<00:00,  2.11s/it]
相关性评分: 0.8926

=== Embedding模型比较 ===
英文模型评分: 0.8565
中文模型评分: 0.8926
差异: 0.0361 (中文模型更好)
'''