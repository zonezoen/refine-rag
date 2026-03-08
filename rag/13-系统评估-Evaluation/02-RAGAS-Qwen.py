import os
from dotenv import load_dotenv
load_dotenv() # 加载.env文件中的环境变量
import numpy as np
from datasets import Dataset
# 使用新的导入方式，避免 deprecation 警告
from ragas.metrics.collections import Faithfulness, AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate

# 准备评估用的LLM（使用阿里巴巴千问）
# 直接使用 ChatTongyi，测试是否原生支持 RAGAS
llm = LangchainLLMWrapper(ChatTongyi(
    model_name="qwen-max",  # 使用千问最强模型
    temperature=0,
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
))

# 准备数据集
# 这个数据集包含了问题、生成的答案以及相关的上下文信息
data = {
    "question": [
        "《黑神话：悟空》的主角是谁？",
        "《黑神话：悟空》的战斗系统有什么特色？",
        "《黑神话：悟空》的画面质量如何？",
    ],
    "answer": [
        "《黑神话：悟空》的主角是孙悟空，基于中国古典名著《西游记》，但进行了全新的诠释。这个版本的孙悟空更加成熟和沉稳，展现出与传统形象不同的性格特点。",
        "《黑神话：悟空》的战斗系统融合了中国武术与魂类游戏特色，包括轻重攻击组合、法术变化和法术系统。值得注意的是，悟空可以在战斗中变换不同的武器形态，如标志性的金箍棒和双节棍，并使用各种神通法术。",
        "《黑神话：悟空》采用虚幻引擎5开发，展现出惊艳的画面质量。游戏的场景建模、光影效果和角色细节都达到了顶级水准，特别是对中国传统建筑和神话场景的细致还原。",
    ],
    "contexts": [
        [
            "《黑神话：悟空》是游戏科学开发的动作角色扮演游戏，以《西游记》中的孙悟空为主角，但进行了创新性的诠释。游戏中的悟空性格更加沉稳，肩负着特殊的使命。",
            "游戏设定在神话世界中，讲述了一个全新的故事，呈现出与传统孙悟空形象不同的角色塑造。"
        ],
        [
            "游戏的战斗系统深受魂类游戏影响，同时融入了中国传统武术元素。玩家可以使用不同的武器形态，包括标志性的金箍棒和其他变化武器。",
            "战斗中玩家可以施展各种神通法术，配合轻重攻击和连招系统，打造流畅而独特的战斗体验。游戏还具有独特的变身系统。"
        ],
        [
            "《黑神话：悟空》展现出卓越的视觉质量，采用虚幻引擎5打造，实现了极高的画面保真度。游戏的环境和角色模型都经过精心制作。",
            "光影效果、材质渲染和环境细节都达到了3A级标准，完美捕捉了东方神话世界的氛围。"
        ]
    ]
}

# 将字典转换为Hugging Face的Dataset对象，方便Ragas处理
dataset = Dataset.from_dict(data)

print("\n=== Ragas评估指标说明（千问版本）===")
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
1. 使用阿里巴巴千问（Tongyi Qwen）替代 DeepSeek
2. 使用 ChatTongyi 从 langchain_community.chat_models.tongyi 导入
3. 直接使用 ChatTongyi，无需自定义包装器（千问原生支持 RAGAS）
4. 使用中文测试数据（《黑神话：悟空》相关内容）
5. 使用两种开源HuggingFace模型进行对比：
   - all-MiniLM-L6-v2: 英文优化的轻量级模型
   - bge-small-zh: 中文优化的BGE模型
6. 所有API调用都使用项目中已配置的千问密钥

技术要点：
- ChatTongyi 是 LangChain 为阿里云千问提供的官方集成
- 千问 API 原生支持 RAGAS 所需的参数，无需额外包装
- LangchainEmbeddingsWrapper 将 LangChain 嵌入模型适配到 RAGAS 接口
- 代码更简洁，性能更好

环境要求：
- 需要安装: pip install dashscope langchain-community langchain-huggingface ragas datasets
- 需要在 .env 文件中配置 DASHSCOPE_API_KEY

预期输出示例：
=== Ragas评估指标说明（千问版本）===

1. Faithfulness（忠实度）
- 评估生成的答案是否忠实于上下文内容
- 通过将答案分解为简单陈述，然后验证每个陈述是否可以从上下文中推断得出
- 该指标仅依赖LLM，不需要embedding模型

正在评估忠实度...
Evaluating: 100%|███████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:05<00:00,  1.87s/it]
忠实度评分: 0.8552

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
相关性评分: 0.9126

=== Embedding模型比较 ===
英文模型评分: 0.8565
中文模型评分: 0.9126
差异: 0.0561 (中文模型更好)
'''
