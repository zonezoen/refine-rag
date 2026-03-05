"""
混合方案：千问图片理解 + 本地 Embedding

方案说明：
1. 使用千问 VL API 理解图片内容（生成文字描述）
2. 使用本地 BGE 模型将描述转为向量
3. 实现图文检索功能

优势：
- 利用千问强大的图片理解能力
- 本地 embedding 免费无限制
- 可以实现图文检索

安装依赖：
pip install dashscope pillow sentence-transformers python-dotenv
"""

import os
from dotenv import load_dotenv
import dashscope
from dashscope import MultiModalConversation
from sentence_transformers import SentenceTransformer
import numpy as np

load_dotenv()

def describe_image_with_qwen(image_path):
    """
    使用千问 VL 理解图片内容
    
    Args:
        image_path: 图片路径
        
    Returns:
        str: 图片的文字描述
    """
    print(f"\n【步骤1】使用千问 VL 理解图片...")
    print(f"图片路径: {image_path}")
    
    # 设置 API Key
    dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
    
    if not dashscope.api_key:
        print("错误：未找到 DASHSCOPE_API_KEY")
        print("请在 .env 文件中设置：DASHSCOPE_API_KEY=your_api_key")
        return None
    
    # 构建消息
    messages = [
        {
            "role": "user",
            "content": [
                {"image": f"file://{os.path.abspath(image_path)}"},
                {"text": "请详细描述这张图片的内容，包括场景、人物、动作等。"}
            ]
        }
    ]
    
    try:
        # 调用千问 VL API
        response = MultiModalConversation.call(
            model='qwen-vl-plus',
            messages=messages
        )
        
        if response.status_code == 200:
            description = response.output.choices[0].message.content[0]["text"]
            print(f"图片描述: {description}")
            return description
        else:
            print(f"API 调用失败: {response.message}")
            return None
            
    except Exception as e:
        print(f"错误: {e}")
        return None


def generate_embedding(text, model):
    """
    使用本地模型生成文本 embedding
    
    Args:
        text: 文本内容
        model: SentenceTransformer 模型
        
    Returns:
        numpy.ndarray: 文本向量
    """
    print(f"\n【步骤2】生成文本 embedding...")
    embedding = model.encode(text, normalize_embeddings=True)
    print(f"向量维度: {embedding.shape}")
    return embedding


def search_similar(query_embedding, database_embeddings, database_texts, top_k=3):
    """
    在数据库中搜索最相似的内容
    
    Args:
        query_embedding: 查询向量
        database_embeddings: 数据库向量列表
        database_texts: 数据库文本列表
        top_k: 返回前 k 个结果
        
    Returns:
        list: 最相似的结果
    """
    print(f"\n【步骤3】搜索相似内容...")
    
    # 计算余弦相似度
    similarities = np.dot(database_embeddings, query_embedding)
    
    # 排序并返回 top-k
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for i, idx in enumerate(top_indices, 1):
        results.append({
            "rank": i,
            "text": database_texts[idx],
            "similarity": similarities[idx]
        })
    
    return results


def main():
    print("="*60)
    print("混合方案：千问图片理解 + 本地 Embedding")
    print("="*60)
    
    # ========== 1. 加载本地 Embedding 模型 ==========
    print("\n加载本地 BGE 模型...")
    embedding_model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
    print("模型加载完成！")
    
    # ========== 2. 准备图片数据库 ==========
    print("\n" + "="*60)
    print("准备图片数据库")
    print("="*60)
    
    # 模拟图片数据库（实际应用中，这些描述来自千问 VL）
    image_database = [
        "孙悟空手持金箍棒，正在与妖怪激烈战斗，场面宏大",
        "一只可爱的小猫在阳光下睡觉，表情安详",
        "美丽的山水风景，青山绿水，云雾缭绕",
        "现代化的城市夜景，高楼大厦灯火通明",
        "悟空使用火焰技能攻击敌人，火光四射"
    ]
    
    print("图片数据库（文字描述）:")
    for i, desc in enumerate(image_database, 1):
        print(f"  {i}. {desc}")
    
    # 生成数据库的 embeddings
    print("\n生成数据库 embeddings...")
    database_embeddings = embedding_model.encode(
        image_database,
        normalize_embeddings=True
    )
    print(f"数据库向量维度: {database_embeddings.shape}")
    
    # ========== 3. 测试场景1：用文本搜索图片 ==========
    print("\n" + "="*60)
    print("场景1：用文本搜索图片")
    print("="*60)
    
    text_query = "战斗场景"
    print(f"\n查询: {text_query}")
    
    # 生成查询向量
    query_embedding = embedding_model.encode(text_query, normalize_embeddings=True)
    
    # 搜索
    results = search_similar(query_embedding, database_embeddings, image_database)
    
    print("\n搜索结果:")
    for result in results:
        print(f"  {result['rank']}. {result['text']}")
        print(f"     相似度: {result['similarity']:.4f}")
    
    # ========== 4. 测试场景2：用图片搜索图片 ==========
    print("\n" + "="*60)
    print("场景2：用图片搜索图片（需要千问 API）")
    print("="*60)
    
    # 检查是否有 API Key
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("\n跳过此场景（需要 DASHSCOPE_API_KEY）")
        print("如果有千问 API Key，可以这样使用：")
        print("""
        # 1. 用千问理解查询图片
        query_description = describe_image_with_qwen("query_image.jpg")
        
        # 2. 生成查询向量
        query_embedding = embedding_model.encode(query_description)
        
        # 3. 搜索相似图片
        results = search_similar(query_embedding, database_embeddings, image_database)
        """)
    else:
        # 实际调用千问 API
        image_path = "../99-doc-data/多模态/query_image.jpg"
        
        if os.path.exists(image_path):
            # 用千问理解图片
            description = describe_image_with_qwen(image_path)
            print("图片描述002: 【【【"+description+"】】】")
            if description:
                # 生成向量并搜索
                query_embedding = generate_embedding(description, embedding_model)
                results = search_similar(query_embedding, database_embeddings, image_database)
                
                print("\n搜索结果:")
                for result in results:
                    print(f"  {result['rank']}. {result['text']}")
                    print(f"     相似度: {result['similarity']:.4f}")
        else:
            print(f"\n图片不存在: {image_path}")
    
    # ========== 5. 完整流程说明 ==========
    print("\n" + "="*60)
    print("完整流程说明")
    print("="*60)
    print("""
    【构建图片数据库】
    1. 收集所有图片
    2. 用千问 VL 为每张图片生成描述
    3. 用本地模型将描述转为向量
    4. 存储向量到向量数据库
    
    【用户查询】
    方式1 - 文本查询：
    1. 用户输入文本："战斗场景"
    2. 用本地模型生成查询向量
    3. 在向量数据库中搜索
    4. 返回最相似的图片
    
    方式2 - 图片查询：
    1. 用户上传图片
    2. 用千问 VL 理解图片内容
    3. 用本地模型生成查询向量
    4. 在向量数据库中搜索
    5. 返回最相似的图片
    
    【优势】
    ✅ 利用千问强大的图片理解能力
    ✅ 本地 embedding 免费无限制
    ✅ 可以实现图文检索
    ✅ 数据隐私（向量存储在本地）
    
    【成本】
    - 千问 VL：约 ¥0.008/千tokens
    - 本地 embedding：免费
    - 总成本：主要是千问 API 调用
    """)
    
    # ========== 6. 代码示例 ==========
    print("\n" + "="*60)
    print("实际应用代码示例")
    print("="*60)
    print("""
    # 完整的图片检索系统
    
    class ImageSearchSystem:
        def __init__(self):
            self.embedding_model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
            self.image_database = []
            self.embeddings = []
        
        def add_image(self, image_path):
            # 用千问理解图片
            description = describe_image_with_qwen(image_path)
            
            # 生成向量
            embedding = self.embedding_model.encode(description)
            
            # 存储
            self.image_database.append({
                'path': image_path,
                'description': description
            })
            self.embeddings.append(embedding)
        
        def search_by_text(self, query_text, top_k=5):
            # 生成查询向量
            query_embedding = self.embedding_model.encode(query_text)
            
            # 搜索
            similarities = np.dot(self.embeddings, query_embedding)
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            return [self.image_database[i] for i in top_indices]
        
        def search_by_image(self, query_image_path, top_k=5):
            # 用千问理解查询图片
            description = describe_image_with_qwen(query_image_path)
            
            # 用文本搜索
            return self.search_by_text(description, top_k)
    
    # 使用示例
    system = ImageSearchSystem()
    
    # 添加图片到数据库
    for img_path in image_paths:
        system.add_image(img_path)
    
    # 搜索
    results = system.search_by_text("战斗场景")
    results = system.search_by_image("query.jpg")
    """)

if __name__ == '__main__':
    main()
