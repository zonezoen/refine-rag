"""
真正的多模态嵌入 - Jina AI API

Jina AI 是什么？
- 专门为检索优化的多模态 embedding API
- 支持图片和文本的联合编码
- 国内可访问，无需科学上网
- 有免费额度

优势：
✅ 真正的多模态嵌入（图片和文本在同一向量空间）
✅ API 调用，无需下载模型
✅ 国内可访问
✅ 有免费额度（100万 tokens/月）
✅ 中文支持好

注册地址：
https://jina.ai/

安装依赖：
pip install requests pillow python-dotenv
"""

import os
import requests
import base64
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
import numpy as np

load_dotenv()

class JinaMultimodalEmbedding:
    """Jina AI 多模态嵌入客户端"""
    
    def __init__(self, api_key=None):
        """
        初始化 Jina AI 客户端
        
        Args:
            api_key: Jina AI API Key（如果不提供，从环境变量读取）
        """
        self.api_key = api_key or os.getenv("JINA_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "未找到 JINA_API_KEY\n"
                "请在 .env 文件中设置：JINA_API_KEY=your_api_key\n"
                "或访问 https://jina.ai/ 注册获取"
            )
        
        self.api_url = "https://api.jina.ai/v1/embeddings"
        self.model = "jina-clip-v1"  # 多模态模型
    
    def image_to_base64(self, image_path):
        """
        将图片转为 base64 编码
        
        Args:
            image_path: 图片路径
            
        Returns:
            str: base64 编码的图片
        """
        with Image.open(image_path) as img:
            # 调整图片大小（可选，减少传输大小）
            img.thumbnail((512, 512))
            
            # 转为 RGB（如果是 RGBA）
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            
            # 转为 base64
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            return f"data:image/jpeg;base64,{img_str}"
    
    def encode_text(self, text):
        """
        编码文本
        
        Args:
            text: 文本内容（字符串或列表）
            
        Returns:
            numpy.ndarray: 文本向量
        """
        if isinstance(text, str):
            text = [text]
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "input": [{"text": t} for t in text]
        }
        
        response = requests.post(self.api_url, headers=headers, json=data)
        
        if response.status_code == 200:
            result = response.json()
            embeddings = [item["embedding"] for item in result["data"]]
            return np.array(embeddings)
        else:
            raise Exception(f"API 调用失败: {response.status_code} - {response.text}")
    
    def encode_image(self, image_path):
        """
        编码图片
        
        Args:
            image_path: 图片路径（字符串或列表）
            
        Returns:
            numpy.ndarray: 图片向量
        """
        if isinstance(image_path, str):
            image_path = [image_path]
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # 转换图片为 base64
        images_base64 = [self.image_to_base64(path) for path in image_path]
        
        data = {
            "model": self.model,
            "input": [{"image": img} for img in images_base64]
        }
        
        response = requests.post(self.api_url, headers=headers, json=data)
        
        if response.status_code == 200:
            result = response.json()
            embeddings = [item["embedding"] for item in result["data"]]
            return np.array(embeddings)
        else:
            raise Exception(f"API 调用失败: {response.status_code} - {response.text}")
    
    def encode_multimodal(self, image_path, text):
        """
        编码图片+文本（多模态）
        
        Args:
            image_path: 图片路径
            text: 文本描述
            
        Returns:
            numpy.ndarray: 多模态向量
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        image_base64 = self.image_to_base64(image_path)
        
        data = {
            "model": self.model,
            "input": [
                {
                    "image": image_base64,
                    "text": text
                }
            ]
        }
        
        response = requests.post(self.api_url, headers=headers, json=data)
        
        if response.status_code == 200:
            result = response.json()
            embedding = result["data"][0]["embedding"]
            return np.array(embedding)
        else:
            raise Exception(f"API 调用失败: {response.status_code} - {response.text}")


def cosine_similarity(vec1, vec2):
    """计算余弦相似度"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def main():
    print("="*60)
    print("真正的多模态嵌入 - Jina AI")
    print("="*60)
    
    # ========== 1. 初始化客户端 ==========
    print("\n【步骤1】初始化 Jina AI 客户端...")
    
    try:
        client = JinaMultimodalEmbedding()
        print("✅ 客户端初始化成功！")
        print(f"   模型: {client.model}")
    except ValueError as e:
        print(f"❌ {e}")
        print("\n如何获取 API Key：")
        print("1. 访问 https://jina.ai/")
        print("2. 注册账号（支持 GitHub 登录）")
        print("3. 在 Dashboard 中创建 API Key")
        print("4. 在 .env 文件中设置：JINA_API_KEY=your_api_key")
        return
    
    # ========== 2. 准备测试数据 ==========
    print("\n【步骤2】准备测试数据...")
    
    # 图片路径
    image_path = "../99-doc-data/多模态/query_image.jpg"
    
    if not os.path.exists(image_path):
        print(f"⚠️  图片不存在: {image_path}")
        print("将使用文本演示...")
        image_path = None
    else:
        print(f"✅ 图片路径: {image_path}")
    
    # 测试文本
    texts = [
        "悟空在战斗",
        "孙悟空使用金箍棒",
        "一只猫在睡觉",
        "风景照片"
    ]
    
    print(f"\n测试文本:")
    for i, text in enumerate(texts, 1):
        print(f"  {i}. {text}")
    
    # ========== 3. 编码文本 ==========
    print("\n【步骤3】编码文本...")
    
    try:
        text_embeddings = client.encode_text(texts)
        print(f"✅ 文本编码完成！")
        print(f"   向量维度: {text_embeddings.shape[1]}")
        print(f"   文本数量: {text_embeddings.shape[0]}")
    except Exception as e:
        print(f"❌ 编码失败: {e}")
        return
    
    # ========== 4. 编码图片 ==========
    if image_path:
        print("\n【步骤4】编码图片...")
        
        try:
            image_embedding = client.encode_image(image_path)
            print(f"✅ 图片编码完成！")
            print(f"   向量维度: {image_embedding.shape[1]}")
            print(f"   向量范数: {np.linalg.norm(image_embedding):.4f}")
        except Exception as e:
            print(f"❌ 编码失败: {e}")
            image_embedding = None
    else:
        image_embedding = None
    
    # ========== 5. 计算图文相似度 ==========
    if image_embedding is not None:
        print("\n【步骤5】计算图文相似度...")
        
        print("\n图片与文本的相似度:")
        similarities = []
        for i, (text, text_emb) in enumerate(zip(texts, text_embeddings), 1):
            sim = cosine_similarity(image_embedding[0], text_emb)
            similarities.append(sim)
            print(f"  {i}. {text:20s} - 相似度: {sim:.4f}")
        
        # 找到最匹配的文本
        best_idx = np.argmax(similarities)
        print(f"\n✅ 最匹配的文本: {texts[best_idx]} (相似度: {similarities[best_idx]:.4f})")
    
    # ========== 6. 场景演示：文本搜索图片 ==========
    print("\n" + "="*60)
    print("场景演示：用文本搜索图片")
    print("="*60)
    
    # 模拟图片数据库
    image_database_texts = [
        "孙悟空手持金箍棒战斗",
        "小猫在睡觉",
        "山水风景",
        "城市夜景",
        "悟空使用火焰技能"
    ]
    
    print("\n图片数据库（用文本描述代替）:")
    for i, desc in enumerate(image_database_texts, 1):
        print(f"  {i}. {desc}")
    
    # 编码数据库
    print("\n编码图片数据库...")
    try:
        db_embeddings = client.encode_text(image_database_texts)
        print(f"✅ 数据库编码完成！")
    except Exception as e:
        print(f"❌ 编码失败: {e}")
        return
    
    # 用户查询
    query = "战斗场景"
    print(f"\n用户查询: {query}")
    
    try:
        query_embedding = client.encode_text(query)
        
        # 计算相似度
        similarities = np.dot(db_embeddings, query_embedding[0])
        
        # 排序
        top_indices = np.argsort(similarities)[::-1][:3]
        
        print("\n搜索结果:")
        for rank, idx in enumerate(top_indices, 1):
            print(f"  {rank}. {image_database_texts[idx]}")
            print(f"     相似度: {similarities[idx]:.4f}")
    except Exception as e:
        print(f"❌ 搜索失败: {e}")
    
    # ========== 7. 完整应用示例 ==========
    print("\n" + "="*60)
    print("完整应用示例")
    print("="*60)
    print("""
    # 图片搜索系统
    
    class ImageSearchSystem:
        def __init__(self, jina_api_key):
            self.client = JinaMultimodalEmbedding(jina_api_key)
            self.image_database = []
            self.embeddings = []
        
        def add_image(self, image_path, description=None):
            '''添加图片到数据库'''
            # 编码图片
            embedding = self.client.encode_image(image_path)
            
            self.image_database.append({
                'path': image_path,
                'description': description
            })
            self.embeddings.append(embedding[0])
        
        def search_by_text(self, query_text, top_k=5):
            '''用文本搜索图片'''
            # 编码查询文本
            query_embedding = self.client.encode_text(query_text)[0]
            
            # 计算相似度
            similarities = [
                cosine_similarity(query_embedding, emb)
                for emb in self.embeddings
            ]
            
            # 返回 top-k
            top_indices = np.argsort(similarities)[::-1][:top_k]
            return [self.image_database[i] for i in top_indices]
        
        def search_by_image(self, query_image_path, top_k=5):
            '''用图片搜索图片'''
            # 编码查询图片
            query_embedding = self.client.encode_image(query_image_path)[0]
            
            # 计算相似度
            similarities = [
                cosine_similarity(query_embedding, emb)
                for emb in self.embeddings
            ]
            
            # 返回 top-k
            top_indices = np.argsort(similarities)[::-1][:top_k]
            return [self.image_database[i] for i in top_indices]
    
    # 使用示例
    system = ImageSearchSystem(jina_api_key="your_api_key")
    
    # 添加图片
    system.add_image("image1.jpg", "悟空战斗")
    system.add_image("image2.jpg", "风景照片")
    
    # 搜索
    results = system.search_by_text("战斗场景")
    results = system.search_by_image("query.jpg")
    """)
    
    # ========== 8. 对比说明 ==========
    print("\n" + "="*60)
    print("Jina AI vs 其他方案")
    print("="*60)
    print("""
    | 方案 | 类型 | 优势 | 劣势 |
    |------|------|------|------|
    | Jina AI | 真多模态 | API调用、国内可访问 | 付费（有免费额度） |
    | CLIP本地 | 真多模态 | 免费、无限制 | 需要下载模型 |
    | 千问+BGE | 伪多模态 | 可解释性强 | 两步处理、精度稍低 |
    
    Jina AI 定价：
    - 免费额度：100万 tokens/月
    - 付费：$0.02/千次调用
    - 图片按分辨率计费
    
    推荐场景：
    ✅ 不想下载模型
    ✅ 需要真正的多模态嵌入
    ✅ 国内访问
    ✅ 有一定预算
    """)

if __name__ == '__main__':
    main()
