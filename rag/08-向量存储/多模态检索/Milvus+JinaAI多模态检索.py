"""
多模态图像检索系统：基于 Jina AI 和 Milvus 实现
功能：使用 Jina AI API 对图像和文本进行多模态编码，并在图像数据库中检索相似内容

优势：
✅ 无需下载模型，直接调用 API
✅ 真正的多模态嵌入（图像和文本在同一向量空间）
✅ 国内可访问，无需科学上网
✅ 有免费额度（100万 tokens/月）
✅ 中文支持好

安装依赖：
-----------
pip install pymilvus requests pillow python-dotenv opencv-python numpy tqdm

启动 Milvus：
-----------
cd rag/08-向量存储
docker-compose up -d

配置 API Key：
-----------
1. 访问 https://jina.ai/ 注册账号
2. 在 Dashboard 中创建 API Key
3. 在项目根目录的 .env 文件中添加：
   JINA_API_KEY=your_api_key_here

注意事项：
-----------
⚠️ Jina AI 免费额度：100万 tokens/月
⚠️ 如果提示余额不足，需要：
   - 等待下个月重置
   - 或者充值（$0.02/千次调用）
   - 或者使用其他 API Key
"""

# ==================== 1. 导入必要的库 ====================
import os
import requests
import base64
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
import numpy as np
import json
from dataclasses import dataclass
from typing import List
from tqdm import tqdm
import cv2
from pymilvus import MilvusClient

# 加载环境变量
load_dotenv()

# ==================== 2. Jina AI 多模态编码器类 ====================
class JinaMultimodalEncoder:
    """
    Jina AI 多模态编码器
    
    使用 Jina AI 的 CLIP 模型将图像和文本编码到同一向量空间
    支持：
    1. 纯文本编码
    2. 纯图像编码
    3. 图像+文本组合编码
    """
    
    def __init__(self, api_key: str = None):
        """
        初始化 Jina AI 客户端
        
        参数:
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
        
        print(f"✅ Jina AI 客户端初始化成功")
        print(f"   模型: {self.model}")
    
    def _image_to_base64(self, image_path: str) -> str:
        """
        将图片转为 base64 编码
        
        参数:
            image_path: 图片路径
            
        返回:
            base64 编码的图片字符串
        """
        with Image.open(image_path) as img:
            # 调整图片大小以减少传输大小
            img.thumbnail((512, 512))
            
            # 转为 RGB（如果是 RGBA）
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            
            # 转为 base64
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            return f"data:image/jpeg;base64,{img_str}"
    
    def encode_image(self, image_path: str) -> List[float]:
        """
        编码图像
        
        参数:
            image_path: 图像文件路径
            
        返回:
            图像的向量表示
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # 转换图片为 base64
        image_base64 = self._image_to_base64(image_path)
        
        data = {
            "model": self.model,
            "input": [{"image": image_base64}]
        }
        
        response = requests.post(self.api_url, headers=headers, json=data)
        
        if response.status_code == 200:
            result = response.json()
            embedding = result["data"][0]["embedding"]
            return embedding
        else:
            raise Exception(f"API 调用失败: {response.status_code} - {response.text}")
    
    def encode_text(self, text: str) -> List[float]:
        """
        编码文本
        
        参数:
            text: 文本内容
            
        返回:
            文本的向量表示
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "input": [{"text": text}]
        }
        
        response = requests.post(self.api_url, headers=headers, json=data)
        
        if response.status_code == 200:
            result = response.json()
            embedding = result["data"][0]["embedding"]
            return embedding
        else:
            raise Exception(f"API 调用失败: {response.status_code} - {response.text}")
    
    def encode_query(self, image_path: str, text: str) -> List[float]:
        """
        编码图像和文本的组合查询
        
        参数:
            image_path: 图像文件路径
            text: 文本描述
            
        返回:
            多模态向量表示
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # 转换图片为 base64
        image_base64 = self._image_to_base64(image_path)
        
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
            return embedding
        else:
            raise Exception(f"API 调用失败: {response.status_code} - {response.text}")

# ==================== 3. 数据集管理 ====================
@dataclass
class WukongImage:
    """
    图像元数据结构
    
    存储每张图像的详细信息，用于检索时的过滤和展示
    """
    image_id: str
    file_path: str
    title: str
    category: str
    description: str
    tags: List[str]
    game_chapter: str
    location: str
    characters: List[str]
    abilities_shown: List[str]
    environment: str
    time_of_day: str

class WukongDataset:
    """
    图像数据集管理类
    
    负责加载和管理图像数据集
    """
    
    def __init__(self, data_dir: str, metadata_path: str):
        """
        初始化数据集
        
        参数:
            data_dir: 图像文件所在目录
            metadata_path: 元数据JSON文件路径
        """
        self.data_dir = data_dir
        self.metadata_path = metadata_path
        self.images: List[WukongImage] = []
        self._load_metadata()
    
    def _load_metadata(self):
        """从JSON文件加载图像元数据"""
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for img_data in data['images']:
                # 修正文件路径
                img_data['file_path'] = f"{self.data_dir}/{img_data['file_path'].split('/')[-1]}"
                self.images.append(WukongImage(**img_data))
        
        print(f"✅ 加载了 {len(self.images)} 张图片的元数据")

# ==================== 4. 主程序 ====================
def main():
    print("="*70)
    print("多模态图像检索系统 - Jina AI + Milvus")
    print("="*70)
    
    # ========== 步骤1: 初始化编码器 ==========
    print("\n【步骤1】初始化 Jina AI 编码器...")
    try:
        encoder = JinaMultimodalEncoder()
    except ValueError as e:
        print(f"❌ {e}")
        return
    
    # ========== 步骤2: 加载数据集 ==========
    print("\n【步骤2】加载图像数据集...")
    dataset = WukongDataset(
        data_dir="../../99-doc-data/多模态",
        metadata_path="../../99-doc-data/多模态/metadata.json"
    )
    
    # ========== 步骤3: 生成图像嵌入向量 ==========
    print("\n【步骤3】生成图像嵌入向量...")
    print("⚠️  注意：这将调用 Jina AI API，可能需要一些时间")
    
    image_dict = {}
    failed_images = []
    
    for image in tqdm(dataset.images, desc="编码图片"):
        try:
            embedding = encoder.encode_image(image.file_path)
            image_dict[image.file_path] = embedding
        except Exception as e:
            print(f"\n⚠️  处理图片 {image.file_path} 失败：{str(e)}")
            failed_images.append(image.file_path)
            continue
    
    print(f"\n✅ 成功编码 {len(image_dict)} 张图片")
    if failed_images:
        print(f"⚠️  失败 {len(failed_images)} 张图片")
    
    # 检查是否有成功编码的图片
    if len(image_dict) == 0:
        print("\n❌ 没有成功编码的图片，无法继续")
        print("\n可能的原因：")
        print("1. Jina AI 余额不足 - 访问 https://jina.ai/api-dashboard/key-manager 充值")
        print("2. API Key 无效 - 检查 .env 文件中的 JINA_API_KEY")
        print("3. 网络问题 - 检查网络连接")
        print("4. 图片路径错误 - 检查图片文件是否存在")
        return
    
    # ========== 步骤4: 创建 Milvus 向量数据库 ==========
    print("\n【步骤4】创建 Milvus 向量数据库...")
    
    collection_name = "wukong_scenes_jina"
    
    # 连接到 Docker Milvus
    try:
        milvus_client = MilvusClient(uri="http://localhost:19530")
        print("✅ 已连接到 Docker Milvus (localhost:19530)")
    except Exception as e:
        print(f"\n❌ 无法连接到 Docker Milvus: {e}")
        print("\n请确保 Milvus 服务已启动：")
        print("  cd rag/08-向量存储")
        print("  docker-compose up -d")
        print("\n检查服务状态：")
        print("  docker-compose ps")
        return
    
    # 获取向量维度
    dim = len(list(image_dict.values())[0])
    print(f"   向量维度: {dim}")
    
    # 如果集合已存在，先删除
    if milvus_client.has_collection(collection_name):
        milvus_client.drop_collection(collection_name)
        print(f"   已删除旧集合: {collection_name}")
    
    # 创建新集合
    milvus_client.create_collection(
        collection_name=collection_name,
        dimension=dim,
        auto_id=True,
        enable_dynamic_field=True
    )
    print(f"✅ 创建集合成功: {collection_name}")
    
    # ========== 步骤5: 插入数据到 Milvus ==========
    print("\n【步骤5】插入数据到 Milvus...")
    
    insert_data = []
    for image in dataset.images:
        if image.file_path in image_dict:
            insert_data.append({
                "image_path": image.file_path,
                "vector": image_dict[image.file_path],
                "title": image.title,
                "category": image.category,
                "description": image.description,
                "tags": ",".join(image.tags),
                "game_chapter": image.game_chapter,
                "location": image.location,
                "characters": ",".join(image.characters),
                "abilities": ",".join(image.abilities_shown),
                "environment": image.environment,
                "time_of_day": image.time_of_day
            })
    
    result = milvus_client.insert(
        collection_name=collection_name,
        data=insert_data
    )
    
    print(f"✅ 索引构建完成，共插入 {result['insert_count']} 条记录")
    
    # 加载集合到内存（重要！）
    print("\n   加载集合到内存...")
    milvus_client.load_collection(collection_name=collection_name)
    print("   ✅ 集合已加载")
    
    # ========== 步骤6: 执行多模态检索 ==========
    print("\n【步骤6】执行多模态检索...")
    
    query_image = "../../99-doc-data/多模态/query_image.jpg"
    query_text = "寻找悟空面对建筑物战斗场景"
    
    print(f"   查询图像: {query_image}")
    print(f"   查询文本: {query_text}")
    
    # 编码查询
    print("\n   正在编码查询...")
    query_vec = encoder.encode_query(query_image, query_text)
    
    # 搜索参数
    search_params = {
        "metric_type": "COSINE",
        "params": {
            "nprobe": 10,
            "radius": 0.1,
            "range_filter": 0.8
        }
    }

    # 执行搜索
    results = milvus_client.search(
        collection_name=collection_name,
        data=[query_vec],
        output_fields=[
            "image_path", "title", "category", "description",
            "tags", "game_chapter", "location", "characters",
            "abilities", "environment", "time_of_day"
        ],
        limit=9,
        # search_params=search_params
    )[0]
    
    # ========== 步骤7: 显示检索结果 ==========
    print("\n【步骤7】检索结果:")
    print("="*70)
    
    if len(results) == 0:
        print("\n⚠️  没有找到匹配的结果")
        print("\n可能的原因：")
        print("1. 查询向量与数据库向量差异太大")
        print("2. 集合未正确加载到内存")
        print("3. 向量维度不匹配")
        print("\n尝试查看数据库中的数据：")
        print(f"   集合中的记录数: {milvus_client.num_entities(collection_name)}")
    else:
        for idx, result in enumerate(results, 1):
            print(f"\n结果 {idx}:")
            print(f"  图片: {result['entity']['image_path']}")
            print(f"  标题: {result['entity']['title']}")
            print(f"  描述: {result['entity']['description']}")
            print(f"  相似度分数: {result['distance']:.4f}")
    
    # ========== 步骤8: 可视化结果 ==========
    print("\n【步骤8】生成可视化结果...")
    visualize_results(query_image, results, "search_results_jina.jpg")
    print("✅ 可视化结果已保存: search_results_jina.jpg")
    
    print("\n" + "="*70)
    print("检索完成！")
    print("="*70)

# ==================== 5. 可视化函数 ====================
def visualize_results(query_image: str, results: List[dict], output_path: str):
    """
    可视化搜索结果
    
    将查询图像和检索结果组合成一个网格图像
    """
    img_size = (300, 300)
    grid_size = (3, 3)
    
    # 创建画布
    canvas_height = img_size[0] * (grid_size[0] + 1)
    canvas_width = img_size[1] * (grid_size[1] + 1)
    canvas = np.full((canvas_height, canvas_width, 3), 255, dtype=np.uint8)
    
    # 添加查询图片（左上角，红色边框）
    query_img = Image.open(query_image).convert("RGB")
    query_array = np.array(query_img)
    query_resized = cv2.resize(query_array, (img_size[0] - 20, img_size[1] - 20))
    bordered_query = cv2.copyMakeBorder(
        query_resized, 10, 10, 10, 10,
        cv2.BORDER_CONSTANT,
        value=(255, 0, 0)  # 红色边框
    )
    canvas[:img_size[0], :img_size[1]] = bordered_query
    
    # 添加检索结果
    for idx, result in enumerate(results[:grid_size[0] * grid_size[1]]):
        row = (idx // grid_size[1]) + 1
        col = idx % grid_size[1]
        
        img = Image.open(result["entity"]["image_path"]).convert("RGB")
        img_array = np.array(img)
        resized = cv2.resize(img_array, (img_size[0], img_size[1]))
        
        y_start = row * img_size[0]
        x_start = col * img_size[1]
        
        canvas[y_start:y_start + img_size[0], x_start:x_start + img_size[1]] = resized
        
        # 添加相似度分数
        score_text = f"Score: {result['distance']:.2f}"
        cv2.putText(
            canvas,
            score_text,
            (x_start + 10, y_start + img_size[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1
        )
    
    cv2.imwrite(output_path, canvas)

if __name__ == "__main__":
    main()
