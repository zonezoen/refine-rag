"""
多模态图像检索系统：基于Visualized-BGE和Milvus实现
功能：对图像和文本进行多模态编码，并在图像数据库中检索相似内容

这个程序展示了如何构建一个完整的多模态检索系统：
1. 使用Visual-BGE模型将图像和文本编码为向量
2. 使用Milvus向量数据库存储和检索图像向量
3. 支持图像+文本的组合查询
4. 可视化检索结果

安装依赖：
-----------
1. 克隆并安装 FlagEmbedding:
   git clone https://github.com/FlagOpen/FlagEmbedding.git
   cd FlagEmbedding/research/visual_bge
   pip install -e .

2. 安装其他依赖:
   pip install torchvision timm einops ftfy pymilvus

3. 下载模型权重:
   # 多语言模型（支持中文）
   huggingface-cli download BAAI/bge-visualized-m3 --local-dir ./models/bge-visualized-m3
   
   # 或手动从 https://huggingface.co/BAAI/bge-visualized-m3 下载

详细安装说明请查看：安装说明.md
"""

# ==================== 1. 导入必要的库 ====================
import torch  # PyTorch深度学习框架
from visual_bge.modeling import Visualized_BGE  # 多模态编码模型
from dataclasses import dataclass  # 用于创建数据类
from typing import List, Optional  # 类型提示
import json  # JSON文件处理
from tqdm import tqdm  # 进度条显示
import numpy as np  # 数值计算
import cv2  # OpenCV图像处理
from PIL import Image  # Python图像库
from pymilvus import MilvusClient  # Milvus向量数据库客户端

# ==================== 2. 多模态编码器类定义 ====================
class WukongEncoder:
    """
    多模态编码器：将图像和文本编码成向量
    
    这个类封装了Visual-BGE模型，可以：
    1. 将图像编码为向量
    2. 将图像+文本组合编码为向量
    3. 支持多模态检索
    """
    def __init__(self, model_name: str, model_path: str):
        """
        初始化编码器
        参数:
            model_name: 基础BGE模型名称（如BAAI/bge-m3）
            model_path: Visual-BGE模型权重文件路径
        """
        # 加载Visual-BGE模型，这是一个多模态编码模型
        self.model = Visualized_BGE(model_name_bge=model_name, model_weight=model_path)
        # 设置为评估模式，不进行训练
        self.model.eval()
    
    def encode_query(self, image_path: str, text: str) -> list[float]:
        """
        编码图像和文本的组合查询
        参数:
            image_path: 图像文件路径
            text: 文本描述
        返回:
            向量表示（浮点数列表）
        """
        # 使用torch.no_grad()禁用梯度计算，节省内存和计算
        with torch.no_grad():
            # 同时编码图像和文本，生成统一的向量表示
            query_emb = self.model.encode(image=image_path, text=text)
        # 将张量转换为Python列表，取第一个元素（批次维度）
        return query_emb.tolist()[0]
    
    def encode_image(self, image_path: str) -> list[float]:
        """
        仅编码图像（不包含文本）
        参数:
            image_path: 图像文件路径
        返回:
            图像的向量表示
        """
        with torch.no_grad():
            # 只编码图像，不包含文本信息
            query_emb = self.model.encode(image=image_path)
        return query_emb.tolist()[0]

# 初始化编码器实例
model_name = "BAAI/bge-m3"  # 使用BGE-M3作为基础模型（支持中文）
model_path = "./Visualized_m3.pth"  # Visual-BGE的预训练权重文件
encoder = WukongEncoder(model_name, model_path)

# ==================== 3. 数据集管理 ====================
@dataclass
class WukongImage:
    """
    图像元数据结构
    
    使用@dataclass装饰器自动生成__init__、__repr__等方法
    存储每张图像的详细信息，用于检索时的过滤和展示
    """
    image_id: str          # 图像唯一标识符
    file_path: str         # 图像文件路径
    title: str             # 图像标题
    category: str          # 图像类别（如：战斗、探索、剧情等）
    description: str       # 图像详细描述
    tags: List[str]        # 标签列表（如：武器、技能、场景等）
    game_chapter: str      # 游戏章节
    location: str          # 地点信息
    characters: List[str]  # 出现的角色列表
    abilities_shown: List[str]  # 展示的技能列表
    environment: str       # 环境类型（如：雪地、森林、建筑等）
    time_of_day: str      # 时间（如：白天、夜晚、黄昏等）

class WukongDataset:
    """
    图像数据集管理类
    
    负责加载和管理图像数据集，包括：
    1. 从JSON文件加载元数据
    2. 管理图像文件路径
    3. 提供数据访问接口
    """
    def __init__(self, data_dir: str, metadata_path: str):
        """
        初始化数据集
        参数:
            data_dir: 图像文件所在目录
            metadata_path: 元数据JSON文件路径
        """
        self.data_dir = data_dir           # 图像文件目录
        self.metadata_path = metadata_path # 元数据文件路径
        self.images: List[WukongImage] = [] # 存储所有图像对象的列表
        self._load_metadata()              # 加载元数据
    
    def _load_metadata(self):
        """
        从JSON文件加载图像元数据
        
        JSON文件格式应该包含images数组，每个元素包含图像的所有属性
        """
        # 以UTF-8编码打开JSON文件（支持中文）
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)  # 解析JSON数据
            # 遍历JSON中的每个图像数据
            for img_data in data['images']:
                # 修正文件路径：确保路径相对于data_dir
                # 从完整路径中提取文件名，然后拼接到data_dir
                img_data['file_path'] = f"{self.data_dir}/{img_data['file_path'].split('/')[-1]}"
                # 使用**操作符将字典解包为关键字参数，创建WukongImage对象
                self.images.append(WukongImage(**img_data))

# 初始化数据集实例
# 指定图像文件目录和元数据文件路径
dataset = WukongDataset("90-文档-Data/多模态", "90-文档-Data/多模态/metadata.json")

# ==================== 4. 生成图像嵌入向量 ====================
# 为数据集中的所有图像生成向量表示
image_dict = {}  # 存储图像路径到向量的映射

# 使用tqdm显示进度条，遍历所有图像
for image in tqdm(dataset.images, desc="生成图片嵌入"):
    try:
        # 调用编码器将图像转换为向量
        # 这里只编码图像，不包含文本信息
        image_dict[image.file_path] = encoder.encode_image(image.file_path)
    except Exception as e:
        # 如果某张图片处理失败（如文件损坏、格式不支持等），跳过并继续
        print(f"处理图片 {image.file_path} 失败：{str(e)}")
        continue

# 输出成功编码的图片数量
print(f"成功编码 {len(image_dict)} 张图片")

# ==================== 5. Milvus向量数据库设置 ====================
# 定义集合名称（类似于数据库中的表名）
collection_name = "wukong_scenes"

# 创建Milvus客户端连接
# uri指定数据库文件路径，这里使用本地SQLite文件存储
milvus_client = MilvusClient(uri="./wukong_images.db")

# 获取向量维度（从第一个向量中获取长度）
dim = len(list(image_dict.values())[0])

# 创建向量集合（Collection）
milvus_client.create_collection(
    collection_name=collection_name,  # 集合名称
    dimension=dim,                    # 向量维度
    auto_id=True,                    # 自动生成ID
    enable_dynamic_field=True        # 允许动态字段（可以存储额外的元数据）
)

# 准备插入数据
insert_data = []  # 存储要插入的数据列表

# 遍历数据集中的每张图像
for image in dataset.images:
    # 只处理成功编码的图像
    if image.file_path in image_dict:
        # 构建要插入的数据记录
        insert_data.append({
            "image_path": image.file_path,                    # 图像路径
            "vector": image_dict[image.file_path],           # 图像向量
            "title": image.title,                            # 图像标题
            "category": image.category,                      # 类别
            "description": image.description,                # 描述
            "tags": ",".join(image.tags),                   # 标签（转为逗号分隔的字符串）
            "game_chapter": image.game_chapter,             # 游戏章节
            "location": image.location,                      # 地点
            "characters": ",".join(image.characters),       # 角色列表
            "abilities": ",".join(image.abilities_shown),   # 技能列表
            "environment": image.environment,                # 环境
            "time_of_day": image.time_of_day                # 时间
        })

# 批量插入数据到Milvus
result = milvus_client.insert(
    collection_name=collection_name,  # 目标集合
    data=insert_data                  # 要插入的数据
)

# 输出插入结果
print(f"索引构建完成，共插入 {result['insert_count']} 条记录")

# ==================== 6. 搜索功能实现 ====================
def search_similar_images(
    query_image: str,
    query_text: str,
    limit: int = 9
) -> List[dict]:
    """
    搜索相似图像函数
    
    这个函数实现了多模态检索的核心功能：
    1. 将查询图像和文本编码为向量
    2. 在Milvus中执行向量相似度搜索
    3. 返回最相似的图像及其元数据
    
    参数:
        query_image: 查询图像的文件路径
        query_text: 查询文本描述
        limit: 返回结果的数量（默认9个）
    返回:
        检索结果列表，每个元素包含图像信息和相似度分数
    """
    # 使用编码器将查询图像和文本编码为统一的向量表示
    # 这是多模态检索的关键：图像和文本被映射到同一个向量空间
    query_vec = encoder.encode_query(query_image, query_text)
    
    # 构建搜索参数
    search_params = {
        "metric_type": "COSINE",  # 使用余弦相似度作为距离度量
        "params": {
            "nprobe": 10,         # 搜索的聚类数量（影响搜索精度和速度）
            "radius": 0.1,        # 搜索半径（最小相似度阈值）
            "range_filter": 0.8   # 范围过滤器（最大相似度阈值）
        }
    }

    # 在Milvus中执行向量搜索
    results = milvus_client.search(
        collection_name=collection_name,  # 搜索的集合名称
        data=[query_vec],                 # 查询向量（列表格式，支持批量查询）
        output_fields=[                   # 需要返回的字段
            "image_path", "title", "category", "description",
            "tags", "game_chapter", "location", "characters",
            "abilities", "environment", "time_of_day"
        ],
        limit=limit,                      # 返回结果数量
        search_params=search_params       # 搜索参数
    )[0]  # 取第一个查询的结果（因为我们只有一个查询向量）
    
    return results

# ==================== 7. 可视化函数 ====================
def visualize_results(query_image: str, results: List[dict], output_path: str):
    """
    可视化搜索结果函数
    
    将查询图像和检索结果组合成一个网格图像，便于直观查看检索效果
    
    参数:
        query_image: 查询图像的文件路径
        results: 搜索结果列表（来自search_similar_images函数）
        output_path: 输出图像的保存路径
    """
    # 设置每张图片的显示大小和网格布局
    img_size = (300, 300)  # 每张图片调整后的大小（宽，高）
    grid_size = (3, 3)     # 网格布局：3x3显示检索结果
    
    # 创建画布（背景图像）
    # 高度 = 图片高度 × (网格行数 + 1)，+1是为了放置查询图片
    canvas_height = img_size[0] * (grid_size[0] + 1)
    # 宽度 = 图片宽度 × 网格列数
    canvas_width = img_size[1] * (grid_size[1] + 1)
    # 创建白色背景的画布，dtype=np.uint8表示像素值范围0-255
    canvas = np.full((canvas_height, canvas_width, 3), 255, dtype=np.uint8)
    
    # 在画布左上角添加查询图片
    query_img = Image.open(query_image).convert("RGB")  # 打开并转换为RGB格式
    query_array = np.array(query_img)                   # 转换为numpy数组
    # 调整查询图片大小，留出边框空间
    query_resized = cv2.resize(query_array, (img_size[0] - 20, img_size[1] - 20))
    # 添加红色边框，标识这是查询图片
    bordered_query = cv2.copyMakeBorder(
        query_resized, 10, 10, 10, 10,  # 上下左右各10像素边框
        cv2.BORDER_CONSTANT,            # 常数边框类型
        value=(255, 0, 0)              # 红色边框 (BGR格式)
    )
    # 将带边框的查询图片放置在画布左上角
    canvas[:img_size[0], :img_size[1]] = bordered_query
    
    # 在网格中添加检索结果图片
    # 只显示前grid_size[0] * grid_size[1]个结果（最多9个）
    for idx, result in enumerate(results[:grid_size[0] * grid_size[1]]):
        # 计算当前图片在网格中的位置
        row = (idx // grid_size[1]) + 1  # 行号，+1是因为第0行放查询图片
        col = idx % grid_size[1]         # 列号
        
        # 加载并处理结果图片
        img = Image.open(result["entity"]["image_path"]).convert("RGB")
        img_array = np.array(img)
        # 调整图片大小以适应网格
        resized = cv2.resize(img_array, (img_size[0], img_size[1]))
        
        # 计算图片在画布中的位置
        y_start = row * img_size[0]  # 起始Y坐标
        x_start = col * img_size[1]  # 起始X坐标
        
        # 将图片放置到画布的指定位置
        canvas[y_start:y_start + img_size[0], x_start:x_start + img_size[1]] = resized
        
        # 在图片上添加相似度分数文本
        score_text = f"Score: {result['distance']:.2f}"  # 格式化分数，保留2位小数
        cv2.putText(
            canvas,                                    # 目标图像
            score_text,                               # 要添加的文本
            (x_start + 10, y_start + img_size[0] - 10), # 文本位置（左下角）
            cv2.FONT_HERSHEY_SIMPLEX,                 # 字体类型
            0.5,                                      # 字体大小
            (0, 0, 0),                               # 文本颜色（黑色）
            1                                        # 线条粗细
        )
    
    # 保存最终的可视化结果图像
    cv2.imwrite(output_path, canvas)

# ==================== 8. 执行查询示例 ====================
# 这部分展示如何使用构建好的多模态检索系统

# 定义查询参数
query_image = "../../90-文档-Data/多模态/query_image.jpg"  # 查询图像路径
query_text = "寻找悟空面对建筑物战斗场景"              # 查询文本描述

# 执行多模态检索
# 这里同时使用图像和文本作为查询条件
# 系统会找到在视觉和语义上都相似的图像
results = search_similar_images(
    query_image=query_image,  # 输入查询图像
    query_text=query_text,    # 输入查询文本
    limit=9                   # 返回最相似的9个结果
)

# 在控制台输出详细的检索结果信息
print("\n搜索结果:")
for idx, result in enumerate(results):
    print(f"\n结果 {idx + 1}:")  # 结果编号（从1开始）
    # 输出图像路径
    print(f"图片：{result['entity']['image_path']}")
    # 输出图像标题
    print(f"标题：{result['entity']['title']}")
    # 输出图像描述
    print(f"描述：{result['entity']['description']}")
    # 输出相似度分数（越小表示越相似）
    print(f"相似度分数：{result['distance']:.4f}")

# 生成可视化结果图像
# 将查询图像和检索结果组合成一张图片，保存为"search_results.jpg"
visualize_results(query_image, results, "search_results.jpg")

# ==================== 程序总结 ====================
"""
这个多模态检索系统的完整流程：

1. 数据准备：
   - 加载图像数据集和元数据
   - 使用Visual-BGE模型为所有图像生成向量表示

2. 索引构建：
   - 将图像向量和元数据存储到Milvus向量数据库
   - 建立高效的向量索引用于快速检索

3. 多模态查询：
   - 接受图像+文本的组合查询
   - 将查询编码为向量并在数据库中搜索相似向量
   - 返回最相似的图像及其元数据

4. 结果展示：
   - 在控制台输出详细信息
   - 生成可视化图像展示检索效果

应用场景：
- 游戏资产管理：通过描述和参考图找到相似的游戏场景
- 图像搜索引擎：支持文本+图像的复合查询
- 内容推荐系统：基于用户偏好推荐相似内容
- 设计素材库：帮助设计师快速找到灵感素材
"""
