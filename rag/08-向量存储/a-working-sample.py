"""
Milvus 完整示例：黑神话悟空妖怪数据库

功能：
1. 创建 Collection 并定义 Schema
2. 使用 sentence-transformers 生成向量
3. 插入数据到 Milvus
4. 向量相似度搜索
5. 条件过滤查询

依赖：
pip install pymilvus sentence-transformers pandas tqdm

注意：
- 需要先启动 Milvus 服务（docker compose up -d）
- 使用 sentence-transformers 而不是 pymilvus.model（兼容性更好）
"""

import logging
# 准备示例数据集
import pandas as pd

# 隐藏 pymilvus 的 ERROR 日志
logging.getLogger("pymilvus").setLevel(logging.CRITICAL)
data_records = [
    {
        "monster_id": "BM001",
        "monster_name": "虎先锋",
        "location": "竹林关隘",
        "difficulty": "High",
        "synonyms": "猛虎妖, 虎妖",
        "description": "在竹林关卡中出现的猛虎型妖怪，力量强大。"
    },
    {
        "monster_id": "BM002",
        "monster_name": "火猿",
        "location": "火山洞窟",
        "difficulty": "Low",
        "synonyms": "烈焰猿, 炎猿",
        "description": "生活在火山洞窟的猿类妖怪，只是插科打诨的小兵。"
    },]
df = pd.DataFrame(data_records)

# 建立/连接Milvus
from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema
# 使用 sentence-transformers 直接生成向量
from sentence_transformers import SentenceTransformer

# 连接到 Docker 中的 Milvus 服务器
client = MilvusClient(uri="http://localhost:19530")
collection_name = "Wukong_Monsters"

# 获取嵌入模型的向量维度
print("加载 embedding 模型...")
embedding_model = SentenceTransformer('BAAI/bge-small-zh-v1.5')  # 使用小模型，更快
sample_embedding = embedding_model.encode(["示例文本"])[0]
vector_dim = len(sample_embedding)
print(f"向量维度: {vector_dim}")

# 定义集合模式并创建集合
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_dim),
    FieldSchema(name="monster_id", dtype=DataType.VARCHAR, max_length=50),
    FieldSchema(name="monster_name", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="location", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="difficulty", dtype=DataType.VARCHAR, max_length=20),
    FieldSchema(name="synonyms", dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=500),
]
schema = CollectionSchema(fields, description=" Wukong Monsters", enable_dynamic_field=True)
if not client.has_collection(collection_name):
    client.create_collection(collection_name=collection_name, schema=schema)

# 创建索引
index_params = client.prepare_index_params()
index_params.add_index(
    field_name="vector",
    index_type="AUTOINDEX",
    metric_type="L2"
    # AUTOINDEX 不需要额外参数
)
client.create_index(
    collection_name=collection_name, 
    index_params=index_params
)

# 批量插入数据
from tqdm import tqdm
for start_idx in tqdm(range(0, len(df)), desc="插入数据"):
    row = df.iloc[start_idx]    
    # 准备向量文本
    doc_parts = [str(row['monster_name'])]
    if row['synonyms']:
        doc_parts.append(f"(别名：{row['synonyms']})")
    if row['location']:
        doc_parts.append(f"场景：{row['location']}")
    if row['description']:
        doc_parts.append(f"描述：{row['description']}")
    doc_text = "；".join(doc_parts)    
    # 生成向量并插入数据
    embedding = embedding_model.encode([doc_text])[0]  # 使用 embedding_model
    data_to_insert = [{
        "vector": embedding.tolist(),  # 转为列表
        "monster_id": str(row["monster_id"]),
        "monster_name": str(row["monster_name"]),
        "location": str(row["location"]),
        "difficulty": str(row["difficulty"]),
        "synonyms": str(row["synonyms"]),
        "description": str(row["description"])
    }]    
    client.insert(collection_name=collection_name, data=data_to_insert)

# 加载 Collection 到内存（必须！）
print("\n加载 Collection 到内存...")
client.load_collection(collection_name=collection_name)
print("✓ Collection 已加载")

# 测试搜索
search_query = "高难度妖怪"
search_embedding = embedding_model.encode([search_query])[0]  # 使用 embedding_model
search_result = client.search(
    collection_name=collection_name,
    data=[search_embedding.tolist()],
    limit=3,
    output_fields=["monster_name", "location", "difficulty", "synonyms"]
)
print(f"\n搜索结果 '{search_query}':")
for hits in search_result:
    for hit in hits:
        print(f"  - {hit['entity']}")

# 测试条件查询
query_result = client.query(
    collection_name=collection_name,
    filter="difficulty == 'Low'",
    output_fields=["monster_name", "location", "difficulty", "synonyms"]
)
print(f"\n难度为Low的妖怪：")
for result in query_result:
    print(f"  - {result}")
