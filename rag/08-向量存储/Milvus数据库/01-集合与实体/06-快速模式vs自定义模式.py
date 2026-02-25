"""
快速模式 vs 自定义模式对比示例

演示：
1. 快速模式：自动 Schema + 动态字段
2. 自定义模式：手动 Schema + 严格字段
"""

import logging
from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema

# 隐藏 pymilvus 的 ERROR 日志
logging.getLogger("pymilvus").setLevel(logging.CRITICAL)

client = MilvusClient(uri="http://localhost:19530")

print("="*60)
print("示例1：快速模式（自动 Schema + 动态字段）")
print("="*60)

# ——————————————
# 1. 创建集合（快速模式）
# ——————————————
collection_name = "quick_setup"

# 删除已存在的集合
if collection_name in client.list_collections():
    client.drop_collection(collection_name=collection_name)

# 创建集合（只需要指定向量维度）
client.create_collection(
    collection_name=collection_name,
    dimension=5,  # 向量维度
)
print("\n✓ 集合创建成功（快速模式）")

# 查看自动生成的 Schema
info = client.describe_collection(collection_name=collection_name)
print("\n自动生成的 Schema：")
for field in info['fields']:
    print(f"  - {field['name']}: {field['type']}")
print(f"  - enable_dynamic_field: {info['enable_dynamic_field']}")

# ——————————————
# 2. 插入数据（可以包含任意字段）
# ——————————————
print("\n插入数据（包含动态字段）...")
data = [
    {
        "id": 0,
        "vector": [0.1, 0.2, 0.3, 0.4, 0.5],
        "color": "pink_8682",      # ✅ 动态字段
        "price": 99.9,             # ✅ 动态字段
        "tag": "hot"               # ✅ 动态字段
    },
    {
        "id": 1,
        "vector": [0.2, 0.3, 0.4, 0.5, 0.6],
        "color": "red_7025",       # ✅ 动态字段
        "stock": 100,              # ✅ 不同的动态字段也可以！
        "category": "electronics"  # ✅ 动态字段
    },
    {
        "id": 2,
        "vector": [0.3, 0.4, 0.5, 0.6, 0.7],
        "color": "blue_1234",
        "author": "张三",          # ✅ 又是不同的字段
        "year": 2024
    }
]

client.insert(
    collection_name=collection_name,
    data=data
)
print("✓ 数据插入成功")

# ——————————————
# 3. 搜索数据
# ——————————————
print("\n搜索结果（前3条）：")
# 使用向量搜索（自动创建索引）
results = client.search(
    collection_name=collection_name,
    data=[[0.1, 0.2, 0.3, 0.4, 0.5]],
    limit=3,
    output_fields=["*"]
)
for hits in results:
    for hit in hits:
        print(f"  ID {hit['id']}: {hit['entity']}")

# ——————————————
# 4. 使用动态字段过滤搜索
# ——————————————
print("\n使用动态字段过滤搜索（color == 'red_7025'）：")
results = client.search(
    collection_name=collection_name,
    data=[[0.2, 0.3, 0.4, 0.5, 0.6]],
    limit=3,
    filter='color == "red_7025"',
    output_fields=["id", "color", "stock", "category"]
)
for hits in results:
    for hit in hits:
        print(f"  {hit['entity']}")

# 清理
client.drop_collection(collection_name=collection_name)
print("\n✓ 集合已删除")

# ============================================================
print("\n" + "="*60)
print("示例2：自定义模式（手动 Schema + 严格字段）")
print("="*60)

# ——————————————
# 1. 手动定义 Schema
# ——————————————
collection_name = "custom_setup"

# 删除已存在的集合
if collection_name in client.list_collections():
    client.drop_collection(collection_name=collection_name)

print("\n定义 Schema...")
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=5),
    FieldSchema(name="color", dtype=DataType.VARCHAR, max_length=50),  # 明确定义
    FieldSchema(name="price", dtype=DataType.FLOAT)                    # 明确定义
]

schema = CollectionSchema(
    fields=fields,
    description="自定义 Schema 示例",
    enable_dynamic_field=False  # ⭐ 关闭动态字段
)

# ——————————————
# 2. 创建集合
# ——————————————
client.create_collection(
    collection_name=collection_name,
    schema=schema
)
print("✓ 集合创建成功（自定义模式）")

# 查看 Schema
info = client.describe_collection(collection_name=collection_name)
print("\n自定义的 Schema：")
for field in info['fields']:
    print(f"  - {field['name']}: {field['type']}")
print(f"  - enable_dynamic_field: {info['enable_dynamic_field']}")

# ——————————————
# 3. 插入数据（必须符合 Schema）
# ——————————————
print("\n插入符合 Schema 的数据...")
data = [
    {
        "id": 0,
        "vector": [0.1, 0.2, 0.3, 0.4, 0.5],
        "color": "pink_8682",
        "price": 99.9
    },
    {
        "id": 1,
        "vector": [0.2, 0.3, 0.4, 0.5, 0.6],
        "color": "red_7025",
        "price": 88.8
    }
]

client.insert(
    collection_name=collection_name,
    data=data
)
print("✓ 数据插入成功")

# ——————————————
# 4. 尝试插入额外字段（会报错）
# ——————————————
print("\n尝试插入包含额外字段的数据...")
try:
    data_with_extra = [
        {
            "id": 2,
            "vector": [0.3, 0.4, 0.5, 0.6, 0.7],
            "color": "blue_1234",
            "price": 77.7,
            "tag": "hot",          # ❌ 额外字段
            "stock": 100           # ❌ 额外字段
        }
    ]
    client.insert(
        collection_name=collection_name,
        data=data_with_extra
    )
    print("✓ 数据插入成功（额外字段被忽略）")
    
except Exception as e:
    print(f"❌ 插入失败：{e}")
    print("   原因：enable_dynamic_field=False，不允许额外字段")

# ——————————————
# 5. 说明
# ——————————————
print("\n说明：")
print("  - 自定义模式下，只能插入 Schema 中定义的字段")
print("  - 额外字段会被拒绝")
print("  - 如果需要灵活性，使用混合模式（enable_dynamic_field=True）")

# 清理
client.drop_collection(collection_name=collection_name)
print("\n✓ 集合已删除")

# ============================================================
print("\n" + "="*60)
print("示例3：自定义模式 + 开启动态字段（两全其美）")
print("="*60)

# ——————————————
# 1. 定义 Schema（开启动态字段）
# ——————————————
collection_name = "hybrid_setup"

# 删除已存在的集合
if collection_name in client.list_collections():
    client.drop_collection(collection_name=collection_name)

print("\n定义 Schema（开启动态字段）...")
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=5),
    FieldSchema(name="color", dtype=DataType.VARCHAR, max_length=50),  # 明确定义
    FieldSchema(name="price", dtype=DataType.FLOAT)                    # 明确定义
]

schema = CollectionSchema(
    fields=fields,
    description="混合模式：自定义 Schema + 动态字段",
    enable_dynamic_field=True  # ⭐ 开启动态字段
)

# ——————————————
# 2. 创建集合
# ——————————————
client.create_collection(
    collection_name=collection_name,
    schema=schema
)
print("✓ 集合创建成功（混合模式）")

# 查看 Schema
info = client.describe_collection(collection_name=collection_name)
print("\n混合模式的 Schema：")
for field in info['fields']:
    print(f"  - {field['name']}: {field['type']}")
print(f"  - enable_dynamic_field: {info['enable_dynamic_field']}")

# ——————————————
# 3. 插入数据（既有固定字段，也有动态字段）
# ——————————————
print("\n插入数据（包含固定字段和动态字段）...")
data = [
    {
        "id": 0,
        "vector": [0.1, 0.2, 0.3, 0.4, 0.5],
        "color": "pink_8682",      # 固定字段
        "price": 99.9,             # 固定字段
        "tag": "hot",              # ✅ 动态字段
        "stock": 100               # ✅ 动态字段
    },
    {
        "id": 1,
        "vector": [0.2, 0.3, 0.4, 0.5, 0.6],
        "color": "red_7025",
        "price": 88.8,
        "author": "张三",          # ✅ 不同的动态字段
        "year": 2024
    }
]

client.insert(
    collection_name=collection_name,
    data=data
)
print("✓ 数据插入成功")

# ——————————————
# 4. 说明搜索功能
# ——————————————
print("\n说明：")
print("  - 固定字段和动态字段都可以用于过滤")
print("  - 每条数据的动态字段可以不同")
print("  - 搜索前需要先加载 Collection 到内存")

# ——————————————
# 5. 为固定字段创建索引（动态字段不能创建索引）
# ——————————————
print("\n说明：")
print("  - 固定字段（color, price）可以创建索引")
print("  - 动态字段（tag, stock, author, year）不能创建索引")
print("  - 但动态字段可以用于过滤搜索")

# 清理
client.drop_collection(collection_name=collection_name)
print("\n✓ 集合已删除")

# ============================================================
print("\n" + "="*60)
print("总结")
print("="*60)
print("""
1. 快速模式：
   - 自动生成 Schema（id + vector）
   - 自动开启动态字段
   - 适合：快速原型、灵活数据

2. 自定义模式（关闭动态字段）：
   - 手动定义所有字段
   - 严格的类型检查
   - 适合：生产环境、规范数据

3. 混合模式（自定义 + 动态字段）：
   - 手动定义核心字段（可创建索引）
   - 支持额外的动态字段（灵活性）
   - 适合：需要索引 + 需要灵活性

推荐：生产环境使用混合模式！
""")
