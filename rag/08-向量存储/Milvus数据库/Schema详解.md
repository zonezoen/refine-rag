# Milvus Schema 详解：快速模式 vs 自定义模式

## 两种创建 Collection 的方式

### 方式1：快速模式（Quick Setup）⚡

**特点**：自动创建 Schema，支持动态字段

```python
# 只需要指定最基本的参数
client.create_collection(
    collection_name="quick_setup",
    dimension=5,  # 向量维度
    # 其他参数可选
)
```

**Milvus 自动创建的 Schema**：
```python
# 等价于以下完整定义：
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=5)
]
schema = CollectionSchema(
    fields=fields,
    enable_dynamic_field=True  # ⭐ 关键：开启动态字段！
)
```

**enable_dynamic_field=True 的作用**：
- 允许插入 Schema 中未定义的字段
- 就像 MongoDB 的灵活 Schema
- `color` 字段就是这样来的！

---

## color 字段从哪里来？

### 类比：图书馆的登记系统

**MySQL（严格模式）**：
```sql
CREATE TABLE books (
    id INT PRIMARY KEY,
    title VARCHAR(100)
);

-- 插入数据时，只能插入定义好的字段
INSERT INTO books VALUES (1, '三体');

-- ❌ 错误！author 字段不存在
INSERT INTO books (id, title, author) VALUES (2, '活着', '余华');
```

**Milvus 快速模式（灵活模式）**：
```python
# 创建时只定义了 id 和 vector
client.create_collection(
    collection_name="quick_setup",
    dimension=5
)

# ✅ 可以插入额外的字段！
client.insert(
    collection_name="quick_setup",
    data=[
        {
            "id": 0,
            "vector": [0.1, 0.2, 0.3, 0.4, 0.5],
            "color": "pink_8682",      # ✅ 动态字段
            "price": 99.9,             # ✅ 动态字段
            "author": "刘慈欣"          # ✅ 动态字段
        }
    ]
)
```

---

## 完整示例对比

### 示例1：快速模式（自动 Schema + 动态字段）

```python
import logging
from pymilvus import MilvusClient

logging.getLogger("pymilvus").setLevel(logging.CRITICAL)

client = MilvusClient(uri="http://localhost:19530")

# ——————————————
# 1. 创建集合（快速模式）
# ——————————————
client.create_collection(
    collection_name="quick_setup",
    dimension=5,  # 只需要指定向量维度
)
print("✓ 集合创建成功（快速模式）")

# 查看自动生成的 Schema
info = client.describe_collection(collection_name="quick_setup")
print("\n自动生成的 Schema：")
for field in info['fields']:
    print(f"  - {field['name']}: {field['type']}")
print(f"  - enable_dynamic_field: {info['enable_dynamic_field']}")

# ——————————————
# 2. 插入数据（可以包含任意字段）
# ——————————————
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
        "stock": 100               # ✅ 不同的动态字段也可以！
    }
]

client.insert(
    collection_name="quick_setup",
    data=data
)
print("\n✓ 数据插入成功（包含动态字段）")

# ——————————————
# 3. 查询数据
# ——————————————
results = client.query(
    collection_name="quick_setup",
    filter="id in [0, 1]",
    output_fields=["id", "color", "price", "tag", "stock"]
)
print("\n查询结果：")
for result in results:
    print(f"  {result}")

# 清理
client.drop_collection(collection_name="quick_setup")
```

**输出**：
```
✓ 集合创建成功（快速模式）

自动生成的 Schema：
  - id: DataType.INT64
  - vector: DataType.FLOAT_VECTOR
  - enable_dynamic_field: True  ← 关键！

✓ 数据插入成功（包含动态字段）

查询结果：
  {'id': 0, 'color': 'pink_8682', 'price': 99.9, 'tag': 'hot'}
  {'id': 1, 'color': 'red_7025', 'stock': 100}
```

---

### 示例2：自定义模式（手动 Schema + 严格字段）

```python
from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema

client = MilvusClient(uri="http://localhost:19530")

# ——————————————
# 1. 手动定义 Schema
# ——————————————
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=5),
    FieldSchema(name="color", dtype=DataType.VARCHAR, max_length=50),  # 明确定义 color
    FieldSchema(name="price", dtype=DataType.FLOAT)                    # 明确定义 price
]

schema = CollectionSchema(
    fields=fields,
    description="自定义 Schema",
    enable_dynamic_field=False  # ⭐ 关闭动态字段
)

# ——————————————
# 2. 创建集合
# ——————————————
client.create_collection(
    collection_name="custom_setup",
    schema=schema
)
print("✓ 集合创建成功（自定义模式）")

# ——————————————
# 3. 插入数据（必须符合 Schema）
# ——————————————
data = [
    {
        "id": 0,
        "vector": [0.1, 0.2, 0.3, 0.4, 0.5],
        "color": "pink_8682",
        "price": 99.9
        # ❌ 不能添加其他字段！
    }
]

client.insert(
    collection_name="custom_setup",
    data=data
)
print("✓ 数据插入成功（严格模式）")

# ——————————————
# 4. 尝试插入额外字段（会失败）
# ——————————————
try:
    data_with_extra = [
        {
            "id": 1,
            "vector": [0.2, 0.3, 0.4, 0.5, 0.6],
            "color": "red_7025",
            "price": 88.8,
            "tag": "hot"  # ❌ 额外字段
        }
    ]
    client.insert(
        collection_name="custom_setup",
        data=data_with_extra
    )
except Exception as e:
    print(f"\n❌ 插入失败：{e}")
    print("   原因：enable_dynamic_field=False，不允许额外字段")

# 清理
client.drop_collection(collection_name="custom_setup")
```

---

## 两种模式对比

| 特性 | 快速模式 | 自定义模式 |
|------|---------|-----------|
| Schema 定义 | 自动生成 | 手动定义 |
| 动态字段 | ✅ 支持 | ❌ 不支持（默认） |
| 灵活性 | 高（像 MongoDB） | 低（像 MySQL） |
| 类型检查 | 宽松 | 严格 |
| 适用场景 | 快速原型、灵活数据 | 生产环境、严格规范 |

---

## 动态字段的工作原理

### 类比：图书馆的备注栏

**快速模式（enable_dynamic_field=True）**：
```
图书登记表：
┌────┬──────┬─────────────────┐
│ ID │ 书名 │ 备注栏（随意写）│
├────┼──────┼─────────────────┤
│ 1  │ 三体 │ 颜色:红色       │
│ 2  │ 活着 │ 价格:39元       │
│ 3  │ 史记 │ 作者:司马迁     │
└────┴──────┴─────────────────┘
```
- 每本书的备注可以不同
- 灵活但不规范

**自定义模式（enable_dynamic_field=False）**：
```
图书登记表：
┌────┬──────┬──────┬──────┐
│ ID │ 书名 │ 颜色 │ 价格 │
├────┼──────┼──────┼──────┤
│ 1  │ 三体 │ 红色 │ 45元 │
│ 2  │ 活着 │ 蓝色 │ 39元 │
│ 3  │ 史记 │ 黄色 │ 88元 │
└────┴──────┴──────┴──────┘
```
- 每本书必须填写相同的字段
- 规范但不灵活

---

## 实际应用建议

### 场景1：快速原型开发 → 使用快速模式

```python
# 快速开始，不确定需要哪些字段
client.create_collection(
    collection_name="prototype",
    dimension=128
)

# 可以随时添加新字段
data = [
    {"id": 1, "vector": [...], "name": "张三"},
    {"id": 2, "vector": [...], "age": 25},      # 不同字段
    {"id": 3, "vector": [...], "city": "北京"}  # 不同字段
]
```

### 场景2：生产环境 → 使用自定义模式

```python
# 明确定义所有字段
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128),
    FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="age", dtype=DataType.INT64),
    FieldSchema(name="city", dtype=DataType.VARCHAR, max_length=50)
]

schema = CollectionSchema(fields=fields, enable_dynamic_field=False)

client.create_collection(
    collection_name="production",
    schema=schema
)

# 所有数据必须符合 Schema
data = [
    {"id": 1, "vector": [...], "name": "张三", "age": 25, "city": "北京"}
]
```

---

## 常见问题

### Q1: 快速模式下，动态字段可以被索引吗？

**A**: 不可以。只有在 Schema 中明确定义的字段才能创建索引。

```python
# ❌ 不能为动态字段创建索引
client.create_index(
    collection_name="quick_setup",
    field_name="color",  # 动态字段
    index_type="INVERTED"
)
# 会报错：field not found

# ✅ 只能为 Schema 中定义的字段创建索引
client.create_index(
    collection_name="quick_setup",
    field_name="vector",  # Schema 中的字段
    index_type="IVF_FLAT"
)
```

### Q2: 快速模式下，动态字段可以用于过滤吗？

**A**: 可以！

```python
# ✅ 可以用动态字段过滤
results = client.query(
    collection_name="quick_setup",
    filter='color == "pink_8682"',  # 动态字段
    output_fields=["id", "color"]
)
```

### Q3: 如何在自定义模式下也支持动态字段？

**A**: 设置 `enable_dynamic_field=True`

```python
schema = CollectionSchema(
    fields=fields,
    enable_dynamic_field=True  # ⭐ 开启动态字段
)
```

---

## 总结

### 快速模式的本质

```python
# 你写的代码
client.create_collection(
    collection_name="quick_setup",
    dimension=5
)

# Milvus 实际做的事
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=5)
]
schema = CollectionSchema(
    fields=fields,
    enable_dynamic_field=True  # ← color 字段的来源！
)
client.create_collection(
    collection_name="quick_setup",
    schema=schema
)
```

### 记忆口诀

```
快速模式 = 自动 Schema + 动态字段
自定义模式 = 手动 Schema + 严格字段

color 字段 = 动态字段（enable_dynamic_field=True）
```

### 选择建议

- 🚀 **学习/原型** → 快速模式（灵活）
- 🏭 **生产环境** → 自定义模式（规范）
- 🔄 **需要索引** → 必须在 Schema 中定义字段
- 📊 **只需过滤** → 动态字段就够了
