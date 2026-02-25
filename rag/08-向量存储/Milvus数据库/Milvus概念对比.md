# Milvus 概念对比：MySQL vs MongoDB vs Milvus

## 核心概念对比表

| MySQL | MongoDB | Milvus | 说明 |
|-------|---------|--------|------|
| Database | Database | Database | 数据库 |
| Table | Collection | Collection | 表/集合 |
| Row | Document | Entity | 一条数据 |
| Column | Field | Field | 字段 |
| Index | Index | Index | 索引 |
| - | - | Partition | 分区 |
| Schema | Schema | Schema | 表结构定义 |

---

## 1. Database（数据库）

### 类比：图书馆的不同楼层

**MySQL**:
```sql
CREATE DATABASE company;
USE company;
```

**MongoDB**:
```javascript
use company
```

**Milvus**:
```python
client.create_database(db_name="company")
client.use_database(db_name="company")
```

**比喻**：
- Database 就像图书馆的不同楼层
- 一楼存放小说，二楼存放技术书籍
- 每层楼（Database）独立管理，互不干扰

---

## 2. Collection（集合/表）

### 类比：图书馆里的书架

**MySQL**:
```sql
CREATE TABLE users (
    id INT PRIMARY KEY,
    name VARCHAR(100),
    age INT
);
```

**MongoDB**:
```javascript
db.createCollection("users")
```

**Milvus**:
```python
client.create_collection(
    collection_name="users",
    dimension=128  # 向量维度
)
```

**比喻**：
- Collection 就像图书馆里的一个书架
- "用户信息"书架、"商品信息"书架
- 每个书架（Collection）存放同类型的数据

**Milvus 特点**：
- 必须包含向量字段（这是向量数据库的核心）
- 就像这个书架专门存放"带图片的书"

---

## 3. Schema（表结构）

### 类比：书籍的目录格式

**MySQL**（强制 Schema）:
```sql
CREATE TABLE books (
    id INT PRIMARY KEY,
    title VARCHAR(200),
    author VARCHAR(100),
    price DECIMAL(10,2)
);
```
- 每本书必须有：书名、作者、价格
- 不能随意添加字段

**MongoDB**（灵活 Schema）:
```javascript
// 可以插入不同结构的文档
db.books.insert({title: "三体", author: "刘慈欣"})
db.books.insert({title: "活着", pages: 200})  // 字段可以不同
```
- 每本书可以有不同的信息
- 灵活但不规范

**Milvus**（必须定义 Schema）:
```python
from pymilvus import CollectionSchema, FieldSchema, DataType

# 定义字段
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128)
]

# 创建 Schema
schema = CollectionSchema(fields=fields, description="图书集合")

# 创建 Collection
client.create_collection(
    collection_name="books",
    schema=schema
)
```

**比喻**：
- Schema 就像书籍的目录格式
- MySQL：严格的目录格式，每本书都一样
- MongoDB：自由的目录格式，每本书可以不同
- Milvus：严格的目录格式 + 必须有"图片"（向量）

**Milvus Schema 的特点**：
1. **必须有主键**（就像每本书必须有编号）
2. **必须有向量字段**（就像每本书必须有封面图片）
3. **可以有其他字段**（书名、作者等）

---

## 4. Partition（分区）⭐ 重点

### 类比：书架的不同层

**MySQL**（分区表）:
```sql
CREATE TABLE orders (
    id INT,
    order_date DATE,
    amount DECIMAL
) PARTITION BY RANGE (YEAR(order_date)) (
    PARTITION p2020 VALUES LESS THAN (2021),
    PARTITION p2021 VALUES LESS THAN (2022),
    PARTITION p2022 VALUES LESS THAN (2023)
);
```
- 按年份分区
- 查询 2021 年数据时，只扫描 p2021 分区

**MongoDB**（分片）:
```javascript
// 按地区分片
sh.shardCollection("mydb.users", {region: 1})
```
- 按地区分片
- 查询"北京"用户时，只查询北京分片

**Milvus**（Partition）:
```python
# 创建分区
client.create_partition(
    collection_name="books",
    partition_name="fiction"      # 小说分区
)
client.create_partition(
    collection_name="books",
    partition_name="technology"   # 技术书分区
)

# 插入数据到指定分区
client.insert(
    collection_name="books",
    data=[...],
    partition_name="fiction"  # 插入到小说分区
)

# 只在小说分区搜索
client.search(
    collection_name="books",
    data=[query_vector],
    partition_names=["fiction"]  # 只搜索小说分区
)
```

**比喻**：
```
书架（Collection）
├── 第1层（Partition: fiction）
│   ├── 三体
│   ├── 活着
│   └── 平凡的世界
├── 第2层（Partition: technology）
│   ├── Python编程
│   ├── 算法导论
│   └── 深度学习
└── 第3层（Partition: history）
    ├── 史记
    └── 资治通鉴
```

**为什么需要 Partition？**

1. **提高搜索速度**
   ```python
   # 不用分区：搜索整个书架（10000本书）
   results = client.search(collection_name="books", data=[vector])
   
   # 使用分区：只搜索小说层（3000本书）
   results = client.search(
       collection_name="books",
       data=[vector],
       partition_names=["fiction"]  # 速度提升 3 倍！
   )
   ```

2. **数据管理更方便**
   ```python
   # 删除所有历史书籍
   client.drop_partition(
       collection_name="books",
       partition_name="history"
   )
   ```

3. **节省内存**
   ```python
   # 只加载小说分区到内存
   client.load_partitions(
       collection_name="books",
       partition_names=["fiction"]
   )
   ```

---

## 5. Entity（实体/数据行）

### 类比：书架上的一本书

**MySQL**:
```sql
INSERT INTO books VALUES (1, '三体', '刘慈欣', 45.00);
```
- 一行数据 = 一本书

**MongoDB**:
```javascript
db.books.insert({
    _id: 1,
    title: "三体",
    author: "刘慈欣",
    price: 45.00
})
```
- 一个文档 = 一本书

**Milvus**:
```python
client.insert(
    collection_name="books",
    data=[{
        "id": 1,
        "title": "三体",
        "embedding": [0.1, 0.2, ..., 0.128]  # 128维向量
    }]
)
```
- 一个 Entity = 一本书 + 它的"图片"（向量）

**比喻**：
- Entity 就是书架上的一本具体的书
- 每本书有：编号、书名、封面图片（向量）

---

## 实际应用场景对比

### 场景1：用户管理系统

**MySQL**（适合结构化数据）:
```sql
-- 查询年龄大于 25 的用户
SELECT * FROM users WHERE age > 25;
```

**MongoDB**（适合灵活数据）:
```javascript
// 查询有"爱好"字段的用户
db.users.find({hobbies: {$exists: true}})
```

**Milvus**（适合相似度搜索）:
```python
# 查询"长得像张三"的用户（人脸识别）
results = client.search(
    collection_name="users",
    data=[zhangsan_face_vector],  # 张三的人脸向量
    limit=10
)
```

---

### 场景2：电商商品搜索

**MySQL**（精确搜索）:
```sql
-- 查询价格在 100-200 之间的商品
SELECT * FROM products 
WHERE price BETWEEN 100 AND 200;
```

**MongoDB**（灵活查询）:
```javascript
// 查询包含"手机"标签的商品
db.products.find({tags: "手机"})
```

**Milvus**（相似度搜索）:
```python
# 用户上传图片，搜索相似商品（以图搜图）
results = client.search(
    collection_name="products",
    data=[user_upload_image_vector],
    partition_names=["electronics"],  # 只在电子产品分区搜索
    limit=20
)
```

---

## Partition 的实际应用

### 应用1：按时间分区

```python
# 创建按月份的分区
client.create_partition(collection_name="logs", partition_name="2024_01")
client.create_partition(collection_name="logs", partition_name="2024_02")

# 查询最近一个月的日志
results = client.search(
    collection_name="logs",
    data=[query_vector],
    partition_names=["2024_02"]  # 只搜索2月的数据
)

# 删除旧数据
client.drop_partition(collection_name="logs", partition_name="2024_01")
```

### 应用2：按类别分区

```python
# 电商商品分区
client.create_partition(collection_name="products", partition_name="electronics")
client.create_partition(collection_name="products", partition_name="clothing")
client.create_partition(collection_name="products", partition_name="food")

# 用户搜索"手机"，只在电子产品分区搜索
results = client.search(
    collection_name="products",
    data=[query_vector],
    partition_names=["electronics"]  # 速度更快！
)
```

### 应用3：按用户分区

```python
# VIP 用户和普通用户分区
client.create_partition(collection_name="users", partition_name="vip")
client.create_partition(collection_name="users", partition_name="normal")

# 只在 VIP 用户中搜索
results = client.search(
    collection_name="users",
    data=[query_vector],
    partition_names=["vip"]
)
```

---

## 总结

### 核心区别

| 特性 | MySQL | MongoDB | Milvus |
|------|-------|---------|--------|
| 数据类型 | 结构化数据 | 半结构化数据 | 向量数据 |
| 查询方式 | 精确查询 | 灵活查询 | 相似度搜索 |
| Schema | 强制 | 可选 | 强制（必须有向量） |
| 分区 | 支持 | 分片 | 支持（性能优化） |
| 适用场景 | 事务处理 | 灵活存储 | AI搜索 |

### 记忆口诀

```
Database = 图书馆的楼层（隔离不同业务）
Collection = 书架（存放同类数据）
Schema = 目录格式（定义数据结构）
Partition = 书架的层（加速搜索）
Entity = 一本书（一条数据）
Vector = 书的封面图片（用于相似度搜索）
```

### 什么时候用 Partition？

✅ **应该用**：
- 数据量大（百万级以上）
- 有明确的分类（时间、类别、地区）
- 查询时只需要搜索部分数据

❌ **不需要用**：
- 数据量小（几万条）
- 没有明确的分类
- 每次都要搜索全部数据

### 实际例子

```python
# 场景：图片搜索系统（100万张图片）

# 不用分区：每次搜索 100万 张图片（慢）
results = client.search(
    collection_name="images",
    data=[query_vector],
    limit=10
)

# 使用分区：只搜索"风景"类别的 20万 张图片（快5倍！）
results = client.search(
    collection_name="images",
    data=[query_vector],
    partition_names=["landscape"],  # 只搜索风景分区
    limit=10
)
```

希望这些类比能帮你理解 Milvus 的概念！
