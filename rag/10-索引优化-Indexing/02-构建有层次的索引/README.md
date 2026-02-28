# 构建有层次的索引 - 解释文档

## 📖 什么是层次化索引？

想象你在超市购物：
- **传统方式**：在整个超市里找一瓶酱油（慢，效率低）
- **层次化方式**：先找到调味品区 → 再找酱油货架 → 最后找到目标品牌

层次化索引就是这样：**先粗略定位，再精确查找**。

---

## 🎯 核心原理

### 为什么需要层次化索引？

在大规模数据检索中存在的问题：

```
单层索引的问题：
├─ 数据量大：需要在海量数据中检索
├─ 检索慢：每次都要遍历所有数据
└─ 精度低：相似的内容太多，难以区分

层次化索引的解决方案：
├─ 第一层（粗）：快速定位到相关类别/文档
├─ 第二层（细）：在小范围内精确检索
└─ 结果：速度快 + 精度高
```

### 通俗比喻

**图书馆找书**：
```
第一层：按学科分类（计算机、文学、历史...）
         ↓ 快速定位到"计算机"区域
第二层：在计算机区域找具体的书
         ↓ 找到"Python编程"

而不是在整个图书馆里一本本找！
```

**外卖点餐**：
```
第一层：选择餐厅类型（中餐、西餐、快餐...）
         ↓ 选择"中餐"
第二层：在中餐里选择具体菜品
         ↓ 选择"宫保鸡丁"
```

---

## 🏗️ 实现方法

### 方法1：双层索引（Summary + Details）

**原理**：建立两个向量数据库，一个存摘要，一个存详情。

**结构示例**：
```
【第一层：摘要索引】
├─ 2020年富豪榜 → 向量1
├─ 2021年富豪榜 → 向量2
├─ 2022年富豪榜 → 向量3
└─ 2023年富豪榜 → 向量4

【第二层：详细索引】
├─ 2020年富豪榜 → 详细数据（10条记录）
├─ 2021年富豪榜 → 详细数据（10条记录）
├─ 2022年富豪榜 → 详细数据（10条记录）
└─ 2023年富豪榜 → 详细数据（10条记录）
```

**检索流程**：
```python
# 用户问："2023年世界首富是谁？"

# 第一步：在摘要索引中检索
query = "2023年世界首富是谁？"
summary_result = search_summary(query)
# 结果：匹配到"2023年富豪榜"

# 第二步：在详细索引中检索
matched_table = "2023年富豪榜"
detail_result = search_details(
    table_name=matched_table,
    query="首富"
)
# 结果：返回2023年排名第一的富豪信息
```

**代码示例**：
```python
# 创建两个集合
summary_collection = "billionaires_summary"  # 摘要
details_collection = "billionaires_details"  # 详情

# 第一层检索：找到相关年份
summary_results = client.search(
    collection_name=summary_collection,
    data=[query_embedding],
    limit=1
)
matched_table = summary_results[0]['table_name']  # 如："2023年富豪榜"

# 第二层检索：在该年份内查找
details_results = client.search(
    collection_name=details_collection,
    data=[query_embedding],
    filter=f"table_name == '{matched_table}'",  # 只在2023年数据中找
    limit=1
)
```

**适用场景**：
- ✅ 多个独立的数据表/文档集合
- ✅ 数据有明确的分类（年份、类别、主题）
- ✅ 需要先定位再查找的场景
- ❌ 不适合：数据无明确分类、小规模数据

---

### 方法2：递归检索（Recursive Retriever）

**原理**：使用索引节点（IndexNode）指向具体的查询引擎。

**结构示例**：
```
【顶层：索引节点】
├─ IndexNode1: "2020年富豪信息" → 指向 PandasQueryEngine1
├─ IndexNode2: "2021年富豪信息" → 指向 PandasQueryEngine2
├─ IndexNode3: "2022年富豪信息" → 指向 PandasQueryEngine3
└─ IndexNode4: "2023年富豪信息" → 指向 PandasQueryEngine4

【底层：查询引擎】
├─ PandasQueryEngine1 → 2020年DataFrame
├─ PandasQueryEngine2 → 2021年DataFrame
├─ PandasQueryEngine3 → 2022年DataFrame
└─ PandasQueryEngine4 → 2023年DataFrame
```

**检索流程**：
```
1. 用户查询 → 匹配IndexNode
2. 找到对应的查询引擎
3. 在该查询引擎中执行查询
4. 返回结果
```

**代码示例**：
```python
from llama_index.core.schema import IndexNode
from llama_index.core.retrievers import RecursiveRetriever

# 创建索引节点
df_nodes = [
    IndexNode(
        text="2020年世界富豪信息",
        index_id="pandas0"
    ),
    IndexNode(
        text="2021年世界富豪信息",
        index_id="pandas1"
    ),
    # ...
]

# 创建查询引擎映射
df_id_query_engine_mapping = {
    "pandas0": pandas_engine_2020,
    "pandas1": pandas_engine_2021,
    # ...
}

# 创建递归检索器
recursive_retriever = RecursiveRetriever(
    "vector",
    retriever_dict={"vector": vector_retriever},
    query_engine_dict=df_id_query_engine_mapping,
    verbose=True
)
```

**适用场景**：
- ✅ 不同数据源需要不同的查询方式
- ✅ 结构化数据（表格、数据库）
- ✅ 需要灵活的查询引擎
- ❌ 不适合：简单的文本检索

---

### 方法3：粗中有细（Coarse-to-Fine）

**原理**：先用粗粒度快速过滤，再用细粒度精确匹配。

**流程示例**：
```
【粗粒度检索】
查询："Python的列表推导式"
↓
快速匹配到：Python相关文档（100篇）

【细粒度检索】
在Python文档中检索："列表推导式"
↓
精确找到：列表推导式的详细说明（3篇）
```

**适用场景**：
- ✅ 超大规模数据集
- ✅ 需要快速响应
- ✅ 可以明确分类的数据

---

## ⚡ 性能对比

### 检索速度

```
单层索引：     ⭐⭐        （需要遍历所有数据）
双层索引：     ⭐⭐⭐⭐    （先过滤再检索）
递归检索：     ⭐⭐⭐⭐    （直接定位到引擎）
粗中有细：     ⭐⭐⭐⭐⭐  （最快，两次快速检索）
```

### 检索精度

```
单层索引：     ⭐⭐⭐      （数据多时容易混淆）
双层索引：     ⭐⭐⭐⭐⭐  （分层过滤，精度高）
递归检索：     ⭐⭐⭐⭐⭐  （精确定位）
粗中有细：     ⭐⭐⭐⭐    （取决于粗粒度质量）
```

### 实现复杂度

```
单层索引：     ⭐          （最简单）
双层索引：     ⭐⭐⭐      （需要维护两个索引）
递归检索：     ⭐⭐⭐⭐    （需要配置多个引擎）
粗中有细：     ⭐⭐⭐      （需要设计分层策略）
```

### 存储成本

```
单层索引：     ⭐          （只存一份）
双层索引：     ⭐⭐        （存两份索引）
递归检索：     ⭐⭐        （索引+原始数据）
粗中有细：     ⭐⭐⭐      （粗+细两层索引）
```

---

## 🎓 实战案例

### 案例1：富豪榜查询系统

**场景**：有2020-2023年的世界富豪榜数据，用户查询"2023年首富是谁？"

**数据结构**：
```
世界十大富豪.xlsx
├─ Sheet: 2020年
├─ Sheet: 2021年
├─ Sheet: 2022年
└─ Sheet: 2023年
```

**使用双层索引**：

```python
# 第一层：年份索引
summary_data = [
    {"table_name": "2020年", "vector": embed("2020年富豪榜")},
    {"table_name": "2021年", "vector": embed("2021年富豪榜")},
    {"table_name": "2022年", "vector": embed("2022年富豪榜")},
    {"table_name": "2023年", "vector": embed("2023年富豪榜")},
]

# 第二层：详细数据索引
details_data = [
    {"table_name": "2020年", "content": "...", "vector": embed("...")},
    {"table_name": "2021年", "content": "...", "vector": embed("...")},
    # ...
]

# 查询流程
query = "2023年首富是谁？"

# Step 1: 在summary中找年份
matched_year = search_summary(query)  # 结果："2023年"

# Step 2: 在details中找具体信息
result = search_details(
    filter=f"table_name == '{matched_year}'",
    query="首富"
)
# 结果：2023年排名第一的富豪信息
```

**效果**：
- 不需要在所有年份的数据中检索
- 先定位到2023年，再在2023年数据中查找
- 速度快，精度高

---

### 案例2：技术文档检索

**场景**：有多个技术文档，用户查询"如何配置Redis？"

**文档结构**：
```
技术文档/
├─ 数据库配置.md
│   ├─ MySQL配置
│   ├─ PostgreSQL配置
│   └─ Redis配置  ← 目标
├─ 服务器配置.md
├─ 网络配置.md
└─ 安全配置.md
```

**使用递归检索**：

```python
# 创建文档级别的索引节点
doc_nodes = [
    IndexNode(text="数据库配置文档", index_id="db_doc"),
    IndexNode(text="服务器配置文档", index_id="server_doc"),
    IndexNode(text="网络配置文档", index_id="network_doc"),
    IndexNode(text="安全配置文档", index_id="security_doc"),
]

# 每个文档有自己的查询引擎
query_engines = {
    "db_doc": db_query_engine,
    "server_doc": server_query_engine,
    # ...
}

# 查询流程
query = "如何配置Redis？"

# Step 1: 匹配到"数据库配置文档"
# Step 2: 使用db_query_engine在数据库文档中查找
# Step 3: 返回Redis配置部分
```

**效果**：
- 自动定位到正确的文档
- 在该文档内精确查找
- 避免在无关文档中浪费时间

---

### 案例3：电商商品检索

**场景**：电商平台有数百万商品，用户搜索"红色连衣裙"

**使用粗中有细**：

```python
# 粗粒度：商品类别
categories = ["服装", "电子", "食品", "家居", ...]

# Step 1: 粗检索
query = "红色连衣裙"
matched_category = coarse_search(query)  # 结果："服装"

# Step 2: 细检索
# 只在"服装"类别中检索
results = fine_search(
    category="服装",
    query="红色连衣裙"
)
# 结果：服装类别下的红色连衣裙商品
```

**效果**：
- 第一步快速过滤掉99%的无关商品
- 第二步在小范围内精确匹配
- 检索速度提升10倍以上

---

## 🔧 参数调优指南

### 双层索引参数

```python
# 第一层检索参数
summary_search_params = {
    "limit": 1,           # 通常只需要1个结果
    "metric_type": "COSINE",
    "params": {"nprobe": 10}
}

# 第二层检索参数
details_search_params = {
    "limit": 3,           # 可以返回多个结果
    "metric_type": "COSINE",
    "params": {"nprobe": 10}
}
```

**调优建议**：
```
第一层limit：
- 设为1：只定位到最相关的类别（推荐）
- 设为2-3：考虑多个可能的类别

第二层limit：
- 根据实际需求设置
- 一般3-5个结果即可
```

---

### 递归检索参数

```python
RecursiveRetriever(
    "vector",
    retriever_dict={"vector": vector_retriever},
    query_engine_dict=df_id_query_engine_mapping,
    verbose=True  # 开启调试信息
)
```

**调优建议**：
```
verbose=True：
- 开发阶段：开启，方便调试
- 生产环境：关闭，提高性能

retriever的similarity_top_k：
- 设为1-2：快速定位
- 设为3-5：考虑多个可能
```

---

## 💡 最佳实践

### 实践1：合理设计层次结构

```python
# ✅ 好的层次设计
第一层：年份（2020, 2021, 2022, 2023）
第二层：具体数据

# ❌ 不好的层次设计
第一层：年份
第二层：月份
第三层：日期
第四层：具体数据
# 层次太多，反而降低效率
```

**原则**：
- 2-3层最佳
- 每层的分类要清晰
- 避免过度分层

---

### 实践2：第一层要足够"粗"

```python
# ✅ 好的粗粒度
categories = ["技术文档", "产品文档", "用户手册"]  # 3个大类

# ❌ 不好的粗粒度
categories = ["Python", "Java", "C++", "Go", "Rust", ...]  # 太细了
# 第一层就有几十个类别，失去了"粗"的意义
```

**原则**：
- 第一层类别数：3-10个
- 太少：过滤效果不明显
- 太多：第一层检索就很慢

---

### 实践3：监控两层的检索效果

```python
def hierarchical_search(query):
    # 第一层
    start_time = time.time()
    category = first_layer_search(query)
    first_layer_time = time.time() - start_time
    
    # 第二层
    start_time = time.time()
    results = second_layer_search(category, query)
    second_layer_time = time.time() - start_time
    
    # 记录日志
    log_search(
        query=query,
        category=category,
        first_layer_time=first_layer_time,
        second_layer_time=second_layer_time,
        results_count=len(results)
    )
    
    return results
```

**监控指标**：
- 第一层准确率：是否定位到正确类别
- 第二层召回率：是否找到相关结果
- 总体耗时：是否比单层快

---

### 实践4：处理第一层定位错误

```python
# 策略1：返回top-2类别，都检索
categories = first_layer_search(query, limit=2)
results = []
for category in categories:
    results.extend(second_layer_search(category, query))

# 策略2：设置置信度阈值
category, confidence = first_layer_search_with_score(query)
if confidence < 0.7:
    # 置信度低，回退到单层检索
    return single_layer_search(query)
else:
    return second_layer_search(category, query)
```

---

## ⚠️ 常见问题

### 问题1：第一层定位错误

**症状**：第一层定位到错误的类别，导致第二层找不到结果。

**原因**：
- 第一层的描述不够准确
- 查询和类别的语义差距大

**解决方案**：
```python
# 方案1：丰富第一层的描述
# ❌ 简单描述
summary = "2023年富豪榜"

# ✅ 详细描述
summary = "2023年世界十大富豪排行榜，包含姓名、财富、行业等信息"

# 方案2：使用多个描述
summaries = [
    "2023年富豪榜",
    "2023年世界首富排名",
    "2023年亿万富翁名单"
]
```

---

### 问题2：层次划分不合理

**症状**：某些类别的数据特别多，某些特别少。

**原因**：数据分布不均匀。

**解决方案**：
```python
# 动态调整层次
if category_size > 10000:
    # 数据太多，再分一层
    add_sub_layer(category)
elif category_size < 100:
    # 数据太少，合并到其他类别
    merge_category(category)
```

---

### 问题3：两层检索比单层还慢

**症状**：使用层次化索引后，检索速度反而变慢。

**原因**：
- 数据量太小，层次化的开销大于收益
- 两次检索的总时间超过一次检索

**解决方案**：
```python
# 只在数据量大时使用层次化
if total_documents > 10000:
    return hierarchical_search(query)
else:
    return single_layer_search(query)
```

**经验值**：
- 数据量 < 1000：不需要层次化
- 数据量 1000-10000：可选
- 数据量 > 10000：推荐层次化

---

### 问题4：存储成本增加

**症状**：需要存储两份索引，成本翻倍。

**解决方案**：
```python
# 第一层只存储摘要（小）
summary_index = {
    "table_name": "2023年",
    "vector": embed("2023年富豪榜")  # 只有一个向量
}

# 第二层存储详细数据（大）
details_index = {
    "table_name": "2023年",
    "content": "完整的表格数据...",
    "vector": embed("完整的表格数据...")
}

# 第一层的存储成本很小，可以忽略
```

---

## 🚀 进阶技巧

### 技巧1：动态层次调整

根据数据规模动态调整层次：

```python
def adaptive_hierarchical_search(query, data_size):
    if data_size < 1000:
        # 小数据：单层
        return single_layer_search(query)
    elif data_size < 100000:
        # 中等数据：两层
        return two_layer_search(query)
    else:
        # 大数据：三层
        return three_layer_search(query)
```

---

### 技巧2：混合层次策略

结合不同的层次化方法：

```python
# 第一层：按类别（双层索引）
category = search_category(query)

# 第二层：按时间（递归检索）
time_range = search_time_range(query, category)

# 第三层：精确检索
results = search_details(category, time_range, query)
```

---

### 技巧3：缓存第一层结果

```python
# 缓存常见查询的第一层结果
cache = {
    "2023年": "2023年富豪榜",
    "2022年": "2022年富豪榜",
    # ...
}

def search_with_cache(query):
    # 检查缓存
    for keyword, category in cache.items():
        if keyword in query:
            return second_layer_search(category, query)
    
    # 缓存未命中，正常检索
    return hierarchical_search(query)
```

---

## 📚 相关知识点

### 本目录未涉及但相关的技术

#### 1. 多级索引（Multi-level Index）

**概念**：超过两层的索引结构。

**示例**：
```
第一层：大类（技术、产品、市场）
第二层：子类（Python、Java、C++）
第三层：具体主题（基础语法、高级特性）
第四层：详细内容
```

**适用场景**：
- 超大规模数据（百万级以上）
- 有明确的多级分类体系

**注意**：层次太多会降低效率，一般不超过3层。

---

#### 2. 倒排索引（Inverted Index）

**概念**：从关键词到文档的映射。

**示例**：
```
关键词 → 文档列表
"Python" → [文档1, 文档5, 文档10]
"Java" → [文档2, 文档3, 文档8]
```

**优势**：
- 关键词检索速度极快
- 适合精确匹配

**参考**：BM25算法使用倒排索引

---

#### 3. 图索引（Graph Index）

**概念**：将文档和概念构建成图结构。

**示例**：
```
文档A → 概念1 → 文档B
       ↓
     概念2 → 文档C
```

**优势**：
- 可以发现文档间的关联
- 支持多跳检索

**适用场景**：
- 知识图谱
- 关联推荐

---

#### 4. 混合索引（Hybrid Index）

**概念**：结合多种索引方法。

**示例**：
```
层次化索引 + 向量索引 + 倒排索引

第一步：用层次化索引定位类别
第二步：用倒排索引过滤关键词
第三步：用向量索引语义匹配
```

**优势**：
- 结合多种方法的优势
- 提高检索精度和速度

---

#### 5. 自适应索引（Adaptive Index）

**概念**：根据查询模式动态调整索引结构。

**示例**：
```python
# 统计查询模式
if most_queries_about("2023年"):
    # 为2023年数据建立更细的索引
    optimize_index("2023年")
```

**优势**：
- 针对热点数据优化
- 提高常见查询的速度

---

## 📖 总结

### 核心要点

1. **核心思想**：先粗略定位，再精确查找

2. **三种方法**：
   - 双层索引：摘要+详情，适合多表/多文档
   - 递归检索：索引节点+查询引擎，适合结构化数据
   - 粗中有细：快速过滤+精确匹配，适合超大规模

3. **适用场景**：
   - 数据量大（>10000条）
   - 有明确分类
   - 需要快速响应

4. **关键参数**：
   - 层次数：2-3层
   - 第一层类别数：3-10个
   - 第一层limit：1-2

5. **性能提升**：
   - 检索速度：提升5-10倍
   - 检索精度：提升20-30%
   - 存储成本：增加10-20%

### 何时使用层次化索引？

```
✅ 使用层次化：
- 数据量 > 10000
- 有明确分类
- 查询集中在某些类别

❌ 不使用层次化：
- 数据量 < 1000
- 数据无法分类
- 查询分布均匀
```

### 下一步学习

- 学习 [03-构建多表示的索引](../03-构建多表示的索引/README.md)
- 实践：为自己的数据构建层次化索引
- 对比：测试单层 vs 双层的性能差异

---

*最后更新：2026-02-27*
