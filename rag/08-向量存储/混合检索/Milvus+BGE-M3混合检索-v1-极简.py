"""
Milvus + BGE-M3 混合检索系统 - 极简版

这个程序展示了如何使用BGE-M3模型和Milvus向量数据库构建混合检索系统：
1. BGE-M3可以同时生成密集向量和稀疏向量
2. 密集向量：捕获语义信息，适合语义相似度搜索
3. 稀疏向量：捕获关键词信息，适合精确匹配
4. 混合检索：结合两种向量的优势，提供更准确的搜索结果

应用场景：
- 文档检索：既要语义理解又要关键词匹配
- 问答系统：提高答案检索的准确性
- 推荐系统：平衡内容相似性和用户偏好
"""

# ==================== 导入必要的库 ====================
import json  # 用于处理JSON数据文件
import time  # 用于添加延时，确保操作稳定性
from milvus_model.hybrid import BGEM3EmbeddingFunction  # BGE-M3混合嵌入模型
from pymilvus import (  # Milvus向量数据库相关组件
    connections,        # 数据库连接
    utility,           # 工具函数
    FieldSchema,       # 字段模式定义
    CollectionSchema,  # 集合模式定义
    DataType,          # 数据类型
    Collection,        # 集合操作
    AnnSearchRequest,  # 近似最近邻搜索请求
    WeightedRanker     # 加权排序器（用于混合搜索）
)
from pymilvus.exceptions import MilvusException  # Milvus异常处理
import scipy.sparse  # 稀疏矩阵处理库，BGE-M3生成稀疏向量时需要

# ==================== 配置参数 ====================
# 集中配置所有参数，方便修改和调试
DATA_PATH = "../../99-doc-data/灭神纪/战斗场景.json"  # Linux路径
# DATA_PATH = r"F:\work\rag\rag-in-action-master\rag-in-action-master\90-文档-Data\灭神纪\战斗场景.json"  # Windows路径
COLLECTION_NAME = "wukong_hybrid_v4"  # Milvus集合名称，使用v4避免与旧数据冲突
MILVUS_URI = "./wukong_v4.db"         # Milvus数据库文件路径（本地SQLite模式）
BATCH_SIZE = 50                       # 批量插入数据的大小，可根据内存情况调整
DEVICE = "cpu"                        # 计算设备：cpu或cuda（如果有GPU）

print("脚本开始执行...")

# ==================== 1. 数据加载和预处理 ====================
print(f"1. 正在从 {DATA_PATH} 加载数据...")

# 安全地加载JSON数据文件
try:
    # 使用UTF-8编码打开文件，确保中文字符正确读取
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        dataset = json.load(f)  # 解析JSON数据
except FileNotFoundError:
    # 文件不存在时的错误处理
    print(f"错误: 数据文件 {DATA_PATH} 未找到。请检查路径。")
    exit()
except json.JSONDecodeError:
    # JSON格式错误时的错误处理
    print(f"错误: 数据文件 {DATA_PATH} JSON 格式错误。")
    exit()

# 初始化存储列表
docs = []      # 存储处理后的文档文本
metadata = []  # 存储原始元数据

# 遍历数据集中的每个条目
for item in dataset.get('data', []):  # 使用.get()方法避免'data'键不存在的错误
    # 收集文本片段：从不同字段提取文本信息
    text_parts = [
        item.get('title', ''),        # 标题
        item.get('description', '')   # 描述
    ]
    
    # 处理战斗细节信息
    if 'combat_details' in item and isinstance(item['combat_details'], dict):
        # 添加战斗风格信息
        text_parts.extend(item['combat_details'].get('combat_style', []))
        # 添加使用的技能信息
        text_parts.extend(item['combat_details'].get('abilities_used', []))
    
    # 处理场景信息
    if 'scene_info' in item and isinstance(item['scene_info'], dict):
        text_parts.extend([
            item['scene_info'].get('location', ''),     # 地点
            item['scene_info'].get('environment', ''),  # 环境
            item['scene_info'].get('time_of_day', '')   # 时间
        ])
    
    # 文本预处理：
    # 1. 过滤掉None值和空字符串
    # 2. 转换为字符串并去除首尾空格
    # 3. 用空格连接所有文本片段
    processed_text = ' '.join(filter(None, [str(part).strip() for part in text_parts if part]))
    
    # 存储处理后的文档和原始元数据
    docs.append(processed_text)
    metadata.append(item)

# 验证数据加载结果
if not docs:
    print("错误: 未能从数据文件中加载任何文档。请检查文件内容和结构。")
    exit()

print(f"数据加载完成，共 {len(docs)} 条文档。")

# ==================== 2. 向量生成（BGE-M3混合嵌入） ====================
print("2. 正在生成向量...")

try:
    # 初始化BGE-M3嵌入函数
    # BGE-M3是一个多功能嵌入模型，可以同时生成：
    # 1. 密集向量（dense）：用于语义相似度搜索
    # 2. 稀疏向量（sparse）：用于关键词匹配
    # 3. 多向量（multi-vector）：用于更精细的表示
    ef = BGEM3EmbeddingFunction(
        use_fp16=False,  # 不使用半精度浮点数，确保精度
        device=DEVICE    # 指定计算设备（CPU或CUDA）
    )
    
    # 准备要嵌入的文档
    docs_to_embed = docs
    print(f"将为 {len(docs_to_embed)} 条文档生成向量...")
    
    # 生成嵌入向量
    # ef()函数会返回一个字典，包含不同类型的向量表示
    docs_embeddings = ef(docs_to_embed)
    
    print("向量生成完成。")
    print(f"  密集向量维度: {ef.dim['dense']}")  # 密集向量的维度（通常是1024）
    
    # 检查稀疏向量生成情况
    if "sparse" in docs_embeddings and docs_embeddings["sparse"].shape[0] > 0:
        print(f"  稀疏向量类型 (整体): {type(docs_embeddings['sparse'])}")
        
        # 获取第一个稀疏向量进行检查
        # 稀疏向量通常以scipy.sparse格式存储，需要特殊处理
        first_sparse_vector_row_obj = docs_embeddings['sparse'][0]  # 获取第一行稀疏向量
        print(f"  第一个稀疏向量 (行对象类型): {type(first_sparse_vector_row_obj)}")
        print(f"  第一个稀疏向量 (行对象形状): {first_sparse_vector_row_obj.shape}")
        
        # 根据稀疏矩阵的类型提取数据
        # COO格式（Coordinate format）使用.col和.data属性
        if hasattr(first_sparse_vector_row_obj, 'col') and hasattr(first_sparse_vector_row_obj, 'data'):
            print(f"  第一个稀疏向量 (部分列索引/col): {first_sparse_vector_row_obj.col[:5]}")
            print(f"  第一个稀疏向量 (部分数据/data): {first_sparse_vector_row_obj.data[:5]}")
        # CSR格式（Compressed Sparse Row）使用.indices和.data属性
        elif hasattr(first_sparse_vector_row_obj, 'indices') and hasattr(first_sparse_vector_row_obj, 'data'):
            print(f"  第一个稀疏向量 (部分索引/indices): {first_sparse_vector_row_obj.indices[:5]}")
            print(f"  第一个稀疏向量 (部分数据/data): {first_sparse_vector_row_obj.data[:5]}")
        else:
            print("  无法直接获取第一个稀疏向量的列索引和数据属性。")
    else:
        print("警告: 未生成稀疏向量或稀疏向量为空。")

except Exception as e:
    print(f"生成向量时发生错误: {e}")
    exit()

# ==================== 3. 连接Milvus向量数据库 ====================
print(f"3. 正在连接 Milvus (URI: {MILVUS_URI})...")
try:
    # 连接到Milvus数据库
    # 这里使用本地SQLite模式，数据存储在本地文件中
    connections.connect(uri=MILVUS_URI)
    print("成功连接到 Milvus。")
except MilvusException as e:
    print(f"连接 Milvus 失败: {e}")
    exit()

# ==================== 4. 创建Milvus集合（类似数据库表） ====================
print(f"4. 正在准备集合 '{COLLECTION_NAME}'...")

# 定义集合的字段结构
# 每个字段都有名称、数据类型和其他属性
fields = [
    # 主键字段：自动生成的唯一标识符
    FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
    
    # 文本内容字段：存储完整的文档文本
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    
    # 元数据字段：存储文档的各种属性
    FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100),           # 文档ID
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),        # 标题
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=128),     # 类别
    FieldSchema(name="location", dtype=DataType.VARCHAR, max_length=256),     # 地点
    FieldSchema(name="environment", dtype=DataType.VARCHAR, max_length=128),  # 环境
    
    # 向量字段：存储BGE-M3生成的两种向量
    FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),    # 稀疏向量（关键词匹配）
    FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=ef.dim["dense"])  # 密集向量（语义理解）
]

# 创建集合模式（Schema）
schema = CollectionSchema(fields, description="Wukong Hybrid Search Collection v4")

try:
    # 如果集合已存在，先删除（避免冲突）
    if utility.has_collection(COLLECTION_NAME):
        print(f"集合 '{COLLECTION_NAME}' 已存在，正在删除...")
        utility.drop_collection(COLLECTION_NAME)
        print(f"集合 '{COLLECTION_NAME}' 删除成功。")
        time.sleep(1)  # 等待删除操作完成

    # 创建新集合
    print(f"正在创建集合 '{COLLECTION_NAME}'...")
    collection = Collection(
        name=COLLECTION_NAME, 
        schema=schema, 
        consistency_level="Strong"  # 强一致性：确保数据写入后立即可读
    )
    print(f"集合 '{COLLECTION_NAME}' 创建成功。")

    # 为稀疏向量创建索引
    # SPARSE_INVERTED_INDEX：专门用于稀疏向量的倒排索引
    # IP（Inner Product）：内积相似度度量
    print("正在为 sparse_vector 创建索引 (SPARSE_INVERTED_INDEX, IP)...")
    collection.create_index("sparse_vector", {
        "index_type": "SPARSE_INVERTED_INDEX", 
        "metric_type": "IP"
    })
    print("sparse_vector 索引创建成功。")
    time.sleep(0.5)

    # 为密集向量创建索引
    # AUTOINDEX：自动选择最适合的索引类型
    # IP：内积相似度度量（与稀疏向量保持一致）
    print("正在为 dense_vector 创建索引 (AUTOINDEX, IP)...")
    collection.create_index("dense_vector", {
        "index_type": "AUTOINDEX", 
        "metric_type": "IP"
    })
    print("dense_vector 索引创建成功。")
    time.sleep(0.5)

    # 加载集合到内存中，使其可以进行搜索
    print(f"正在加载集合 '{COLLECTION_NAME}'...")
    collection.load()
    print(f"集合 '{COLLECTION_NAME}' 加载成功。")

except MilvusException as e:
    print(f"创建或加载集合/索引时发生 Milvus 错误: {e}")
    exit()
except Exception as e:
    print(f"创建或加载集合/索引时发生未知错误: {e}")
    exit()

# ==================== 5. 批量插入数据到Milvus ====================
print("5. 正在准备插入数据...")
num_docs_to_insert = len(docs_to_embed)  # 要插入的文档总数

try:
    # 分批处理数据，避免内存溢出和提高插入效率
    for i in range(0, num_docs_to_insert, BATCH_SIZE):
        # 计算当前批次的结束索引
        end_idx = min(i + BATCH_SIZE, num_docs_to_insert)
        batch_data = []  # 存储当前批次的数据
        print(f"  正在准备批次 {i // BATCH_SIZE + 1} (索引 {i} 到 {end_idx - 1})...")

        # 处理当前批次中的每个文档
        for j in range(i, end_idx):
            item_metadata = metadata[j]  # 获取当前文档的元数据

            # ========== 关键步骤：转换稀疏向量格式 ==========
            # BGE-M3生成的稀疏向量是scipy.sparse格式，需要转换为Milvus接受的字典格式
            # 从稀疏矩阵中提取单行数据
            sparse_row_obj = docs_embeddings["sparse"][j]
            
            # 根据稀疏矩阵的具体类型进行转换
            # COO格式（Coordinate format）：使用.col和.data属性
            if hasattr(sparse_row_obj, 'col') and hasattr(sparse_row_obj, 'data'):
                # 将稀疏向量转换为字典格式：{索引: 值}
                milvus_sparse_vector = {
                    int(idx_col): float(val) 
                    for idx_col, val in zip(sparse_row_obj.col, sparse_row_obj.data)
                }
            # CSR格式（Compressed Sparse Row）：使用.indices和.data属性
            elif hasattr(sparse_row_obj, 'indices') and hasattr(sparse_row_obj, 'data'):
                milvus_sparse_vector = {
                    int(idx_col): float(val) 
                    for idx_col, val in zip(sparse_row_obj.indices, sparse_row_obj.data)
                }
            else:
                # 无法识别的稀疏向量格式，跳过此条记录
                print(f"警告: 无法识别的稀疏行对象类型 {type(sparse_row_obj)} 在索引 {j}。跳过此条。")
                continue

            # 处理文档文本：确保不超过字段长度限制
            doc_text = docs_to_embed[j]
            if len(doc_text) > 65530:  # VARCHAR字段最大长度限制
                doc_text = doc_text[:65530]  # 截断过长的文本

            # 处理标题：确保不超过字段长度限制
            title_text = item_metadata.get("title", "N/A")
            if len(title_text) > 500:
                title_text = title_text[:500]

            # 构建要插入的数据记录
            batch_data.append({
                "text": doc_text,  # 文档文本
                "id": str(item_metadata.get("id", f"unknown_id_{j}")),  # 文档ID
                "title": title_text,  # 标题
                "category": item_metadata.get("category", "N/A"),  # 类别
                "location": item_metadata.get("scene_info", {}).get("location", "N/A"),  # 地点
                "environment": item_metadata.get("scene_info", {}).get("environment", "N/A"),  # 环境
                "sparse_vector": milvus_sparse_vector,  # 稀疏向量（字典格式）
                "dense_vector": docs_embeddings["dense"][j].tolist()  # 密集向量（列表格式）
            })
        
        # 检查批次是否为空（所有稀疏向量都无法处理的情况）
        if not batch_data:
            print(f"  批次 {i // BATCH_SIZE + 1} 为空，跳过插入。")
            continue

        # 执行批量插入
        print(f"  正在插入批次 {i // BATCH_SIZE + 1} ({len(batch_data)} 条记录)...")
        insert_result = collection.insert(batch_data)
        print(f"  批次 {i // BATCH_SIZE + 1} 插入成功, 主键: {insert_result.primary_keys[:5]}...")
        
        # 强制刷新，确保数据持久化到磁盘
        collection.flush()
        print(f"  批次 {i // BATCH_SIZE + 1} flush 完成。")
        time.sleep(0.5)  # 短暂延时，避免过快的操作

    # 输出最终插入结果
    print(f"所有数据插入完成。总共 {collection.num_entities} 条实体。")

except MilvusException as e:
    # Milvus相关错误处理
    print(f"插入数据时发生 Milvus 错误: {e}")
    if 'batch_data' in locals() and batch_data:
        print("问题批次的第一条数据（部分）:")
        print(f"  Text: {batch_data[0]['text'][:100]}...")
        print(f"  ID: {batch_data[0]['id']}")
        print(f"  Title: {batch_data[0]['title']}")
    exit()
except Exception as e:
    # 其他未知错误处理
    print(f"插入数据时发生未知错误: {e}")
    if 'batch_data' in locals() and batch_data:
        print("问题批次的第一条数据（部分）:")
        print(f"  Text: {batch_data[0]['text'][:100]}...")
    exit()


# ==================== 6. 混合搜索函数定义 ====================
def hybrid_search(query, category=None, environment=None, limit=5, weights=None):
    """
    执行混合搜索的核心函数
    
    混合搜索的工作原理：
    1. 将查询文本同时转换为密集向量和稀疏向量
    2. 分别在两个向量空间中进行搜索
    3. 使用加权排序器合并两种搜索结果
    4. 返回综合排序后的最终结果
    
    参数:
        query: 查询文本
        category: 类别过滤条件（可选）
        environment: 环境过滤条件（可选）
        limit: 返回结果数量
        weights: 向量权重，控制密集向量和稀疏向量的重要性
    
    返回:
        处理后的搜索结果列表
    """
    # 设置默认权重：稀疏向量和密集向量各占50%
    if weights is None:
        weights = {"sparse": 0.5, "dense": 0.5} 

    print(f"\n6. 执行混合搜索: '{query}'")
    print(f"   Category: {category}, Environment: {environment}, Limit: {limit}, Weights: {weights}")
    
    try:
        # ========== 步骤1：生成查询向量 ==========
        # 使用BGE-M3将查询文本转换为密集向量和稀疏向量
        query_embeddings = ef([query])
        
        # ========== 步骤2：构建过滤条件 ==========
        # 根据用户提供的过滤条件构建Milvus查询表达式
        conditions = []
        if category:
            conditions.append(f'category == "{category}"')  # 类别过滤
        if environment:
            conditions.append(f'environment == "{environment}"')  # 环境过滤
        
        # 使用AND逻辑连接多个条件
        expr = " && ".join(conditions) if conditions else None
        print(f"   过滤表达式: {expr}")
        
        # ========== 步骤3：准备搜索参数 ==========
        # 为密集向量和稀疏向量分别准备搜索参数
        search_params_dense = {"metric_type": "IP", "params": {}}   # 内积相似度
        search_params_sparse = {"metric_type": "IP", "params": {}}  # 内积相似度

        # 如果有过滤条件，添加到搜索参数中
        if expr:
            search_params_dense["expr"] = expr
            search_params_sparse["expr"] = expr
        
        # ========== 步骤4：创建密集向量搜索请求 ==========
        dense_req = AnnSearchRequest(
            data=[query_embeddings["dense"][0].tolist()],  # 密集向量数据
            anns_field="dense_vector",                     # 搜索字段
            param=search_params_dense,                     # 搜索参数
            limit=limit                                    # 结果数量
        )
        
        # ========== 步骤5：处理稀疏向量格式转换 ==========
        # 将scipy.sparse格式转换为Milvus接受的字典格式
        query_sparse_row_obj = query_embeddings["sparse"][0]  # 获取查询的稀疏向量
        
        # 根据稀疏向量的具体格式进行转换
        if hasattr(query_sparse_row_obj, 'col') and hasattr(query_sparse_row_obj, 'data'):
            # COO格式：使用.col和.data属性
            query_milvus_sparse_vector = {
                int(idx): float(val) 
                for idx, val in zip(query_sparse_row_obj.col, query_sparse_row_obj.data)
            }
        elif hasattr(query_sparse_row_obj, 'indices') and hasattr(query_sparse_row_obj, 'data'):
            # CSR格式：使用.indices和.data属性
            query_milvus_sparse_vector = {
                int(idx): float(val) 
                for idx, val in zip(query_sparse_row_obj.indices, query_sparse_row_obj.data)
            }
        else:
            print(f"错误: 无法识别的查询稀疏向量类型 {type(query_sparse_row_obj)}。")
            return []

        # ========== 步骤6：创建稀疏向量搜索请求 ==========
        sparse_req = AnnSearchRequest(
            data=[query_milvus_sparse_vector],  # 稀疏向量数据（字典格式）
            anns_field="sparse_vector",         # 搜索字段
            param=search_params_sparse,         # 搜索参数
            limit=limit                         # 结果数量
        )
        
        # ========== 步骤7：创建加权排序器 ==========
        # WeightedRanker用于合并两种搜索结果
        # 第一个参数是稀疏向量权重，第二个参数是密集向量权重
        rerank = WeightedRanker(weights["sparse"], weights["dense"])
        
        # ========== 步骤8：执行混合搜索 ==========
        print("   发送混合搜索请求到 Milvus...")
        results = collection.hybrid_search(
            reqs=[sparse_req, dense_req],  # 搜索请求列表：[稀疏向量请求, 密集向量请求]
            rerank=rerank,                 # 重排序器
            limit=limit,                   # 最终结果数量
            output_fields=[                # 需要返回的字段
                "text", "id", "title", "category", 
                "location", "environment", "pk"
            ]
        )
        
        # ========== 步骤9：处理搜索结果 ==========
        print("   搜索完成。结果:")
        if not results or not results[0]:
            print("   未找到结果。")
            return []

        # 格式化搜索结果，便于后续处理和显示
        processed_results = []
        for hit in results[0]:
            processed_results.append({
                "id": hit.entity.get("id"),                              # 文档ID
                "pk": hit.id,                                           # 主键
                "title": hit.entity.get("title"),                      # 标题
                "text_preview": hit.entity.get("text", "")[:200] + "...",  # 文本预览（前200字符）
                "category": hit.entity.get("category"),                # 类别
                "location": hit.entity.get("location"),                # 地点
                "environment": hit.entity.get("environment"),          # 环境
                "distance": hit.distance                                # 相似度分数（越小越相似）
            })
        return processed_results

    except MilvusException as e:
        print(f"混合搜索时发生 Milvus 错误: {e}")
        return []
    except Exception as e:
        print(f"混合搜索时发生未知错误: {e}")
        return []

# ==================== 7. 示例搜索演示 ====================
# 检查集合中是否有数据
if collection.num_entities > 0:
    print("\n开始示例搜索...")
    
    # ========== 示例1：带类别过滤的搜索 ==========
    print("\n=== 示例1：搜索孙悟空的战斗技巧（类别过滤：神魔大战） ===")
    search_results = hybrid_search(
        query="孙悟空的战斗技巧",  # 查询文本
        category="神魔大战",        # 类别过滤条件
        limit=3                   # 返回3个结果
    )
    
    # 显示搜索结果
    if search_results:
        for res in search_results:
            print(f"  - PK: {res['pk']}, Title: {res['title']}, Distance: {res['distance']:.4f}")
            print(f"    Category: {res['category']}, Location: {res['location']}")
            print(f"    Preview: {res['text_preview']}\n")
    else:
        print("  未找到相关结果。")
    
    # ========== 示例2：带环境过滤的搜索 ==========
    print("\n=== 示例2：搜索火焰山的战斗（环境过滤：火山） ===")
    search_results_filtered = hybrid_search(
        query="火焰山的战斗",      # 查询文本
        environment="火山",       # 环境过滤条件
        limit=2                  # 返回2个结果
    )
    
    # 显示搜索结果
    if search_results_filtered:
        for res in search_results_filtered:
            print(f"  - PK: {res['pk']}, Title: {res['title']}, Distance: {res['distance']:.4f}")
            print(f"    Category: {res['category']}, Location: {res['location']}, Environment: {res['environment']}")
            print(f"    Preview: {res['text_preview']}\n")
    else:
        print("  未找到相关结果。")
        
    # ========== 示例3：调整权重的搜索 ==========
    print("\n=== 示例3：调整权重的搜索（更重视关键词匹配） ===")
    search_results_weighted = hybrid_search(
        query="悟空使用金箍棒",
        weights={"sparse": 0.8, "dense": 0.2},  # 稀疏向量权重80%，密集向量权重20%
        limit=2
    )
    
    if search_results_weighted:
        for res in search_results_weighted:
            print(f"  - PK: {res['pk']}, Title: {res['title']}, Distance: {res['distance']:.4f}")
            print(f"    Preview: {res['text_preview']}\n")
    else:
        print("  未找到相关结果。")
        
else:
    print("\n集合中没有实体，跳过示例搜索。")

print("\n脚本执行完毕。")

# ==================== 程序总结 ====================
"""
这个混合检索系统的完整流程总结：

1. 数据准备阶段：
   - 从JSON文件加载战斗场景数据
   - 提取和合并不同字段的文本信息
   - 预处理文本数据

2. 向量生成阶段：
   - 使用BGE-M3模型生成密集向量和稀疏向量
   - 密集向量：捕获语义信息，维度通常为1024
   - 稀疏向量：捕获关键词信息，以字典格式存储

3. 数据库构建阶段：
   - 连接Milvus向量数据库
   - 创建集合和索引（稀疏向量用倒排索引，密集向量用自动索引）
   - 批量插入数据，处理稀疏向量格式转换

4. 混合搜索阶段：
   - 将查询同时转换为两种向量
   - 分别在两个向量空间中搜索
   - 使用加权排序器合并结果
   - 支持元数据过滤和权重调整

优势：
- 结合了关键词匹配的精确性和语义搜索的智能性
- 支持灵活的过滤条件和权重调整
- 适用于需要高精度检索的应用场景

应用场景：
- 文档检索系统：既要匹配关键词又要理解语义
- 智能问答系统：提高答案检索的准确性
- 内容推荐系统：平衡精确匹配和相关推荐
- 知识库搜索：支持复杂查询和多维度过滤
"""

