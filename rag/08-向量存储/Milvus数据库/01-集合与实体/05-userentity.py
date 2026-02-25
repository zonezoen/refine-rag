import logging
from pymilvus import MilvusClient, CollectionSchema, FieldSchema, DataType
import random
from langchain_community.embeddings import HuggingFaceEmbeddings  # 开源的文本嵌入模型

# 隐藏 pymilvus 的 ERROR 日志
logging.getLogger("pymilvus").setLevel(logging.CRITICAL)

# 连接到 Milvus
client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)
collection_name="user"
if client.has_collection("user"):
    client.drop_collection("user")

user_schema = CollectionSchema(
    fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="age", dtype=DataType.INT32),
        FieldSchema(name="email", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="phone", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="description", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name="description_text", dtype=DataType.VARCHAR, max_length=65535)
    ]
)

client.create_collection("user", schema=user_schema)
sun_description = "孙悟空是《西游记》中的主要角色之一，拥有七十二变的神通广大能力，手持金箍棒，能够上天入地，降妖除魔。他保护唐僧西天取经，一路上斩妖除魔，不畏艰险，是正义与勇敢的化身。"
zhu_description = "猪八戒是《西游记》中的重要角色，原为天蓬元帅，因犯错被贬下凡间，错投猪胎。他使用九齿钉耙作为武器，虽然有时贪吃懒惰，但也在取经路上发挥了重要作用，性格憨厚可爱。"


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # 多语言模型
    model_kwargs={'device': 'cpu'}  # 使用CPU运行，如果有GPU可以改为'cuda'
)
description_embedding = embeddings.embed_documents([sun_description, zhu_description])
data = [
    {
        "id": 1,
        "name": "孙悟空",
        "age": 18,
        "email": "zhangsan@example.com",
        "phone": "12345678901",
        "description": description_embedding[0],
        "description_text": sun_description
    },
    {
        "id": 2,
        "name": "猪八戒",
        "age": 19,
        "email": "lisi@example.com",
        "phone": "12345678902",
        "description": description_embedding[1],
        "description_text": zhu_description
    }
]

res = client.insert(collection_name=collection_name, data=data)
print(res)

index_params = MilvusClient.prepare_index_params()
index_params.add_index(
    field_name="description",
    metric_type="COSINE",
    index_type="FLAT",
    index_name="description_index"
)

client.create_index(
    collection_name=collection_name,
    index_params=index_params,
    sync=True
)

client.load_collection(collection_name=collection_name)

query_text = "请给出一个关于孙悟空的描述"
query_embedding = embeddings.embed_query(query_text)
results = client.search(collection_name=collection_name,
              data=[query_embedding],
              limit=2,
              output_fields=["name", "age", "email", "phone", "description_text"],
              search_params={"metric_type": "COSINE", "params": {"nprobe": 10}}
              )

for rs in results:
    for item in rs:
        print("   搜索结果:")
        print("   姓名:", item["id"])
        print("   年龄:", item)


info = client.describe_collection(collection_name=collection_name)
print("Collection详情：", info)





