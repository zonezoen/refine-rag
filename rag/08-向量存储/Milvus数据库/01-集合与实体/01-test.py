import logging
from pymilvus import MilvusClient

# 隐藏 pymilvus 的 ERROR 日志
logging.getLogger("pymilvus").setLevel(logging.CRITICAL)

client = MilvusClient(uri="http://localhost:19530")

# 开头清除数据
client.drop_database(db_name="my_database")
client.create_database(db_name="my_database",properties=3)

db_list = client.list_databases()
print("当前所有数据库：", db_list)

des_default_db_info = client.describe_database(db_name="default")
print("默认数据库详情：", des_default_db_info)

client.use_database(db_name="my_database")

client.use_database(db_name="default")
client.drop_database(db_name="my_database")



db_list = client.list_databases()
print("当前所有数据库：", db_list)

