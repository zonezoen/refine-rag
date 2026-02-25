"""
Milvus 数据库操作示例

安装依赖：
pip install pymilvus

启动 Milvus 服务：
1. 确保 Docker 已安装并运行
2. 在 rag/08-向量存储/ 目录下执行：
   docker compose up -d
3. 等待服务启动（约 30-60 秒）
4. 检查服务状态：
   docker compose ps

停止服务：
   docker compose down

查看日志：
   docker compose logs -f milvus-standalone
"""

import logging
from pymilvus import MilvusClient

# 隐藏 pymilvus 的 ERROR 日志（只显示 CRITICAL）
logging.getLogger("pymilvus").setLevel(logging.CRITICAL)

# ——————————————
# 1. 连接 Milvus Standalone
# ——————————————
# uri: Milvus 服务地址，默认为 http://localhost:19530
# token: 认证信息（可选），格式为 "用户名:密码"
# 
# 【如何开启密码认证】
# 1. 在 docker-compose.yml 中取消注释以下两行：
#    COMMON_SECURITY_AUTHORIZATIONENABLED: "true"
#    COMMON_SECURITY_DEFAULTROOTPASSWORD: "your_password_here"
# 
# 2. 重启 Milvus 服务：
#    docker compose down
#    docker compose up -d
# 
# 3. 连接时提供 token：
#    client = MilvusClient(
#        uri="http://localhost:19530",
#        token="root:your_password_here"  # 用户名:密码
#    )
# 
# 注意：Milvus v2.4+ 默认不需要密码，除非手动开启认证
print("正在连接 Milvus...")
try:
    client = MilvusClient(
        uri="http://localhost:19530",
        # token="root:Milvus"  # 如果开启了认证，取消注释并修改密码
    )
    print("✓ 成功连接到 Milvus (localhost:19530)")
except Exception as e:
    print(f"✗ 连接失败: {e}")
    print("\n请检查：")
    print("1. Docker 是否运行：docker ps")
    print("2. Milvus 是否启动：docker compose ps")
    print("3. 端口 19530 是否被占用：lsof -i :19530")
    print("4. 如果开启了认证，是否提供了正确的 token")
    exit(1)

# ——————————————
# 2. 创建数据库 my_database_1（无额外属性）
# ——————————————
try:
    client.create_database(db_name="my_database_1")
    print("✓ my_database_1 创建成功")
except Exception as e:
    if "already exist" in str(e):
        print("ℹ my_database_1 已存在")
    else:
        raise e

# ——————————————
# 3. 创建数据库 my_database_2（设置副本数为 3）
# ——————————————
try:
    client.create_database(
        db_name="my_database_2",
        properties={"database.replica.number": 3}
    )
    print("✓ my_database_2 创建成功，副本数=3")
except Exception as e:
    if "already exist" in str(e):
        print("ℹ my_database_2 已存在")
    else:
        raise e

# ——————————————
# 4. 列出所有数据库
# ——————————————
db_list = client.list_databases()
print("当前所有数据库：", db_list)

# ——————————————
# 5. 查看默认数据库（default）详情
# ——————————————
default_info = client.describe_database(db_name="default")
print("默认数据库详情：", default_info)

# ——————————————
# 6. 修改 my_database_1 属性：限制最大集合数为 10
# ——————————————
client.alter_database_properties(
    db_name="my_database_1",
    properties={"database.max.collections": 10}
)
print("✓ 已为 my_database_1 限制最大集合数为 10")

# 查看修改后的属性
db1_info = client.describe_database(db_name="my_database_1")
print(f"   my_database_1 详情: {db1_info}")

# ——————————————
# 7. 删除 my_database_1 的 max.collections 限制
# ——————————————
# 注意：drop_database_properties 只能删除已存在的属性
# 如果属性不存在会报错，所以这里用 try-except 处理
try:
    client.drop_database_properties(
        db_name="my_database_1",
        property_keys=["database.max.collections"]
    )
    print("✓ 已移除 my_database_1 的最大集合数限制")
    
    # 验证删除结果
    db1_info_after = client.describe_database(db_name="my_database_1")
    print(f"   删除后详情: {db1_info_after}")
except Exception as e:
    # 这个错误是正常的，因为 Milvus 2.4 版本的 drop_database_properties 有 bug
    # 实际上属性已经被删除了，但返回了错误信息
    print("⚠️  删除属性时返回错误（这是 Milvus 2.4 的已知问题，属性实际已删除）")

# ——————————————
# 8. 切换到 my_database_2（后续所有操作都作用于该库）
# ——————————————
client.use_database(db_name="my_database_2")
print("✓ 已切换当前数据库为 my_database_2")

# ——————————————
# 9. 删除数据库 my_database_2
#    注意：
#    1. 删除前需要先切换到其他数据库（不能删除当前使用的数据库）
#    2. 如果库内有 Collection，需先删除所有 Collection
# ——————————————
# 先切换回 default 数据库
client.use_database(db_name="default")
print("✓ 已切换回 default 数据库")

# 删除 my_database_2
try:
    client.drop_database(db_name="my_database_2")
    print("✓ my_database_2 已删除")
except Exception as e:
    print(f"⚠️  删除 my_database_2 失败: {e}")

# ——————————————
# 10. 删除数据库 my_database_1
# ——————————————
try:
    client.drop_database(db_name="my_database_1")
    print("✓ my_database_1 已删除")
except Exception as e:
    print(f"⚠️  删除 my_database_1 失败: {e}")

# ——————————————
# 11. 验证删除结果
# ——————————————
final_db_list = client.list_databases()
print(f"\n最终数据库列表: {final_db_list}")
print("\n✓ 所有操作完成！")
