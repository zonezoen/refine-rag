# Milvus 向量数据库使用指南

## 快速启动

### 1. 启动 Milvus 服务

在 `rag/08-向量存储/` 目录下执行：

```bash
# 启动所有服务（etcd + minio + milvus）
docker compose up -d

# 查看服务状态
docker compose ps

# 查看日志
docker compose logs -f milvus-standalone
```

等待约 30-60 秒，直到看到类似输出：
```
milvus-standalone    | [INFO] Milvus Proxy successfully started
```

### 2. 验证服务

```bash
# 检查端口是否监听
lsof -i :19530

# 或使用 netstat
netstat -an | grep 19530
```

### 3. 运行示例代码

```bash
cd rag/08-向量存储/Milvus数据库/01-集合与实体
python 01-database.py
```

### 4. 停止服务

```bash
# 停止服务（保留数据）
docker compose stop

# 停止并删除容器（保留数据）
docker compose down

# 停止并删除所有数据
docker compose down -v
rm -rf milvus_data/
```

## 服务说明

| 服务 | 端口 | 说明 |
|------|------|------|
| milvus-standalone | 19530 | Milvus 主服务（gRPC） |
| milvus-standalone | 9091 | Milvus 监控端口 |
| minio | 9000 | MinIO 对象存储（API） |
| minio | 9001 | MinIO 控制台 |
| etcd | 2379 | etcd 元数据存储 |

## 开启密码认证

### 1. 修改 docker-compose.yml

在 `milvus-standalone` 服务的 `environment` 部分取消注释：

```yaml
environment:
  ETCD_ENDPOINTS: etcd:2379
  MINIO_ADDRESS: minio:9000
  COMMON_SECURITY_AUTHORIZATIONENABLED: "true"  # 开启认证
  COMMON_SECURITY_DEFAULTROOTPASSWORD: "MyPassword123"  # 设置密码
```

### 2. 重启服务

```bash
docker compose down
docker compose up -d
```

### 3. 修改 Python 代码

```python
from pymilvus import MilvusClient

client = MilvusClient(
    uri="http://localhost:19530",
    token="root:MyPassword123"  # 用户名:密码
)
```

### 4. 创建新用户（可选）

```python
from pymilvus import connections, utility

# 连接（使用 root 账号）
connections.connect(
    alias="default",
    host="localhost",
    port="19530",
    user="root",
    password="MyPassword123"
)

# 创建新用户
utility.create_user(user="myuser", password="MyUserPassword123")

# 授予角色
utility.create_role(role_name="myrole")
utility.grant_privilege(
    role_name="myrole",
    object_type="Collection",
    privilege="Search",
    object_name="*"
)
utility.add_user_to_role(username="myuser", role_name="myrole")
```

## 访问 MinIO 控制台

浏览器访问：http://localhost:9001

- 用户名：`minioadmin`
- 密码：`minioadmin`

## 常见问题

### 1. 端口被占用

```bash
# 查看端口占用
lsof -i :19530
lsof -i :9000
lsof -i :9001

# 修改 docker-compose.yml 中的端口映射
ports:
  - "19531:19530"  # 改为其他端口
```

### 2. 连接失败

检查步骤：
1. 确认 Docker 正在运行：`docker ps`
2. 确认 Milvus 容器状态：`docker compose ps`
3. 查看 Milvus 日志：`docker compose logs milvus-standalone`
4. 等待服务完全启动（约 30-60 秒）

### 3. 数据持久化

数据存储在 `milvus_data/` 目录：
- `milvus_data/etcd/` - 元数据
- `milvus_data/minio/` - 向量数据
- `milvus_data/milvus/` - Milvus 运行数据

### 4. 重置数据库

```bash
# 停止服务
docker compose down

# 删除数据
rm -rf milvus_data/

# 重新启动
docker compose up -d
```

## Python 客户端使用

### 安装

```bash
pip install pymilvus
```

### 基本连接

```python
from pymilvus import MilvusClient

# 连接到本地 Milvus
client = MilvusClient(uri="http://localhost:19530")

# 列出所有数据库
databases = client.list_databases()
print(databases)
```

## 版本信息

- Milvus: v2.5.4
- etcd: v3.5.5
- MinIO: RELEASE.2023-03-20T20-16-18Z
- pymilvus: 建议使用 2.4.x

## 更多资源

- [Milvus 官方文档](https://milvus.io/docs)
- [pymilvus API 文档](https://milvus.io/api-reference/pymilvus/v2.4.x/About.md)
