# Milvus 使用说明

## 当前代码使用的连接方式

代码已修改为连接到标准 Milvus 服务器：
```python
client = MilvusClient(uri="http://localhost:19530")
```

## 如何启动 Milvus 服务

### 方法1：使用项目中的 Docker Compose（推荐）

在终端中运行：

```bash
# 进入向量存储目录
cd rag/08-向量存储

# 启动 Milvus 服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f milvus-standalone
```

启动成功后，Milvus 会在 `localhost:19530` 运行。

### 方法2：停止 Milvus 服务

```bash
cd rag/08-向量存储
docker-compose down
```

## 如果遇到连接错误

### 错误信息：
```
MilvusException: (code=2, message=Fail connecting to server on localhost:19530, 
illegal connection params or server unavailable)
```

### 解决方案：

1. **检查 Milvus 是否运行**：
   ```bash
   docker ps | grep milvus
   ```
   应该看到 `milvus-standalone` 容器在运行。

2. **检查端口是否被占用**：
   ```bash
   lsof -i :19530
   ```

3. **重启 Milvus 服务**：
   ```bash
   cd rag/08-向量存储
   docker-compose restart
   ```

4. **查看 Milvus 日志**：
   ```bash
   cd rag/08-向量存储
   docker-compose logs milvus-standalone
   ```

## 替代方案：使用 Milvus Lite（本地文件模式）

如果你不想启动 Docker 服务，可以使用 Milvus Lite（本地文件模式）：

### 修改连接方式：

```python
# 从标准 Milvus 服务器
client = MilvusClient(uri="http://localhost:19530")

# 改为本地文件模式
client = MilvusClient("./milvus_local.db")
```

### 优缺点对比：

| 特性 | 标准 Milvus | Milvus Lite |
|------|------------|-------------|
| 性能 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| 功能 | 完整 | 基础功能 |
| 部署 | 需要 Docker | 无需 Docker |
| 数据持久化 | 独立存储 | 本地文件 |
| 适用场景 | 生产环境 | 开发测试 |

## 推荐配置

### 开发环境：
- 使用标准 Milvus（通过 Docker）
- 与生产环境保持一致
- 便于调试和测试

### 快速测试：
- 可以使用 Milvus Lite
- 无需启动额外服务
- 适合快速验证代码

## 数据存储位置

### 标准 Milvus：
```
rag/08-向量存储/milvus_data/
├── etcd/       # 元数据
├── minio/      # 对象存储
└── milvus/     # Milvus 数据
```

### Milvus Lite：
```
当前目录下的 .db 文件
例如：richman_bge_m3_v2.db
```

## 常见问题

### Q1: 如何清空 Milvus 数据？

**标准 Milvus**：
```bash
cd rag/08-向量存储
docker-compose down -v  # 删除数据卷
docker-compose up -d    # 重新启动
```

**Milvus Lite**：
```bash
rm *.db  # 删除 .db 文件
```

### Q2: 如何查看 Milvus 中的集合？

```python
from pymilvus import MilvusClient

client = MilvusClient(uri="http://localhost:19530")
collections = client.list_collections()
print(collections)
```

### Q3: 端口 19530 被占用怎么办？

修改 `docker-compose.yml` 中的端口映射：
```yaml
ports:
  - "19531:19530"  # 改为其他端口
```

然后修改代码：
```python
client = MilvusClient(uri="http://localhost:19531")
```

---

*最后更新：2026-02-27*
