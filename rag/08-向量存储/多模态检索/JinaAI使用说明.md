# Jina AI 多模态检索系统使用说明

## 为什么选择 Jina AI？

相比原来的 Visual-BGE 方案，Jina AI 有以下优势：

✅ **无需下载模型** - 直接调用 API，省去复杂的安装过程  
✅ **真正的多模态** - 图像和文本在同一向量空间  
✅ **国内可访问** - 无需科学上网  
✅ **有免费额度** - 100万 tokens/月  
✅ **中文支持好** - 对中文文本友好  

## 快速开始

### 1. 安装依赖

```bash
pip install pymilvus requests pillow python-dotenv opencv-python numpy tqdm
```

### 2. 启动 Milvus

```bash
# 进入向量存储目录
cd rag/08-向量存储

# 启动 Milvus Docker 服务
docker-compose up -d

# 检查服务状态
docker-compose ps
```

### 3. 获取 API Key

1. 访问 https://jina.ai/
2. 注册账号（支持 GitHub 登录）
3. 在 Dashboard 中创建 API Key
4. 复制 API Key

### 4. 配置环境变量

在项目根目录的 `.env` 文件中添加：

```bash
JINA_API_KEY=your_api_key_here
```

### 5. 运行程序

```bash
python Milvus+JinaAI多模态检索.py
```

## 程序流程

1. **初始化编码器** - 连接 Jina AI API
2. **加载数据集** - 读取图像和元数据
3. **生成嵌入向量** - 调用 API 编码所有图像
4. **创建向量数据库** - 在 Milvus 中建立索引
5. **插入数据** - 存储向量和元数据
6. **执行检索** - 使用图像+文本查询
7. **显示结果** - 输出最相似的图像
8. **可视化** - 生成结果网格图

## 与原方案对比

| 特性 | Visual-BGE | Jina AI |
|------|-----------|---------|
| 安装难度 | ⭐⭐⭐⭐⭐ 需要克隆仓库、安装依赖、下载模型 | ⭐ 只需 pip install |
| 模型大小 | 几个 GB | 无需下载 |
| 运行环境 | 需要 GPU（推荐） | 任何环境 |
| 网络要求 | 下载模型时需要 | 调用 API 时需要 |
| 成本 | 免费 | 有免费额度 |
| 速度 | 本地快 | 取决于网络 |

## 费用说明

Jina AI 定价：
- **免费额度**：100万 tokens/月
- **付费**：$0.02/千次调用
- 图片按分辨率计费

对于小规模测试和开发，免费额度完全够用。

## 常见问题

### Q1: 提示 "Insufficient account balance" 怎么办？

这表示 Jina AI 的免费额度已用完。解决方案：

**方案1：等待重置**
- 免费额度每月重置
- 100万 tokens/月

**方案2：充值**
- 访问 https://jina.ai/api-dashboard/key-manager
- 充值金额：$0.02/千次调用

**方案3：创建新账号**
- 使用其他邮箱注册新账号
- 获取新的 API Key

**方案4：改用本地模型**
- 使用原来的 Visual-BGE 方案
- 或使用 CLIP 本地模型

### Q2: 提示 "无法连接到 Docker Milvus" 怎么办？

检查 Milvus 服务状态：

```bash
# 检查容器是否运行
docker-compose ps

# 查看日志
docker-compose logs -f milvus-standalone

# 重启服务
docker-compose restart

# 如果还不行，完全重启
docker-compose down
docker-compose up -d
```

### Q3: 图片路径错误怎么办？

检查：
1. 图片文件是否存在
2. 路径是否正确（相对路径）
3. metadata.json 中的路径配置

### Q4: API 调用失败怎么办？

检查：
1. API Key 是否正确配置在 .env 文件中
2. 网络连接是否正常
3. 是否超出免费额度

### Q5: 编码速度慢怎么办？

- API 调用受网络影响
- 可以考虑批量编码优化
- 如果需要高频调用，建议使用本地模型

### Q6: 如何提高检索精度？

1. 优化查询文本描述（更具体、更详细）
2. 调整搜索参数（nprobe、radius 等）
3. 增加数据集规模
4. 使用更高质量的图片

## 扩展应用

这个系统可以用于：

- 🎮 游戏资产管理
- 🖼️ 图片搜索引擎
- 📱 内容推荐系统
- 🎨 设计素材库
- 📚 多媒体知识库

## 下一步

- 尝试不同的查询组合
- 调整搜索参数优化结果
- 添加更多图像到数据库
- 集成到你的应用中
