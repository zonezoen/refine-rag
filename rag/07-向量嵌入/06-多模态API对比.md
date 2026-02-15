# 国内多模态 API 服务对比

## 支持多模态的国内 API

### 1. 阿里云 - 通义千问（Qwen）

**支持功能**：
- ✅ 图文理解（VL 模型）
- ✅ 图片描述生成
- ✅ 视觉问答
- ❌ 不支持多模态 embedding（只能理解，不能生成向量）

**模型**：
- `qwen-vl-plus`：视觉理解模型
- `qwen-vl-max`：高级视觉模型

**价格**：
- 按 token 计费
- 图片按分辨率计费

**适用场景**：
- 图片内容理解
- 视觉问答
- 图片描述生成

---

### 2. DeepSeek

**支持功能**：
- ❌ 目前不支持多模态
- ✅ 只支持文本

**说明**：
- DeepSeek 目前只有文本模型
- 未来可能会推出多模态版本

---

### 3. 智谱 AI - GLM-4V

**支持功能**：
- ✅ 图文理解
- ✅ 图片描述
- ✅ 视觉问答
- ❌ 不支持多模态 embedding

**模型**：
- `glm-4v`：多模态理解模型

**价格**：
- 按 token 计费

---

### 4. 百度 - 文心一言

**支持功能**：
- ✅ 图文理解
- ✅ 图片生成
- ❌ 不支持多模态 embedding

**模型**：
- `ERNIE-Bot-4`：支持图片输入

---

### 5. Minimax

**支持功能**：
- ✅ 图文理解
- ❌ 不支持多模态 embedding

---

### 6. 火山引擎 - 豆包（字节跳动）

**支持功能**：
- ✅ 图文理解
- ❌ 不支持多模态 embedding

---

## 重要发现

### ⚠️ 国内 API 的限制

**大多数国内多模态 API 只支持**：
- ✅ 图片理解（输入图片，输出文字描述）
- ✅ 视觉问答（输入图片+问题，输出答案）

**不支持**：
- ❌ 多模态 embedding（将图片和文本转为向量）
- ❌ 图文检索（用文本搜图片，或用图片搜文本）

---

## 支持多模态 Embedding 的 API

### 1. OpenAI - CLIP Embedding（国外）

**API**：
```python
import openai

response = openai.Embedding.create(
    model="clip-vit-base-patch32",
    input=["图片URL或base64"]
)
```

**限制**：
- 需要国外 API
- 需要科学上网

---

### 2. Cohere - Embed（国外）

**支持**：
- ✅ 多模态 embedding
- ✅ 图文检索

**限制**：
- 国外服务

---

### 3. Jina AI - Jina Embeddings（推荐！）

**支持**：
- ✅ 多模态 embedding
- ✅ 图文检索
- ✅ 中文支持
- ✅ 有免费额度

**API 示例**：
```python
import requests

url = "https://api.jina.ai/v1/embeddings"
headers = {
    "Authorization": "Bearer YOUR_API_KEY",
    "Content-Type": "application/json"
}

data = {
    "model": "jina-clip-v1",
    "input": [
        {"text": "一只猫"},
        {"image": "图片URL或base64"}
    ]
}

response = requests.post(url, headers=headers, json=data)
embeddings = response.json()["data"]
```

**优势**：
- 国内可访问
- 中文支持好
- 有免费额度
- 专门为检索优化

---

## 实际解决方案

### 方案1：使用 Jina AI（推荐）

**优势**：
- ✅ API 调用，无需本地模型
- ✅ 支持多模态 embedding
- ✅ 国内可访问
- ✅ 有免费额度

**劣势**：
- ⚠️ 需要注册账号
- ⚠️ 有调用限制

---

### 方案2：本地 CLIP 模型（推荐）

**优势**：
- ✅ 完全免费
- ✅ 无调用限制
- ✅ 数据隐私

**劣势**：
- ⚠️ 需要本地计算资源
- ⚠️ 首次下载模型

---

### 方案3：国内 API + 本地 Embedding

**混合方案**：
1. 使用千问等 API 理解图片内容
2. 将理解结果转为文本
3. 使用本地文本 embedding 模型

**示例**：
```python
# 1. 用千问理解图片
description = qwen_vl.describe_image(image)
# 输出："一只猫在睡觉"

# 2. 用本地模型生成 embedding
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
embedding = model.encode(description)
```

**优势**：
- ✅ 利用国内 API 的图片理解能力
- ✅ 本地 embedding 免费
- ✅ 可以实现图文检索

**劣势**：
- ⚠️ 两步处理，速度较慢
- ⚠️ 精度可能不如原生多模态 embedding

---

## 价格对比

| 服务 | 多模态理解 | 多模态 Embedding | 价格 |
|------|-----------|----------------|------|
| 千问 VL | ✅ | ❌ | ¥0.008/千tokens |
| GLM-4V | ✅ | ❌ | ¥0.05/千tokens |
| Jina AI | ❌ | ✅ | $0.02/千次 |
| 本地 CLIP | ✅ | ✅ | 免费 |

---

## 推荐方案

### 场景1：图片内容理解、视觉问答
**推荐**：千问 VL、GLM-4V
```python
# 用户上传图片，问："这是什么？"
answer = qwen_vl.chat(image, "这是什么？")
```

### 场景2：图文检索、以图搜图
**推荐**：本地 CLIP 或 Jina AI
```python
# 用户上传图片，搜索相似图片
query_embedding = clip.encode_image(query_image)
results = search_similar(query_embedding, image_database)
```

### 场景3：多模态 RAG
**推荐**：混合方案
```python
# 1. 用千问理解图片
description = qwen_vl.describe_image(image)

# 2. 用本地模型检索
embedding = bge_model.encode(description)
relevant_docs = vector_store.search(embedding)

# 3. 用 DeepSeek 生成答案
answer = deepseek.chat(question, relevant_docs)
```

---

## 总结

**现状**：
- 国内 API 大多只支持图片理解，不支持 embedding
- 真正的多模态 embedding 需要用国外 API 或本地模型

**最佳实践**：
1. **图片理解**：用国内 API（千问、GLM-4V）
2. **图文检索**：用本地 CLIP 模型
3. **混合使用**：发挥各自优势

**未来展望**：
- 国内厂商可能会推出多模态 embedding API
- 关注阿里云、智谱 AI 的更新
