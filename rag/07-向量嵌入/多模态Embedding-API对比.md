# 多模态 Embedding API 完整对比

## 支持真正多模态 Embedding 的 API

### 1. Jina AI ⭐⭐⭐⭐⭐（强烈推荐）

**官网**: https://jina.ai/

**特点**:
- ✅ 真正的多模态 embedding
- ✅ 国内可访问，无需科学上网
- ✅ 有免费额度（100万 tokens/月）
- ✅ 中文支持好
- ✅ 专门为检索优化

**模型**:
- `jina-clip-v1`: 多模态模型（图片+文本）
- `jina-embeddings-v2`: 纯文本模型

**定价**:
- 免费额度: 100万 tokens/月
- 付费: $0.02/千次调用
- 图片按分辨率计费

**API 示例**:
```python
import requests

url = "https://api.jina.ai/v1/embeddings"
headers = {
    "Authorization": "Bearer YOUR_API_KEY",
    "Content-Type": "application/json"
}

# 编码图片
data = {
    "model": "jina-clip-v1",
    "input": [
        {"image": "data:image/jpeg;base64,/9j/4AAQ..."}
    ]
}

response = requests.post(url, headers=headers, json=data)
embedding = response.json()["data"][0]["embedding"]
```

**优势**:
- 国内访问稳定
- 免费额度充足
- 文档完善
- 响应速度快

**劣势**:
- 需要注册账号
- 超出免费额度需付费

---

### 2. Cohere Embed ⭐⭐⭐⭐

**官网**: https://cohere.com/

**特点**:
- ✅ 真正的多模态 embedding
- ✅ 支持多种语言
- ⚠️ 国内访问可能不稳定

**模型**:
- `embed-english-v3.0`: 英文模型
- `embed-multilingual-v3.0`: 多语言模型

**定价**:
- 免费额度: 有限
- 付费: $0.10/百万 tokens

**API 示例**:
```python
import cohere

co = cohere.Client('YOUR_API_KEY')

response = co.embed(
    texts=["文本内容"],
    model='embed-multilingual-v3.0'
)

embeddings = response.embeddings
```

**优势**:
- 模型质量高
- 支持多语言
- 文档完善

**劣势**:
- 国内访问不稳定
- 免费额度较少
- 价格较高

---

### 3. Voyage AI ⭐⭐⭐⭐

**官网**: https://www.voyageai.com/

**特点**:
- ✅ 专门为 RAG 优化
- ✅ 支持多模态
- ⚠️ 国内访问可能不稳定

**模型**:
- `voyage-multimodal-3`: 多模态模型
- `voyage-3`: 纯文本模型

**定价**:
- 免费额度: 有限
- 付费: $0.12/百万 tokens

**API 示例**:
```python
import voyageai

vo = voyageai.Client(api_key="YOUR_API_KEY")

result = vo.multimodal_embed(
    inputs=[
        {"image": "image_url"},
        {"text": "文本内容"}
    ],
    model="voyage-multimodal-3"
)

embeddings = result.embeddings
```

**优势**:
- 专门为 RAG 优化
- 检索效果好

**劣势**:
- 国内访问不稳定
- 价格较高
- 知名度较低

---

### 4. OpenAI (未来可能支持)

**官网**: https://openai.com/

**现状**:
- ❌ 目前不支持多模态 embedding
- ✅ 只支持文本 embedding (`text-embedding-3-small/large`)
- ⚠️ 未来可能会推出

**如果推出，预计特点**:
- 模型质量高
- 需要科学上网
- 价格可能较高

---

## 不支持多模态 Embedding 的 API

### 国内 API（只支持图片理解，不支持 embedding）

| API | 图片理解 | 多模态 Embedding | 备注 |
|-----|---------|----------------|------|
| 千问 VL | ✅ | ❌ | 只能生成文字描述 |
| GLM-4V | ✅ | ❌ | 只能生成文字描述 |
| 文心一言 | ✅ | ❌ | 只能生成文字描述 |
| DeepSeek | ❌ | ❌ | 目前不支持多模态 |
| Minimax | ✅ | ❌ | 只能生成文字描述 |
| 豆包 | ✅ | ❌ | 只能生成文字描述 |

---

## 完整对比表

| API | 多模态 Embedding | 国内访问 | 免费额度 | 价格 | 推荐度 |
|-----|----------------|---------|---------|------|--------|
| **Jina AI** | ✅ | ✅ | 100万/月 | $0.02/千次 | ⭐⭐⭐⭐⭐ |
| **Cohere** | ✅ | ⚠️ | 有限 | $0.10/百万 | ⭐⭐⭐⭐ |
| **Voyage AI** | ✅ | ⚠️ | 有限 | $0.12/百万 | ⭐⭐⭐⭐ |
| **本地 CLIP** | ✅ | ✅ | 无限 | 免费 | ⭐⭐⭐⭐⭐ |
| **千问+BGE** | ⚠️ | ✅ | 有限 | ¥0.008/千 | ⭐⭐⭐⭐ |

---

## 推荐方案

### 场景1：不想下载模型 + 需要真多模态
**推荐**: Jina AI
```python
from jina_multimodal import JinaMultimodalEmbedding

client = JinaMultimodalEmbedding(api_key="your_key")
image_vec = client.encode_image("image.jpg")
text_vec = client.encode_text("文本")
```

**理由**:
- 国内可访问
- 免费额度充足
- 真正的多模态

---

### 场景2：完全免费 + 无限制
**推荐**: 本地 CLIP
```python
from transformers import CLIPModel, CLIPProcessor

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 编码图片和文本
inputs = processor(images=image, text=text, return_tensors="pt")
outputs = model(**inputs)
```

**理由**:
- 完全免费
- 无调用限制
- 数据隐私

---

### 场景3：需要可解释性
**推荐**: 千问 VL + 本地 BGE
```python
# 1. 用千问理解图片
description = qwen_vl.describe(image)

# 2. 用本地模型生成向量
embedding = bge_model.encode(description)
```

**理由**:
- 可以看到图片描述
- 描述可以展示给用户
- 灵活性高

---

## 成本对比（处理1000张图片）

| 方案 | 成本 | 说明 |
|------|------|------|
| Jina AI | $0.02 | 1000次调用 |
| Cohere | $0.10 | 按 tokens 计费 |
| 本地 CLIP | 免费 | 电费忽略不计 |
| 千问+BGE | ¥8 | 千问 API 调用 |

---

## 快速开始

### 1. Jina AI（推荐）

```bash
# 1. 注册账号
# 访问 https://jina.ai/

# 2. 获取 API Key
# 在 Dashboard 中创建

# 3. 安装依赖
pip install requests pillow

# 4. 运行代码
python rag/07-向量嵌入/08-真正的多模态嵌入-JinaAI.py
```

### 2. 本地 CLIP

```bash
# 1. 安装依赖
pip install transformers pillow torch

# 2. 运行代码
python rag/07-向量嵌入/05-多模态嵌入-CLIP版本.py
```

---

## 总结

**如果不想下载模型，最佳选择是 Jina AI**:
- ✅ 真正的多模态 embedding
- ✅ 国内可访问
- ✅ 免费额度充足
- ✅ 简单易用

**如果想完全免费，选择本地 CLIP**:
- ✅ 无限制使用
- ✅ 数据隐私
- ⚠️ 需要下载模型（约 600MB）

**如果需要可解释性，选择千问+BGE**:
- ✅ 可以看到图片描述
- ✅ 灵活性高
- ⚠️ 不是真正的多模态 embedding
