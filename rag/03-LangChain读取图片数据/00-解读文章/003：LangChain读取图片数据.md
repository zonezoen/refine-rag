# LangChain 读取图片数据

> 本文是 [refine-rag](https://github.com/zonezoen/refine-rag) 系列教程的第三篇，教你如何使用 LangChain 处理图片和 PPT 等多媒体数据。

## 目录

- 前言
- 环境准备
- 读取图片中的文字（OCR）
- 读取 PPT 文档
- 多模态 RAG 的应用场景
- 常见问题
- 下一步学习

## 前言

在前面的文章中，我们学习了如何读取纯文本数据。但在实际应用中，很多重要信息都存储在图片、PPT、扫描件等非文本格式中。比如：

- 产品宣传图中的文字说明
- PPT 演示文稿中的内容
- 扫描的合同和发票
- 截图中的代码和配置

这些数据如果不能被 RAG 系统处理，就会造成信息的缺失。本文将介绍如何使用 Unstructured 库来处理这些多媒体数据。

## 环境准备

### 1. 安装依赖包

```bash
# 基础依赖
pip install langchain langchain-community

# Unstructured 完整版（包含图片处理）
pip install "unstructured[all-docs]"

# OCR 依赖（用于图片文字识别）
pip install pdfminer.six

# 如果需要更好的 OCR 效果，可以安装 Tesseract
# macOS: brew install tesseract
# Ubuntu: sudo apt-get install tesseract-ocr
# Windows: 下载安装包 https://github.com/UB-Mannheim/tesseract/wiki
```

### 2. 安装 LibreOffice（处理 PPT 必需）

Unstructured 处理 Office 文档时需要 LibreOffice 的命令行工具：

```bash
# macOS
brew install libreoffice

# Ubuntu/Debian
sudo apt-get update && sudo apt-get install -y libreoffice

# Windows
# 下载安装：https://www.libreoffice.org/download/download/
```

**为什么需要 LibreOffice？**

Unstructured 会调用 `soffice` 命令将 PPT/Word/Excel 转换为可解析的格式。如果没有安装，会报错：`FileNotFoundError: [Errno 2] No such file or directory: 'soffice'`

### 3. 准备测试数据

准备一些测试文件：
- 一张包含文字的图片（如截图、海报）
- 一个 PPT 文件

## 读取图片中的文字（OCR）

使用 Unstructured 的 OCR 功能提取图片中的文字。

**文件名：** `01-Unstructured读图.py`

```python
from langchain_community.document_loaders import UnstructuredImageLoader

# 图片路径
image_path = "../../99-doc-data/黑悟空/黑悟空英文.jpg"

# 创建加载器
loader = UnstructuredImageLoader(image_path)

# 加载并提取文字
data = loader.load()
print(data)
```

**输出结果：**

```python
Warning: No languages specified, defaulting to English.
[Document(
    metadata={'source': '../99-doc-data/黑悟空/黑悟空英文.jpg'}, 
    page_content='2\n\nPons\n\nBLACK MYTH. WUKONGY\n\n4'
)]
```

**关键点：**

1. **语言设置**：默认使用英文 OCR，如果是中文图片，识别效果可能不佳
2. **OCR 引擎**：Unstructured 默认使用 Tesseract，可以通过参数指定其他引擎
3. **图片质量**：清晰度越高，识别准确率越高

**改进版本（支持中文）：**

```python
from langchain_community.document_loaders import UnstructuredImageLoader

# 指定中文语言
loader = UnstructuredImageLoader(
    image_path,
    strategy="hi_res",  # 高分辨率模式，识别更准确
    languages=["chi_sim", "eng"]  # 支持简体中文和英文
)

data = loader.load()
print(data[0].page_content)
```

**`strategy` 参数说明：**

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| `fast` | 快速模式，不使用 OCR | 图片中没有文字 |
| `hi_res` | 高分辨率模式，使用 OCR | 需要提取图片中的文字 |
| `ocr_only` | 仅使用 OCR | 扫描件、纯文字图片 |

## 读取 PPT 文档

PPT 是常见的演示文稿格式，包含文字、图片、图表等多种元素。

**文件名：** `02-Unstructured读PPT.py`

```python
from unstructured.partition.ppt import partition_ppt
from langchain_core.documents import Document

# 解析 PPT 文件
ppt_elements = partition_ppt(filename="../99-doc-data/黑悟空/黑神话悟空.pptx")

print("PPT 内容：")
for element in ppt_elements:
    print("=====分页=====")
    print(element.text)

# 转换为 LangChain 的 Document 格式
documents = [
    Document(
        page_content=element.text, 
        metadata={
            "source": "data/黑神话悟空PPT.pptx",
            "page": i,
            "type": str(type(element).__name__)
        }
    )
    for i, element in enumerate(ppt_elements)
]

# 输出转换后的 Documents
print(f"\n共提取 {len(documents)} 个元素")
for doc in documents[:3]:  # 只打印前 3 个
    print(f"类型: {doc.metadata['type']}")
    print(f"内容: {doc.page_content[:100]}...")
    print("-" * 50)
```

**PPT 元素类型：**

Unstructured 会将 PPT 解析为不同类型的元素：

- `Title`：标题
- `NarrativeText`：正文段落
- `ListItem`：列表项
- `Table`：表格
- `Image`：图片（会尝试 OCR）

**保留更多元数据：**

```python
documents = [
    Document(
        page_content=element.text, 
        metadata={
            "source": "data/黑神话悟空PPT.pptx",
            "page": i,
            "type": str(type(element).__name__),
            "category": element.category if hasattr(element, 'category') else None,
            "element_id": element.id if hasattr(element, 'id') else None
        }
    )
    for i, element in enumerate(ppt_elements)
]
```

**处理 PPT 中的图片：**

```python
from unstructured.partition.ppt import partition_ppt

# 启用图片推断（会对图片进行 OCR）
ppt_elements = partition_ppt(
    filename="../99-doc-data/黑悟空/黑神话悟空.pptx",
    infer_table_structure=True,  # 推断表格结构
    strategy="hi_res",  # 高分辨率模式
    extract_images_in_pdf=True  # 提取图片
)
```

## 多模态 RAG 的应用场景

处理图片和 PPT 数据后，可以构建更强大的多模态 RAG 系统：

### 1. 企业知识库

```python
# 场景：公司有大量 PPT 培训资料
from langchain_community.document_loaders import DirectoryLoader
from unstructured.partition.ppt import partition_ppt

# 批量加载所有 PPT
loader = DirectoryLoader(
    "./training_materials/",
    glob="**/*.pptx",
    show_progress=True
)

# 构建知识库
documents = []
for doc in loader.load():
    ppt_elements = partition_ppt(filename=doc.metadata['source'])
    for element in ppt_elements:
        documents.append(Document(
            page_content=element.text,
            metadata={"source": doc.metadata['source']}
        ))

# 后续可以进行向量化和检索
```

### 2. 产品文档助手

```python
# 场景：产品手册包含大量截图和说明
from langchain_community.document_loaders import UnstructuredImageLoader

# 处理产品截图
screenshots = ["feature1.png", "feature2.png", "feature3.png"]
docs = []

for img in screenshots:
    loader = UnstructuredImageLoader(
        img,
        strategy="hi_res",
        languages=["chi_sim", "eng"]
    )
    docs.extend(loader.load())

# 结合文本文档构建完整知识库
```

### 3. 合同和发票处理

```python
# 场景：扫描的合同需要提取关键信息
from langchain_community.document_loaders import UnstructuredImageLoader

# 处理扫描件
loader = UnstructuredImageLoader(
    "contract_scan.jpg",
    strategy="ocr_only",  # 纯 OCR 模式
    languages=["chi_sim"]
)

contract_docs = loader.load()

# 提取关键信息（可以结合 LLM）
from langchain_openai import ChatOpenAI

llm = ChatOpenAI()
result = llm.invoke(f"从以下合同中提取甲方、乙方、金额：\n{contract_docs[0].page_content}")
```

### 4. 学术论文分析

```python
# 场景：论文中的图表和公式
# 可以结合多模态模型（如 GPT-4V）进行更深入的理解

from langchain_community.document_loaders import UnstructuredImageLoader

# 提取论文中的图表
loader = UnstructuredImageLoader(
    "paper_figure.png",
    strategy="hi_res"
)

figure_docs = loader.load()

# 使用多模态模型理解图表
# （需要支持图片输入的模型）
```

## 常见问题

### 1. LibreOffice 未安装

```bash
# 错误：FileNotFoundError: [Errno 2] No such file or directory: 'soffice'

# 解决：安装 LibreOffice
# macOS
brew install libreoffice

# Ubuntu
sudo apt-get install libreoffice
```

### 2. OCR 识别效果差

**问题原因：**
- 图片分辨率太低
- 文字太小或模糊
- 语言设置不正确

**解决方案：**

```python
# 1. 使用高分辨率模式
loader = UnstructuredImageLoader(
    image_path,
    strategy="hi_res"  # 而不是 "fast"
)

# 2. 指定正确的语言
loader = UnstructuredImageLoader(
    image_path,
    languages=["chi_sim", "eng"]  # 中英文混合
)

# 3. 预处理图片（提高对比度、去噪等）
from PIL import Image, ImageEnhance

img = Image.open(image_path)
enhancer = ImageEnhance.Contrast(img)
img = enhancer.enhance(2.0)  # 增强对比度
img.save("enhanced.jpg")

# 再进行 OCR
loader = UnstructuredImageLoader("enhanced.jpg")
```

### 3. PPT 解析速度慢

```python
# 问题：大型 PPT 文件解析很慢

# 解决：跳过不必要的处理
ppt_elements = partition_ppt(
    filename="large.pptx",
    include_page_breaks=False,  # 不包含分页符
    infer_table_structure=False,  # 不推断表格结构（如果不需要）
    extract_images_in_pdf=False  # 不提取图片（如果不需要）
)
```

### 4. 中文 OCR 支持

```bash
# 如果使用 Tesseract，需要下载中文语言包

# macOS
brew install tesseract-lang

# Ubuntu
sudo apt-get install tesseract-ocr-chi-sim

# 验证安装
tesseract --list-langs
# 应该能看到 chi_sim（简体中文）
```

### 5. 内存占用过高

```python
# 问题：处理大量图片时内存不足

# 解决：分批处理
def process_images_in_batches(image_paths, batch_size=10):
    results = []
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i+batch_size]
        for img_path in batch:
            loader = UnstructuredImageLoader(img_path)
            results.extend(loader.load())
        # 可以在这里保存中间结果
    return results

# 使用
all_docs = process_images_in_batches(image_list, batch_size=10)
```

## 进阶技巧

### 1. 结合多模态大模型

对于复杂的图片（如图表、流程图），纯 OCR 可能不够：

```python
# 使用支持图片输入的模型（如 GPT-4V、Claude 3）
from langchain_openai import ChatOpenAI
import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# 创建多模态消息
base64_image = encode_image("chart.png")
llm = ChatOpenAI(model="gpt-4-vision-preview")

response = llm.invoke([
    {"type": "text", "text": "请描述这张图表的内容"},
    {"type": "image_url", "image_url": f"data:image/png;base64,{base64_image}"}
])

print(response.content)
```

### 2. 图片预处理流程

```python
from PIL import Image, ImageEnhance, ImageFilter

def preprocess_image(image_path, output_path):
    """图片预处理，提高 OCR 准确率"""
    img = Image.open(image_path)
    
    # 1. 转为灰度图
    img = img.convert('L')
    
    # 2. 增强对比度
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)
    
    # 3. 锐化
    img = img.filter(ImageFilter.SHARPEN)
    
    # 4. 调整大小（如果太小）
    if img.width < 1000:
        scale = 1000 / img.width
        new_size = (int(img.width * scale), int(img.height * scale))
        img = img.resize(new_size, Image.LANCZOS)
    
    img.save(output_path)
    return output_path

# 使用
enhanced_path = preprocess_image("low_quality.jpg", "enhanced.jpg")
loader = UnstructuredImageLoader(enhanced_path)
docs = loader.load()
```

### 3. 构建混合文档索引

```python
from langchain_community.document_loaders import (
    TextLoader, 
    UnstructuredImageLoader,
    DirectoryLoader
)
from unstructured.partition.ppt import partition_ppt

def load_mixed_documents(directory):
    """加载目录下的所有文档（文本、图片、PPT）"""
    all_docs = []
    
    # 1. 加载文本文件
    text_loader = DirectoryLoader(
        directory,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'}
    )
    all_docs.extend(text_loader.load())
    
    # 2. 加载图片
    image_loader = DirectoryLoader(
        directory,
        glob="**/*.{jpg,png,jpeg}",
        loader_cls=UnstructuredImageLoader
    )
    all_docs.extend(image_loader.load())
    
    # 3. 加载 PPT
    import glob
    for ppt_file in glob.glob(f"{directory}/**/*.pptx", recursive=True):
        ppt_elements = partition_ppt(filename=ppt_file)
        for element in ppt_elements:
            all_docs.append(Document(
                page_content=element.text,
                metadata={"source": ppt_file, "type": "ppt"}
            ))
    
    return all_docs

# 使用
docs = load_mixed_documents("./knowledge_base/")
print(f"共加载 {len(docs)} 个文档")
```

## 学习路径

1. 简易RAG 学习
2. LCEL 语法学习
3. LangChain 读取数据
   1. LangChain 读取文本数据
   2. LangChain 读取图片数据
   3. LangChain 读取 PDF 数据
   4. LangChain 读取表格数据
4. 文本切块
5. 向量嵌入
6. 向量存储
7. 检索前处理
8. 索引优化
9. 检索后处理
10. 响应生成
11. 系统评估

## 项目地址

本文所有代码示例都在 GitHub 开源：

https://github.com/zonezoen/refine-rag

欢迎 Star 和 Fork，一起学习 RAG 技术！