# LangChain 读取 PDF 文档

> 本文是 [refine-rag](https://github.com/zonezoen/refine-rag) 系列教程的第四篇，深入讲解如何使用多种方案处理 PDF 文档。

## 目录

- 前言
- 环境准备
- 方案一：PyPDF（轻量快速）
- 方案二：PyMuPDF（功能强大）
- 方案三：Unstructured（智能解析）
- 方案四：父子文档结构解析
- 方案对比与选择
- 常见问题
- 下一步学习

## 前言

PDF 是最常见的文档格式，但也是最难处理的格式之一。不同于纯文本，PDF 包含：

- 复杂的排版布局
- 多列文本
- 表格和图片
- 嵌入字体
- 扫描件（需要 OCR）

本文将介绍 4 种主流的 PDF 处理方案，从简单到复杂，从快速到智能，帮你找到最适合的方案。

## 环境准备

### 1. 安装依赖包

```bash
# 方案一：PyPDF（最轻量）
pip install pypdf langchain-community

# 方案二：PyMuPDF（功能强大）
pip install pymupdf

# 方案三：Unstructured（智能解析）
pip install "unstructured[pdf]" langchain-unstructured

# 中文 OCR 支持（可选）
brew install tesseract-lang  # macOS
# 或
sudo apt-get install tesseract-ocr-chi-sim  # Ubuntu

# 一键安装所有依赖
pip install pypdf pymupdf "unstructured[pdf]" langchain-community langchain-unstructured
```

### 2. 准备测试 PDF

准备一个测试 PDF 文件，可以是：
- 技术文档
- 产品手册
- 学术论文
- 扫描件

## 方案一：PyPDF（轻量快速）

最简单的 PDF 处理方案，适合纯文本提取。

**文件名：** `01-PyPDF读取.py`

```python
from langchain_community.document_loaders import PyPDFLoader

# 加载 PDF 文件
loader = PyPDFLoader("../99-doc-data/黑悟空/黑神话悟空.pdf")
data = loader.load()

# 遍历每一页
for i, page in enumerate(data):
    print(f"=== 第 {i+1} 页 ===")
    print(page.page_content)
    print(f"元数据: {page.metadata}")
    print("-" * 50)
```

**输出结果：**

```python
[
    Document(
        page_content='黑神话：悟空\n\n游戏简介...',
        metadata={'source': '黑神话悟空.pdf', 'page': 0}
    ),
    Document(
        page_content='第一章 黑风山...',
        metadata={'source': '黑神话悟空.pdf', 'page': 1}
    )
]
```

**特点：**

✅ 优势：
- 安装简单，无外部依赖
- 速度快，内存占用低
- 按页分割，结构清晰
- 适合纯文本 PDF

❌ 劣势：
- 不支持复杂布局
- 无法识别文档结构（标题、段落）
- 不支持扫描件 OCR
- 表格提取效果差

**适用场景：**
- 简单的文本 PDF
- 需要快速提取内容
- 对文档结构要求不高

## 方案二：PyMuPDF（功能强大）

更强大的 PDF 处理库，提供丰富的元数据和控制能力。

**文件名：** `02-PyMuPDF.py`

```python
import pymupdf

# 打开 PDF 文件
doc = pymupdf.open("../../99-doc-data/黑悟空/黑神话悟空.pdf")

# 提取所有页面的文本
text = [page.get_text() for page in doc]
print(text)

# 获取文档元数据
print("=== PyMuPDF 基本信息提取 ===")
print(f"文档页数: {len(doc)}")
print(f"文档标题: {doc.metadata.get('title', 'N/A')}")
print(f"文档作者: {doc.metadata.get('author', 'N/A')}")
print(f"创建时间: {doc.metadata.get('creationDate', 'N/A')}")
print(f"完整元数据: {doc.metadata}")

# 遍历每一页，提取详细信息
for page_num, page in enumerate(doc):
    print(f"\n--- 第 {page_num + 1} 页 ---")

    # 提取文本
    text = page.get_text()
    print(f"文本内容: {text[:200]}...")  # 显示前 200 个字符

    # 提取图片
    images = page.get_images()
    print(f"图片数量: {len(images)}")

    # 获取页面链接
    links = page.get_links()
    print(f"链接数量: {len(links)}")

    # 获取页面尺寸
    width, height = page.rect.width, page.rect.height
    print(f"页面尺寸: {width:.2f} x {height:.2f}")

doc.close()
```

**进阶功能：**

```python
import pymupdf

doc = pymupdf.open("document.pdf")
page = doc[0]

# 1. 提取图片并保存
for img_index, img in enumerate(page.get_images()):
    xref = img[0]
    base_image = doc.extract_image(xref)
    image_bytes = base_image["image"]
    
    # 保存图片
    with open(f"image_{img_index}.png", "wb") as f:
        f.write(image_bytes)

# 2. 提取表格（需要额外处理）
tables = page.find_tables()
for table in tables:
    df = table.to_pandas()
    print(df)

# 3. 搜索文本
text_instances = page.search_for("关键词")
for inst in text_instances:
    print(f"找到位置: {inst}")

# 4. 提取带格式的文本
blocks = page.get_text("dict")["blocks"]
for block in blocks:
    if block["type"] == 0:  # 文本块
        for line in block["lines"]:
            for span in line["spans"]:
                print(f"文字: {span['text']}, 字体: {span['font']}, 大小: {span['size']}")

doc.close()
```

**特点：**

✅ 优势：
- 速度快，性能优秀
- 丰富的元数据（作者、创建时间等）
- 可以提取图片、链接
- 支持表格识别
- 可以获取文本格式（字体、大小）
- 内存占用少

❌ 劣势：
- 不自动识别文档结构
- 需要手动处理布局分析
- 扫描件需要额外 OCR

**适用场景：**
- 需要提取图片和元数据
- 需要精细控制 PDF 处理
- 性能要求高
- 需要处理大量 PDF

## 方案三：Unstructured（智能解析）

最智能的 PDF 处理方案，自动识别文档结构。

**文件名：** `03-LangChain-Unstrucured-PDF.py`

```python
from langchain_unstructured import UnstructuredLoader

# 中文 PDF
loader = UnstructuredLoader(
    file_path="../99-doc-data/山西文旅/云冈石窟-ch.pdf",
    strategy="hi_res",      # 高分辨率策略
    languages=["chi_sim"]   # 简体中文 OCR
)

# 英文 PDF
# loader = UnstructuredLoader(
#     file_path="../99-doc-data/山西文旅/云冈石窟-en.pdf",
#     strategy="hi_res"
# )

docs = []

# lazy_load() 延迟加载，节省内存
for doc in loader.lazy_load():
    docs.append(doc)

# 查看解析结果
for i, doc in enumerate(docs[:5]):  # 只显示前 5 个
    print(f"\n=== 元素 {i+1} ===")
    print(f"类型: {doc.metadata.get('category')}")
    print(f"内容: {doc.page_content[:100]}...")
    print(f"页码: {doc.metadata.get('page_number')}")
    print(f"元素ID: {doc.metadata.get('element_id')}")
    print(f"父元素ID: {doc.metadata.get('parent_id')}")
```

**`strategy` 参数说明：**

| 策略 | 说明 | 速度 | 准确度 | 适用场景 |
|------|------|------|--------|----------|
| `fast` | 快速模式 | ⚡⚡⚡ | ⭐⭐ | 纯文本 PDF |
| `hi_res` | 高分辨率 | ⚡ | ⭐⭐⭐⭐ | 复杂布局、扫描件 |
| `ocr_only` | 仅 OCR | ⚡⚡ | ⭐⭐⭐ | 纯图片 PDF |

**使用 partition 函数（更底层）：**

**文件名：** `04-Unstrctured-使用partition函数解析PDF-v1.py`

```python
from unstructured.partition.auto import partition

filename = "../99-doc-data/黑悟空/黑神话悟空.pdf"

# 使用 partition 函数解析 PDF
elements = partition(
    filename=filename,
    content_type="application/pdf"
)

# 展示解析出的元素类型和内容
print("PDF 解析后的 Elements 类型:")
for i, element in enumerate(elements[:5]):
    print(f"\nElement {i+1}:")
    print(f"类型: {type(element).__name__}")
    print(f"内容: {str(element)[:100]}...")
    print("-" * 50)

# 统计不同类型元素的数量
element_types = {}
for element in elements:
    element_type = type(element).__name__
    element_types[element_type] = element_types.get(element_type, 0) + 1

print("\nElements 类型统计:")
for element_type, count in element_types.items():
    print(f"{element_type}: {count} 个")
```

**输出示例：**

```
Elements 类型统计:
Title: 12 个
NarrativeText: 45 个
ListItem: 8 个
Table: 3 个
Image: 2 个
```

**特点：**

✅ 优势：
- 自动识别文档结构（标题、段落、列表）
- 支持扫描件 OCR
- 可以识别表格
- 保留父子关系
- 适合复杂 PDF

❌ 劣势：
- 速度较慢
- 依赖外部工具（Tesseract）
- 内存占用较大

**适用场景：**
- 复杂布局的 PDF
- 需要保留文档结构
- 扫描件处理
- 构建知识图谱

## 方案四：父子文档结构解析

保留文档的层级关系，实现更精准的检索。

### 4.1 使用 LangChain 封装

**文件名：** `05-父子文档解析-Unstructured-LangChain.py`

```python
from langchain_unstructured import UnstructuredLoader

file_path = '../99-doc-data/山西文旅/云冈石窟-en.pdf'

# 加载 PDF
loader = UnstructuredLoader(
    file_path=file_path,
    strategy="hi_res"
)

docs = []
for doc in loader.lazy_load():
    docs.append(doc)

# 仅筛选第一页的文档
page_number = 1
page_docs = [doc for doc in docs if doc.metadata.get("page_number") == page_number]

# 打印每个元素的详细信息
for i, doc in enumerate(page_docs, 1):
    print(f"Doc {i}:")
    print(f"  内容: {doc.page_content}")
    print(f"  分类: {doc.metadata.get('category')}")
    print(f"  ID: {doc.metadata.get('element_id')}")
    print(f"  Parent ID: {doc.metadata.get('parent_id')}")
    print("=" * 50)

# 构建父子关系
title_dict = {}

# 收集 Title，建立 parent_id -> Title 的映射
for doc in docs:
    if (doc.metadata.get("category") == "Title" and 
        doc.metadata.get("page_number") == page_number):
        title_id = doc.metadata.get("element_id")
        title_text = doc.page_content.strip()
        if title_text not in [data["title"] for data in title_dict.values()]:
            title_dict[title_id] = {"title": title_text, "content": []}

# 关联 Title 和其对应的 Text
for doc in docs:
    if (doc.metadata.get("category") in ["NarrativeText", "Text"] and 
        doc.metadata.get("page_number") == page_number):
        parent_id = doc.metadata.get("parent_id")
        if parent_id in title_dict:
            content = doc.page_content.strip()
            if content:
                title_dict[parent_id]["content"].append(content)

# 输出结构化结果
for title_data in title_dict.values():
    if title_data["content"]:
        print("\n=== " + title_data["title"] + " ===")
        for content in title_data["content"]:
            print(content)
        print()
```

### 4.2 使用原生 Unstructured API

**文件名：** `06-父子文档-Unstructured-ParitionPDF.py`

```python
from unstructured.documents.elements import Title, NarrativeText, Text
from unstructured.partition.pdf import partition_pdf

file_path = '../99-doc-data/山西文旅/云冈石窟-en.pdf'

# 使用 unstructured 直接读取 PDF
elements = partition_pdf(
    filename=file_path,
    strategy="hi_res"
)

# 查看第一个元素的完整信息
if elements:
    first_elem = elements[0]
    print("=== 第一个元素的详细信息 ===")
    print(f"类型: {type(first_elem)}")
    print(f"文本: {first_elem.text}")
    print(f"Metadata: {vars(first_elem.metadata)}")
    print("=" * 50)

# 仅筛选第一页的元素
page_number = 1
page_elements = [
    elem for elem in elements 
    if getattr(elem.metadata, "page_number", None) == page_number
]

# 打印每个元素的详细信息
for i, elem in enumerate(page_elements, 1):
    print(f"\nElement {i}:")
    print(f"  内容: {elem.text}")
    print(f"  分类: {type(elem).__name__}")
    print(f"  ID: {getattr(elem, '_element_id', None)}")
    print("=" * 50)

# 构建父子关系
title_dict = {}

# 收集 Title（使用类型检查）
for elem in elements:
    if (isinstance(elem, Title) and 
        getattr(elem.metadata, "page_number", None) == page_number):
        title_id = getattr(elem, '_element_id', None)
        title_text = elem.text.strip()
        if title_text not in [data["title"] for data in title_dict.values()]:
            title_dict[title_id] = {"title": title_text, "content": []}

# 关联 Title 和其对应的 Text
for elem in elements:
    if (isinstance(elem, (NarrativeText, Text)) and 
        getattr(elem.metadata, "page_number", None) == page_number):
        parent_id = getattr(elem.metadata, "parent_id", None)
        if parent_id in title_dict:
            content = elem.text.strip()
            if content:
                title_dict[parent_id]["content"].append(content)

# 输出结构化结果
for title_data in title_dict.values():
    if title_data["content"]:
        print("\n=== " + title_data["title"] + " ===")
        for content in title_data["content"]:
            print(content)
        print()
```

**两种方式的区别：**

| 特性 | LangChain 封装 | 原生 Unstructured |
|------|---------------|------------------|
| 返回类型 | `Document` 对象 | `Element` 对象 |
| 内容访问 | `doc.page_content` | `elem.text` |
| 类型判断 | `doc.metadata["category"] == "Title"` | `isinstance(elem, Title)` |
| 元数据访问 | `doc.metadata["parent_id"]` | `elem.metadata.parent_id` |
| 类型检查 | 字符串比较 | Python 类型检查（更安全） |
| 用途 | LangChain/RAG 集成 | 底层文档处理 |

**parent_id 的生成机制：**

Unstructured 库在解析 PDF 时会：
1. 分析文档的层级结构
2. 识别标题、段落、列表等元素
3. 根据排版、字体大小、位置推断父子关系
4. 为每个元素生成唯一的 `element_id`
5. 为子元素设置 `parent_id` 指向父元素

**示例结构：**

```python
[
    {
        "element_id": "abc123",
        "category": "Title",
        "content": "云冈石窟简介",
        "parent_id": None  # 顶级标题
    },
    {
        "element_id": "def456",
        "category": "NarrativeText",
        "content": "云冈石窟位于山西省...",
        "parent_id": "abc123"  # 属于上面的标题
    }
]
```

## 方案对比与选择

### 性能对比

| 方案 | 速度 | 内存 | 准确度 | 结构识别 | OCR 支持 |
|------|------|------|--------|----------|----------|
| PyPDF | ⚡⚡⚡ | 💾 | ⭐⭐ | ❌ | ❌ |
| PyMuPDF | ⚡⚡⚡ | 💾 | ⭐⭐⭐ | 部分 | ❌ |
| Unstructured | ⚡ | 💾💾💾 | ⭐⭐⭐⭐ | ✅ | ✅ |

### 选择建议

**使用 PyPDF：**
- ✅ 简单的纯文本 PDF
- ✅ 需要快速提取
- ✅ 按页分割即可

**使用 PyMuPDF：**
- ✅ 需要提取图片、元数据
- ✅ 需要精细控制
- ✅ 处理大量 PDF
- ✅ 性能要求高

**使用 Unstructured：**
- ✅ 复杂布局的 PDF
- ✅ 需要识别文档结构
- ✅ 扫描件处理
- ✅ 构建知识图谱
- ✅ 需要父子关系

### 混合方案

实际项目中，可以结合多种方案：

```python
def smart_pdf_loader(pdf_path):
    """智能选择 PDF 加载方案"""
    import pymupdf
    
    # 1. 先用 PyMuPDF 快速检查
    doc = pymupdf.open(pdf_path)
    page_count = len(doc)
    has_images = any(page.get_images() for page in doc)
    doc.close()
    
    # 2. 根据特征选择方案
    if page_count < 10 and not has_images:
        # 简单文档，用 PyPDF
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(pdf_path)
        return loader.load()
    elif has_images:
        # 有图片，用 Unstructured
        from langchain_unstructured import UnstructuredLoader
        loader = UnstructuredLoader(pdf_path, strategy="hi_res")
        return list(loader.lazy_load())
    else:
        # 默认用 PyMuPDF
        text = []
        doc = pymupdf.open(pdf_path)
        for page in doc:
            text.append(page.get_text())
        doc.close()
        return text

# 使用
docs = smart_pdf_loader("document.pdf")
```

## 常见问题

### 1. 中文 OCR 识别不准

```bash
# 安装中文语言包
# macOS
brew install tesseract-lang

# Ubuntu
sudo apt-get install tesseract-ocr-chi-sim

# 验证安装
tesseract --list-langs
# 应该能看到 chi_sim
```

```python
# 使用时指定语言
loader = UnstructuredLoader(
    file_path="chinese.pdf",
    strategy="hi_res",
    languages=["chi_sim", "eng"]  # 中英文混合
)
```

### 2. Unstructured 解析速度慢

```python
# 优化策略
loader = UnstructuredLoader(
    file_path="large.pdf",
    strategy="fast",  # 使用快速模式
    # 或者只处理部分页面
)

# 分批处理
def process_pdf_in_batches(pdf_path, batch_size=10):
    import pymupdf
    doc = pymupdf.open(pdf_path)
    total_pages = len(doc)
    
    for start in range(0, total_pages, batch_size):
        end = min(start + batch_size, total_pages)
        # 提取部分页面到临时文件
        # 然后用 Unstructured 处理
        pass
```

### 3. 内存占用过高

```python
# 使用 lazy_load 延迟加载
loader = UnstructuredLoader(file_path="large.pdf")

for doc in loader.lazy_load():
    # 逐个处理，不一次性加载到内存
    process_document(doc)
```

### 4. 表格提取效果差

```python
# 使用 PyMuPDF 的表格识别
import pymupdf

doc = pymupdf.open("document.pdf")
for page in doc:
    tables = page.find_tables()
    for table in tables:
        df = table.to_pandas()
        print(df)

# 或者使用专门的表格提取库
# pip install camelot-py pdfplumber
```

### 5. 扫描件识别不出来

```python
# 确保使用 hi_res 策略
loader = UnstructuredLoader(
    file_path="scanned.pdf",
    strategy="hi_res",  # 必须使用高分辨率
    languages=["chi_sim"]
)

# 或者使用 ocr_only
loader = UnstructuredLoader(
    file_path="scanned.pdf",
    strategy="ocr_only"  # 纯 OCR 模式
)
```

## 进阶技巧

### 1. 批量处理 PDF

```python
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

# 批量加载目录下的所有 PDF
loader = DirectoryLoader(
    "./documents/",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader,
    show_progress=True
)

docs = loader.load()
print(f"共加载 {len(docs)} 个文档")
```

### 2. PDF 预处理

```python
import pymupdf

def preprocess_pdf(input_path, output_path):
    """PDF 预处理：去除水印、调整对比度等"""
    doc = pymupdf.open(input_path)
    
    for page in doc:
        # 移除注释和水印
        page.clean_contents()
        
        # 可以添加更多预处理逻辑
    
    doc.save(output_path)
    doc.close()

# 使用
preprocess_pdf("original.pdf", "cleaned.pdf")
```

### 3. 构建 PDF 索引

```python
from langchain_unstructured import UnstructuredLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# 1. 加载 PDF
loader = UnstructuredLoader("document.pdf", strategy="hi_res")
docs = list(loader.lazy_load())

# 2. 创建向量索引
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vectorstore = FAISS.from_documents(docs, embeddings)

# 3. 检索
query = "云冈石窟的历史"
results = vectorstore.similarity_search(query, k=3)

for doc in results:
    print(f"内容: {doc.page_content}")
    print(f"来源: {doc.metadata}")
    print("-" * 50)
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