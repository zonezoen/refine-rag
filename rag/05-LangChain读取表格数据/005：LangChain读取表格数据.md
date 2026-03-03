# LangChain 读取表格数据

> 本文是 [refine-rag](https://github.com/zonezoen/refine-rag) 系列教程的第五篇，全面讲解如何处理 CSV、PDF 表格等结构化数据。

## 目录

- 前言
- 环境准备
- 方案一：读取 CSV 文件
- 方案二：Camelot 提取 PDF 表格
- 方案三：pdfplumber 提取 PDF 表格
- 方案四：Unstructured 智能表格提取
- 表格问答系统实战
- 方案对比与选择
- 常见问题
- 下一步学习

## 前言

表格是最常见的结构化数据格式，在企业文档、财务报表、数据分析中无处不在。但表格数据的处理比纯文本更复杂：

- CSV 文件需要正确解析列和行
- PDF 表格可能有复杂的布局
- 表格需要保留结构信息
- 表格数据需要与上下文关联

本文将介绍 4 种主流的表格处理方案，并实现一个完整的表格问答系统。

## 环境准备

### 1. 安装依赖包

```bash
# 方案一：CSV 读取
pip install langchain langchain-community

# 方案二：Camelot（PDF 表格提取）
pip install "camelot-py[base]"
brew install ghostscript  # macOS
# sudo apt-get install ghostscript  # Ubuntu

# 方案三：pdfplumber（推荐）
pip install pdfplumber pandas

# 方案四：Unstructured
pip install "unstructured[pdf]"
brew install poppler tesseract tesseract-lang  # macOS

# 表格问答系统
pip install langchain-deepseek langchain-huggingface sentence-transformers python-dotenv

# 一键安装
pip install langchain langchain-community "camelot-py[base]" pdfplumber pandas \
            "unstructured[pdf]" langchain-deepseek langchain-huggingface \
            sentence-transformers python-dotenv
```

### 2. 配置环境变量

创建 `.env` 文件：

```bash
DEEPSEEK_API_KEY=your_deepseek_api_key_here
```

## 方案一：读取 CSV 文件

最简单的表格数据格式，LangChain 提供了多种加载方式。

**文件名：** `01-读取csv.py`

```python
# 方式1：UnstructuredCSVLoader（智能解析）
from langchain_community.document_loaders import UnstructuredCSVLoader

unstructLoader = UnstructuredCSVLoader("../99-doc-data/黑悟空/黑神话悟空.csv")
unstructDocuments = unstructLoader.load()
print("UnstructuredCSVLoader 结果:")
print(unstructDocuments)

# 方式2：CSVLoader（标准加载）
from langchain_community.document_loaders import CSVLoader

loader = CSVLoader("../99-doc-data/黑悟空/黑神话悟空.csv")
documents = loader.load()
print("\nCSVLoader 结果:")
print(documents)

# 方式3：批量加载目录下的所有 CSV
from langchain_community.document_loaders import DirectoryLoader

loader = DirectoryLoader(
    path="../99-doc-data",
    glob="**/*.csv",
    loader_cls=CSVLoader
)

docs = loader.load()
print(f"\n批量加载：共 {len(docs)} 个文档")
print(docs[0])
```

**两种加载器的区别：**

| 特性 | UnstructuredCSVLoader | CSVLoader |
|------|----------------------|-----------|
| 解析方式 | 智能解析，自动识别结构 | 标准 CSV 解析 |
| 输出格式 | 整个表格作为一个文档 | 每行作为一个文档 |
| 元数据 | 较少 | 包含行号等信息 |
| 适用场景 | 小表格、需要整体理解 | 大表格、需要逐行处理 |

**CSVLoader 高级用法：**

```python
from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(
    file_path="data.csv",
    csv_args={
        'delimiter': ',',      # 分隔符
        'quotechar': '"',      # 引号字符
        'fieldnames': ['col1', 'col2']  # 自定义列名
    },
    encoding='utf-8',          # 编码
    source_column='source'     # 指定来源列
)
```


## 方案二：Camelot 提取 PDF 表格

专业的 PDF 表格提取工具，基于 PDF 的矢量信息。

**文件名：** `02-camelot提取PDF表格.py`

```python
import camelot
import pandas as pd
import time

pdf_path = "../99-doc-data/复杂PDF/billionaires_page-1-5.pdf"

# 记录开始时间
start_time = time.time()

# 提取所有页面的表格
tables = camelot.read_pdf(pdf_path, pages="all")

end_time = time.time()
print(f"PDF 表格解析耗时: {end_time - start_time:.2f}秒")

# 处理所有表格
if tables:
    for i, table in enumerate(tables, 1):
        # 转换为 DataFrame
        df = table.df
        
        print(f"\n表格 {i} 数据:")
        print(df)
        
        print(f"\n表格 {i} 基本信息:")
        print(df.info())
        
        # 保存到 CSV
        csv_filename = f"billionaires_table_{i}.csv"
        df.to_csv(csv_filename, index=False)
        print(f"\n表格 {i} 已保存到 {csv_filename}")
```

**Camelot 参数详解：**

```python
tables = camelot.read_pdf(
    pdf_path,
    pages='1-5',           # 指定页面范围
    flavor='lattice',      # 'lattice' 或 'stream'
    table_areas=['10,500,590,100'],  # 指定表格区域
    columns=['100,200,300'],         # 指定列分隔线
    edge_tol=50,           # 边缘容差
    row_tol=2,             # 行容差
    split_text=True        # 分割单元格文本
)
```

**flavor 参数说明：**

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| `lattice` | 基于表格线识别 | 有明显边框的表格 |
| `stream` | 基于文本流识别 | 无边框或边框不清晰的表格 |

**特点：**

✅ 优势：
- 识别准确率高（基于矢量信息）
- 可以处理复杂表格
- 支持表格区域指定
- 输出为 DataFrame，易于处理

❌ 劣势：
- 依赖 Ghostscript
- 不支持扫描件
- 对无边框表格效果一般
- 速度较慢

## 方案三：pdfplumber 提取 PDF 表格

轻量级的 PDF 表格提取工具，速度快且易用。

**文件名：** `03-pdfplumber提取PDF表格.py`

```python
import pdfplumber
import pandas as pd
import time

start_time = time.time()

# 打开 PDF 文件
pdf = pdfplumber.open("../99-doc-data/复杂PDF/billionaires_page-1-5.pdf")

# 遍历每一页
for page in pdf.pages:
    # 提取表格
    tables = page.extract_tables()
    
    if tables:
        print(f"在第 {page.page_number} 页找到 {len(tables)} 个表格")
        
        for i, table in enumerate(tables):
            print(f"\n处理第 {i+1} 个表格:")
            
            # 转换为 DataFrame
            df = pd.DataFrame(table)
            
            # 设置第一行为列名
            if len(df) > 0:
                df.columns = df.iloc[0]
                df = df.iloc[1:]  # 删除重复的列名行
            
            print(df)
            print("-" * 50)

pdf.close()

end_time = time.time()
print(f"\nPDF 表格提取总耗时: {end_time - start_time:.2f}秒")
```

**pdfplumber 高级用法：**

```python
import pdfplumber

with pdfplumber.open("document.pdf") as pdf:
    page = pdf.pages[0]
    
    # 1. 自定义表格设置
    table_settings = {
        "vertical_strategy": "lines",    # 垂直线策略
        "horizontal_strategy": "lines",  # 水平线策略
        "explicit_vertical_lines": [],   # 显式垂直线
        "explicit_horizontal_lines": [], # 显式水平线
        "snap_tolerance": 3,             # 对齐容差
        "join_tolerance": 3,             # 连接容差
        "edge_min_length": 3,            # 最小边长
        "min_words_vertical": 3,         # 最小垂直单词数
        "min_words_horizontal": 1        # 最小水平单词数
    }
    
    tables = page.extract_tables(table_settings=table_settings)
    
    # 2. 提取表格边界框
    table_bboxes = page.find_tables()
    for bbox in table_bboxes:
        print(f"表格位置: {bbox.bbox}")
    
    # 3. 提取表格周围的文本
    for table in page.find_tables():
        # 表格上方的文本
        above_bbox = (0, 0, page.width, table.bbox[1])
        above_text = page.crop(above_bbox).extract_text()
        print(f"表格上方文本: {above_text}")
```

**特点：**

✅ 优势：
- 速度快，性能优秀
- 安装简单，无外部依赖
- 可以提取表格周围的文本
- 支持自定义表格识别参数
- 适合大批量处理

❌ 劣势：
- 对复杂表格识别率一般
- 不支持扫描件
- 需要手动处理表格格式

## 方案四：Unstructured 智能表格提取

最智能的表格提取方案，自动识别表格结构和上下文。

**文件名：** `05-01-unstructured表格提取.py`

```python
from unstructured.partition.pdf import partition_pdf

file_path = "../99-doc-data/复杂PDF/billionaires_page-1-5.pdf"

# 解析 PDF
elements = partition_pdf(
    file_path,
    strategy="hi_res"  # 高精度策略
)

# 创建元素映射
element_map = {element.id: element for element in elements if hasattr(element, 'id')}

# 遍历并打印表格信息
for element in elements:
    if element.category == "Table":
        print("\n表格数据:")
        print("表格元数据:", vars(element.metadata))
        print("表格内容:")
        print(element.text)
        
        # 获取父节点信息
        parent_id = getattr(element.metadata, 'parent_id', None)
        if parent_id and parent_id in element_map:
            parent_element = element_map[parent_id]
            print("\n父节点信息:")
            print(f"类型: {parent_element.category}")
            print(f"内容: {parent_element.text}")
        
        print("-" * 50)
```

**提取表格上下文：**

**文件名：** `05-02-unstructured表格提取+上下文.py`

```python
from unstructured.partition.pdf import partition_pdf

file_path = "../99-doc-data/复杂PDF/billionaires_page-1-5.pdf"

elements = partition_pdf(file_path)

# 创建元素索引映射
element_index_map = {i: element for i, element in enumerate(elements)}

for i, element in enumerate(elements):
    if element.category == "Table":
        print("\n表格数据:")
        print(element.text)
        
        # 打印表格前 3 个节点的内容（上下文）
        print("\n表格前 3 个节点内容:")
        for j in range(max(0, i-3), i):
            prev_element = element_index_map.get(j)
            if prev_element:
                print(f"节点 {j} ({prev_element.category}):")
                print(prev_element.text)
        
        print("-" * 50)
```

**推断表格结构：**

**文件名：** `05-03-unstructured表格提取推断表格结构.py`

```python
from unstructured.partition.pdf import partition_pdf

file_path = "../99-doc-data/复杂PDF/billionaires_page-1-5.pdf"

# 启用表格结构推断
elements = partition_pdf(
    file_path,
    strategy="hi_res",
    infer_table_structure=True  # 推断表格结构（HTML 格式）
)

for element in elements:
    if element.category == "Table":
        print("\n表格数据:")
        print("表格元数据:", vars(element.metadata))
        
        # 如果推断了结构，会有 text_as_html 属性
        if hasattr(element.metadata, 'text_as_html'):
            print("\nHTML 格式:")
            print(element.metadata.text_as_html)
        
        print("\n文本格式:")
        print(element.text)
        print("-" * 50)
```

**特点：**

✅ 优势：
- 自动识别表格和上下文
- 保留父子关系
- 可以推断表格结构（HTML）
- 支持扫描件
- 适合构建知识图谱

❌ 劣势：
- 速度较慢
- 依赖外部工具
- 内存占用较大


## 表格问答系统实战

将表格数据与 RAG 系统结合，实现智能问答。

**文件名：** `04-03-pdfplumber提取PDF表格并问答-DeepSeek.py`

```python
import os
from dotenv import load_dotenv
import pdfplumber
import pandas as pd

os.environ['USER_AGENT'] = 'my-rag-app/1.0'
load_dotenv()

# 1. 使用 pdfplumber 提取 PDF 表格
pdf_path = "../99-doc-data/复杂PDF/billionaires_page-1-5.pdf"

print("正在提取 PDF 表格...")
with pdfplumber.open(pdf_path) as pdf:
    tables = []
    for page in pdf.pages:
        table = page.extract_table()
        if table:
            tables.append(table)

# 2. 将表格转换为文本文档
from langchain_core.documents import Document

documents = []
if tables:
    for i, table in enumerate(tables, 1):
        # 转换为 DataFrame
        df = pd.DataFrame(table)
        
        # 转换为文本（更易读的格式）
        text = df.to_string(index=False)
        
        # 创建 Document 对象
        doc = Document(
            page_content=text,
            metadata={"source": f"表格{i}", "page": i}
        )
        documents.append(doc)

print(f"成功提取 {len(documents)} 个表格")

# 3. 文档分块
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,  # 表格可能较大
    chunk_overlap=200
)
all_splits = text_splitter.split_documents(documents)

print(f"分块后共 {len(all_splits)} 个文档片段")

# 4. 设置嵌入模型（本地 HuggingFace 模型）
from langchain_huggingface import HuggingFaceEmbeddings

print("正在加载 Embedding 模型...")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 5. 存储到向量数据库
from langchain_core.vectorstores import InMemoryVectorStore

print("正在构建向量索引...")
vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(all_splits)

# 6. 定义问题
questions = [
    "2023年谁是最富有的人?",
    "最年轻的富豪是谁?",
    "有哪些科技行业的富豪?"
]

# 7. 构建提示模板
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""
基于以下表格数据回答问题。如果表格中没有相关信息，就说没有找到对应信息。

表格数据:
{context}

问题: {question}

回答:
""")

# 8. 配置 DeepSeek LLM
from langchain_deepseek import ChatDeepSeek

llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0.7,
    max_tokens=2048,
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

# 9. 问答循环
print("\n" + "="*50)
print("开始问答")
print("="*50)

for question in questions:
    print(f"\n问题: {question}")
    
    # 检索相关文档
    retrieved_docs = vector_store.similarity_search(question, k=2)
    
    # 合并文档内容
    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
    
    # 生成答案
    answer = llm.invoke(prompt.format(question=question, context=docs_content))
    
    print(f"回答: {answer.content}")
    print("-"*50)
```

**核心流程：**

```
PDF 表格 → pdfplumber 提取 → 转文本 → 分块 → 向量化 → 存储 → 检索 → DeepSeek 生成答案
```

**输出示例：**

```
正在提取 PDF 表格...
成功提取 5 个表格
分块后共 5 个文档片段
正在加载 Embedding 模型...
正在构建向量索引...

==================================================
开始问答
==================================================

问题: 2023年谁是最富有的人?
回答: 根据表格数据，2023年最富有的人是埃隆·马斯克（Elon Musk），财富为2190亿美元。
--------------------------------------------------

问题: 最年轻的富豪是谁?
回答: 根据表格数据，最年轻的富豪是马克·扎克伯格（Mark Zuckerberg），年龄为39岁。
--------------------------------------------------
```

**扩展：使用 LCEL 链式调用**

```python
from langchain_core.runnables import RunnablePassthrough

# 构建检索链
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# 构建完整的 RAG 链
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# 使用链进行问答
answer = chain.invoke("2023年谁是最富有的人?")
print(answer.content)
```

## 方案对比与选择

### 性能对比

| 方案 | 速度 | 准确度 | 易用性 | 上下文 | 扫描件 |
|------|------|--------|--------|--------|--------|
| CSV Loader | ⚡⚡⚡ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ❌ | ❌ |
| Camelot | ⚡ | ⭐⭐⭐⭐ | ⭐⭐ | ❌ | ❌ |
| pdfplumber | ⚡⚡⚡ | ⭐⭐⭐ | ⭐⭐⭐⭐ | 部分 | ❌ |
| Unstructured | ⚡ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ✅ | ✅ |

### 选择建议

**使用 CSV Loader：**
- ✅ 数据已经是 CSV 格式
- ✅ 需要快速加载
- ✅ 表格结构简单

**使用 Camelot：**
- ✅ PDF 表格有明显边框
- ✅ 需要高准确率
- ✅ 表格结构复杂
- ✅ 可以接受较慢的速度

**使用 pdfplumber：**
- ✅ 需要快速提取
- ✅ 批量处理大量 PDF
- ✅ 需要提取表格周围的文本
- ✅ 表格结构相对规范

**使用 Unstructured：**
- ✅ 需要保留表格上下文
- ✅ 构建知识图谱
- ✅ 处理扫描件
- ✅ 需要表格结构推断

### 混合方案

实际项目中，可以结合多种方案：

```python
def smart_table_extractor(pdf_path):
    """智能选择表格提取方案"""
    import pdfplumber
    
    # 1. 先用 pdfplumber 快速检查
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[0]
        tables = page.extract_tables()
        
        # 2. 根据表格特征选择方案
        if not tables:
            # 没有表格，可能是扫描件
            from unstructured.partition.pdf import partition_pdf
            elements = partition_pdf(pdf_path, strategy="hi_res")
            return [el for el in elements if el.category == "Table"]
        
        elif len(tables) > 10:
            # 表格很多，用 pdfplumber（速度快）
            all_tables = []
            for page in pdf.pages:
                all_tables.extend(page.extract_tables())
            return all_tables
        
        else:
            # 表格较少，用 Camelot（准确率高）
            import camelot
            tables = camelot.read_pdf(pdf_path, pages="all")
            return [table.df for table in tables]

# 使用
tables = smart_table_extractor("document.pdf")
```

## 常见问题

### 1. Camelot 安装失败

```bash
# 错误：Ghostscript not found
# 解决：安装 Ghostscript

# macOS
brew install ghostscript

# Ubuntu
sudo apt-get install ghostscript

# 验证安装
gs --version
```

### 2. 表格识别不准确

```python
# 问题：表格边框不清晰

# 解决1：使用 Camelot 的 stream 模式
tables = camelot.read_pdf(pdf_path, flavor='stream')

# 解决2：使用 pdfplumber 自定义设置
table_settings = {
    "vertical_strategy": "text",
    "horizontal_strategy": "text"
}
tables = page.extract_tables(table_settings=table_settings)

# 解决3：使用 Unstructured 的高精度模式
elements = partition_pdf(pdf_path, strategy="hi_res")
```

### 3. 表格格式混乱

```python
# 问题：提取的表格格式不整齐

# 解决：使用 pandas 清理数据
import pandas as pd

df = pd.DataFrame(table)

# 1. 删除空行
df = df.dropna(how='all')

# 2. 删除空列
df = df.dropna(axis=1, how='all')

# 3. 重置索引
df = df.reset_index(drop=True)

# 4. 设置列名
df.columns = df.iloc[0]
df = df.iloc[1:]

# 5. 去除空格
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
```

### 4. 中文表格识别问题

```python
# 问题：中文表格识别不准

# 解决：使用 Unstructured 并指定语言
from unstructured.partition.pdf import partition_pdf

elements = partition_pdf(
    pdf_path,
    strategy="hi_res",
    languages=["chi_sim", "eng"]  # 中英文混合
)
```

### 5. 表格太大导致内存不足

```python
# 问题：大表格占用内存过多

# 解决：分批处理
def process_large_table(table, batch_size=100):
    """分批处理大表格"""
    df = pd.DataFrame(table)
    
    results = []
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        # 处理批次
        result = process_batch(batch)
        results.append(result)
    
    return results

# 使用
results = process_large_table(large_table, batch_size=100)
```


## 进阶技巧

### 1. 表格数据增强

```python
import pandas as pd

def enhance_table_data(df):
    """增强表格数据，添加描述性文本"""
    # 1. 添加列描述
    column_desc = f"表格包含以下列: {', '.join(df.columns)}"
    
    # 2. 添加统计信息
    stats = []
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            stats.append(f"{col}的平均值为{df[col].mean():.2f}")
    
    # 3. 添加行数信息
    row_info = f"表格共有{len(df)}行数据"
    
    # 4. 组合所有信息
    enhanced_text = f"{column_desc}\n{row_info}\n" + "\n".join(stats)
    enhanced_text += f"\n\n表格内容:\n{df.to_string(index=False)}"
    
    return enhanced_text

# 使用
df = pd.DataFrame(table)
enhanced_text = enhance_table_data(df)

# 创建 Document
doc = Document(
    page_content=enhanced_text,
    metadata={"source": "enhanced_table"}
)
```

### 2. 表格与文本混合索引

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
import pdfplumber

def create_mixed_index(pdf_path):
    """创建表格和文本的混合索引"""
    all_docs = []
    
    # 1. 提取文本
    text_loader = PyPDFLoader(pdf_path)
    text_docs = text_loader.load()
    all_docs.extend(text_docs)
    
    # 2. 提取表格
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            for i, table in enumerate(tables):
                df = pd.DataFrame(table)
                table_text = df.to_string(index=False)
                
                doc = Document(
                    page_content=table_text,
                    metadata={
                        "source": pdf_path,
                        "page": page_num,
                        "type": "table",
                        "table_index": i
                    }
                )
                all_docs.append(doc)
    
    return all_docs

# 使用
docs = create_mixed_index("document.pdf")

# 构建向量索引
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(docs)
```

### 3. 表格结构化查询

```python
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_deepseek import ChatDeepSeek

# 1. 提取表格并转换为 DataFrame
with pdfplumber.open(pdf_path) as pdf:
    table = pdf.pages[0].extract_table()
    df = pd.DataFrame(table)
    df.columns = df.iloc[0]
    df = df.iloc[1:]

# 2. 创建 Pandas Agent
llm = ChatDeepSeek(model="deepseek-chat")
agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=True,
    allow_dangerous_code=True  # 允许执行代码
)

# 3. 使用自然语言查询
result = agent.invoke("找出年龄最大的前3个人")
print(result)

result = agent.invoke("计算所有人的平均财富")
print(result)
```

### 4. 表格可视化

```python
import matplotlib.pyplot as plt
import pandas as pd

def visualize_table(df, chart_type='bar'):
    """可视化表格数据"""
    plt.figure(figsize=(10, 6))
    
    if chart_type == 'bar':
        df.plot(kind='bar')
    elif chart_type == 'line':
        df.plot(kind='line')
    elif chart_type == 'pie':
        df.plot(kind='pie', y=df.columns[0])
    
    plt.title('表格数据可视化')
    plt.tight_layout()
    plt.savefig('table_visualization.png')
    plt.close()

# 使用
df = pd.DataFrame(table)
visualize_table(df, chart_type='bar')
```

### 5. 表格数据验证

```python
def validate_table_data(df):
    """验证表格数据的完整性和准确性"""
    issues = []
    
    # 1. 检查空值
    null_counts = df.isnull().sum()
    if null_counts.any():
        issues.append(f"发现空值: {null_counts[null_counts > 0].to_dict()}")
    
    # 2. 检查重复行
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        issues.append(f"发现 {duplicates} 行重复数据")
    
    # 3. 检查数据类型
    for col in df.columns:
        if df[col].dtype == 'object':
            # 尝试转换为数字
            try:
                pd.to_numeric(df[col])
                issues.append(f"列 '{col}' 可能应该是数字类型")
            except:
                pass
    
    # 4. 检查异常值
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        mean = df[col].mean()
        std = df[col].std()
        outliers = df[(df[col] < mean - 3*std) | (df[col] > mean + 3*std)]
        if len(outliers) > 0:
            issues.append(f"列 '{col}' 发现 {len(outliers)} 个异常值")
    
    return issues

# 使用
df = pd.DataFrame(table)
issues = validate_table_data(df)
if issues:
    print("数据质量问题:")
    for issue in issues:
        print(f"- {issue}")
else:
    print("数据质量良好")
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