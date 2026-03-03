# LangChain 读取文本数据

> 本文是 [refine-rag](https://github.com/zonezoen/refine-rag) 系列教程的第二篇，带你掌握 LangChain 的各种文本数据加载方式。
> 本文所有代码都在：https://github.com/zonezoen/refine-rag

## 目录

- 前言
- 环境准备
- 读取 TXT 文本
- 读取目录文件
- 读取 JSON 数据
- 读取网页数据
- 读取 Markdown 文件
- 整理父子元素结构
- 常见问题
- 下一步学习

## 前言

前面学习了 RAG 的概念和 LCEL 的语法，现在来继续学习 LangChain 读取文本数据的使用方法。在构建 RAG 系统时，数据加载是第一步，也是非常关键的一步。LangChain 提供了丰富的 Document Loaders，可以轻松处理各种格式的数据源。

本文将介绍如何使用 LangChain 读取：
- TXT 文本文件
- 目录批量文件
- JSON 结构化数据
- 网页数据
- Markdown 文档
- 带层级结构的复杂文档

## 环境准备

### 1. 安装依赖包

```bash
# 基础依赖
pip install langchain langchain-community

# JSON 解析依赖
pip install jq

# Markdown 和复杂文档解析
pip install "unstructured[md]"

# 网页加载依赖
pip install beautifulsoup4 lxml

# 一键安装所有依赖
pip install langchain langchain-community jq "unstructured[md]" beautifulsoup4 lxml
```

### 2. 准备测试数据

创建一个测试文本文件 `设定.txt`：

```
《黑神话：悟空》的故事可分为六个章节，讲述了天命人踏上取经路，寻找散落的六根遗物，揭开当年真相的故事。
```

## 读取 TXT 文本

最基础的文本加载方式，适用于纯文本文件。

**文件名：** `01读取txt文本.py`

```python
from langchain_community.document_loaders import TextLoader

# 加载单个 TXT 文件
loader = TextLoader("./设定.txt", encoding="utf-8")
docs = loader.load()
print(docs)

# 打印结果
"""
[Document(
    metadata={'source': './设定.txt'}, 
    page_content='《黑神话：悟空》的故事可分为六个章节...'
)]
"""

# 也可以加载 JSON 文件（作为纯文本）
json_loader = TextLoader("../99-doc-data/灭神纪/人物角色.json", encoding="utf-8")
json_docs = json_loader.load()
print(json_docs)
```

**关键点：**
- `encoding="utf-8"`：处理中文文件必须指定编码
- 返回的是 `Document` 对象列表，包含 `page_content`（内容）和 `metadata`（元数据）
- `TextLoader` 会将整个文件作为一个 Document 返回

## 读取目录文件

批量加载目录下的多个文件，支持文件过滤和多线程加载。

**文件名：** `02读取目录.py`

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# 加载目录下的所有文件
directory_loader = DirectoryLoader(
    '',  # 当前目录
    # glob="**/*.txt",  # 只加载 .txt 文件（可选）
    # loader_cls=TextLoader,  # 指定加载器类型（可选）
    loader_kwargs={'encoding': 'utf-8'},  # 传递给加载器的参数
    use_multithreading=True,  # 多线程加载，提升速度
    show_progress=True,  # 显示进度条
    silent_errors=True,  # 跳过无法加载的文件
)
documents = directory_loader.load()
print(f"加载了 {len(documents)} 个文档")
print(documents)
```

**参数说明：**

| 参数 | 说明 | 示例 |
|------|------|------|
| `glob` | 文件匹配模式 | `"**/*.txt"` 只加载 txt 文件 |
| `loader_cls` | 指定加载器类 | `TextLoader` 用于文本文件 |
| `loader_kwargs` | 加载器参数 | `{'encoding': 'utf-8'}` |
| `use_multithreading` | 是否多线程 | `True` 提升加载速度 |
| `show_progress` | 显示进度条 | `True` 方便查看进度 |
| `silent_errors` | 忽略错误 | `True` 跳过无法加载的文件 |

**注意事项：**
- 没有指定 `glob` 时，会尝试加载所有文件类型
- 没有指定 `loader_cls` 时，会使用 `UnstructuredFileLoader` 自动识别文件类型
- 对于 Markdown 文件，需要安装 `unstructured[md]` 依赖

## 读取 JSON 数据

使用 `jq` 语法提取和转换 JSON 数据，非常灵活。

**文件名：** `03读取json.py`

```python
from langchain_community.document_loaders import JSONLoader

# 需要先安装：pip install jq

loader = JSONLoader(
    file_path="../99-doc-data/灭神纪/人物角色.json",
    jq_schema='.mainCharacter',  # 提取 mainCharacter 字段
    # jq_schema='.mainCharacter | "姓名：" + .name + "，背景：" + .backstory',  # 格式化输出
    text_content=False  # False 返回 JSON 对象，True 返回纯文本
)
docs = loader.load()
print(docs)
```

**参数详解：**

**1. `jq_schema`**：使用 jq 语法提取和转换数据

假设 JSON 文件内容：
```json
{
  "mainCharacter": {
    "name": "孙悟空",
    "backstory": "花果山美猴王"
  }
}
```

不同的 `jq_schema` 效果：

```python
# 提取整个对象
jq_schema='.mainCharacter'
# 结果：{"name": "孙悟空", "backstory": "花果山美猴王"}

# 格式化输出
jq_schema='.mainCharacter | "姓名：" + .name + "，背景：" + .backstory'
# 结果：姓名：孙悟空，背景：花果山美猴王
```

**2. `text_content`**：控制输出格式

```python
# text_content=True：纯文本
Document(page_content="姓名：孙悟空，背景：花果山美猴王")

# text_content=False：JSON 字符串
Document(page_content='{"name": "孙悟空", "backstory": "花果山美猴王"}')
```

## 读取网页数据

直接从 URL 加载网页内容，适合爬取在线文档。

**文件名：** `04读取网页数据.py`

```python
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader(web_path="https://www.baidu.com")
docs = loader.load()
print(docs)
```

**使用场景：**
- 加载在线文档和教程
- 爬取新闻和博客文章
- 获取实时更新的内容

**注意事项：**
- 需要网络连接
- 某些网站可能需要设置 User-Agent
- 建议添加错误处理和重试机制

## 读取 Markdown 文件

Markdown 是技术文档的常用格式，LangChain 提供了专门的加载器。

**文件名：** `05读取markdown.py`

```python
from langchain_community.document_loaders import UnstructuredMarkdownLoader

loader = UnstructuredMarkdownLoader(
   "002：LangChain读取文本.md",
   mode="elements"  # 拆分成多个元素
)
docs = loader.load()
print(f"加载了 {len(docs)} 个元素")
for doc in docs:
   print(f"类型: {doc.metadata.get('category', 'Unknown')}")
   print(f"内容: {doc.page_content[:100]}...")
   print("-" * 50)
```

**`mode` 参数说明：**

```python
# mode="single"（默认）：整个文件作为一个 Document
[Document(page_content='# 标题\n\n内容...')]

# mode="elements"：按元素拆分（标题、段落、列表等）
[
    Document(page_content='标题', metadata={'category': 'Title'}),
    Document(page_content='段落内容', metadata={'category': 'NarrativeText'}),
    Document(page_content='列表项', metadata={'category': 'ListItem'})
]
```

## 整理父子元素结构

处理复杂文档时，保持元素的层级关系非常重要。有些时候上下文练习比较紧密，缺失上下文就无法回答问题。

**文件名：** `06-Unstrutured-整理父子元素.py`

```python
from langchain_unstructured import UnstructuredLoader
from typing import List
from langchain_core.documents import Document

# 维基百科页面（需要科学上网）
page_url = "https://zh.wikipedia.org/wiki/%E9%BB%91%E7%A5%9E%E8%AF%9D%EF%BC%9A%E6%82%9F%E7%A9%BA"

def _get_setup_docs_from_url(url: str) -> List[Document]:
    loader = UnstructuredLoader(
        web_url=url,
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }
    )
    
    setup_docs = []
    parent_id = None
    current_parent = None
    
    for doc in loader.load():
        # 标题和表格作为父元素
        if doc.metadata["category"] in ["Title", "Table"]:
            parent_id = doc.metadata["element_id"]
            current_parent = doc
            setup_docs.append(doc)
        # 子元素关联到父元素
        elif doc.metadata.get("parent_id") == parent_id:
            setup_docs.append((current_parent, doc))
    
    return setup_docs

# 加载并打印结构
docs = _get_setup_docs_from_url(page_url)
for item in docs:
    if isinstance(item, tuple):
        parent, child = item
        print(f'父元素 - {parent.metadata["category"]}: {parent.page_content}')
        print(f'子元素 - {child.metadata["category"]}: {child.page_content}')
    else:
        print(f'{item.metadata["category"]}: {item.page_content}')
    print("-" * 80)
```


**核心思路：**
1. 识别父元素（Title、Table 等）
2. 通过 `parent_id` 关联子元素
3. 保持文档的层级结构，便于后续处理

**应用场景：**
- 处理带有章节结构的长文档
- 保持表格和说明文字的关联
- 构建知识图谱时保留层级关系


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