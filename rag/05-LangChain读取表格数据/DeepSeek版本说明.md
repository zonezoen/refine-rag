# PDF 表格问答 - LangChain + DeepSeek 版本

## 文件说明

`04-03-pdfplumber提取PDF表格并问答-DeepSeek.py` - 使用 LangChain + DeepSeek 实现的表格问答系统

## 核心流程

```
PDF 表格 → pdfplumber 提取 → 转文本 → 分块 → 向量化 → 存储 → 检索 → DeepSeek 生成答案
```

## 与 LlamaIndex 版本的区别

| 特性 | LlamaIndex 版本 | LangChain 版本 |
|------|----------------|---------------|
| **框架** | LlamaIndex | LangChain |
| **代码风格** | 高度封装 | 更灵活、步骤清晰 |
| **向量存储** | 自动处理 | 手动配置 |
| **提示词** | 内置模板 | 自定义模板 |
| **适合场景** | 快速原型 | 生产环境、自定义需求 |

## 代码结构

### 1. 提取表格
```python
with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        table = page.extract_table()
```

### 2. 转换为 LangChain Document
```python
doc = Document(
    page_content=text,
    metadata={"source": f"表格{i}"}
)
```

### 3. 文档分块
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200
)
all_splits = text_splitter.split_documents(documents)
```

### 4. 向量化（本地模型）
```python
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5"
)
```

### 5. 存储到向量数据库
```python
vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(all_splits)
```

### 6. 检索 + 生成答案
```python
# 检索
retrieved_docs = vector_store.similarity_search(question, k=2)

# 生成答案
answer = llm.invoke(prompt.format(question=question, context=docs_content))
```

## 安装依赖

```bash
# LangChain 核心
pip install langchain langchain-core langchain-community

# DeepSeek
pip install langchain-deepseek

# HuggingFace Embeddings
pip install langchain-huggingface sentence-transformers

# PDF 处理
pip install pdfplumber pandas

# 环境变量
pip install python-dotenv
```

## 配置环境变量

在 `.env` 文件中添加：
```bash
DEEPSEEK_API_KEY=your_deepseek_api_key_here
```

## 运行

```bash
python rag/05-LangChain读取表格数据/04-03-pdfplumber提取PDF表格并问答-DeepSeek.py
```

## 输出示例

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

## 优势

1. **完全使用 LangChain**：与项目其他代码风格一致
2. **本地 Embedding**：不需要为向量化付费
3. **DeepSeek API**：成本低、中文友好
4. **灵活可控**：每个步骤都可以自定义
5. **易于扩展**：可以轻松添加更多功能

## 扩展建议

### 1. 使用持久化向量数据库
```python
from langchain_community.vectorstores import Chroma

vector_store = Chroma.from_documents(
    documents=all_splits,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
```

### 2. 添加对话历史
```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
```

### 3. 使用 LCEL 链式调用
```python
from langchain_core.runnables import RunnablePassthrough

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)
answer = chain.invoke(question)
```

## 注意事项

1. 首次运行会下载 embedding 模型（约 100MB）
2. 确保 PDF 中的表格格式规范
3. 如果表格很大，可以调整 `chunk_size` 参数
4. 检索的 `k` 值可以根据需要调整（默认 k=2）
