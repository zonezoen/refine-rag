from langchain_community.document_loaders import JSONLoader
# JSONLoader 依赖 jq 库来解析 JSON 文件。
# pip install jq
loader = JSONLoader(
    file_path="../99-doc-data/灭神纪/人物角色.json",
    jq_schema='.mainCharacter', # 用于text_content=False的时候
    # jq_schema='.mainCharacter | "姓名：" + .name + "，背景：" + .backstory',
    text_content=False
)
docs = loader.load()
print(docs)

'''
这两个参数用于控制如何从 JSON 文件中提取和格式化数据：

**1. `jq_schema`**：使用 jq 语法来提取和转换 JSON 数据

```python
jq_schema='.mainCharacter | "姓名：" + .name + "，背景：" + .backstory'
```

- `.mainCharacter`：从 JSON 中提取 `mainCharacter` 字段
- `|`：管道操作符，将结果传递给下一步
- `"姓名：" + .name + "，背景：" + .backstory`：格式化输出，拼接字符串

假设 JSON 是这样的：
```json
{
  "mainCharacter": {
    "name": "孙悟空",
    "backstory": "花果山美猴王"
  }
}
```

提取结果会是：
```
姓名：孙悟空，背景：花果山美猴王
```

**2. `text_content=True`**：控制输出格式

- `True`：将提取的内容作为**纯文本字符串**存储在 `page_content` 中
- `False`（默认）：将提取的内容作为 **JSON 对象**存储

**示例对比**：

```python
# text_content=True
Document(page_content="姓名：孙悟空，背景：花果山美猴王")

# text_content=False
Document(page_content='{"name": "孙悟空", "backstory": "花果山美猴王"}')
```

简单说：
- `jq_schema`：定义"提取什么数据、怎么格式化"
- `text_content`：定义"输出是文本还是 JSON"
'''
