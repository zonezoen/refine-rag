from langchain_community.document_loaders import TextLoader

loader = TextLoader("./设定.txt", encoding="utf-8")
docs = loader.load()
print(docs)

# 打印结果
"""
[Document(
metadata={'source': './设定.txt'}, 
page_content='《黑神话：悟空》的故事可分为六个章节，省略...')]
"""

json_loader = TextLoader("../99-doc-data/灭神纪/人物角色.json", encoding="utf-8")
json_docs = json_loader.load()
print(json_docs)
