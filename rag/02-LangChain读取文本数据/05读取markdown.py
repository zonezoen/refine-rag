from langchain_community.document_loaders import UnstructuredMarkdownLoader

loader = UnstructuredMarkdownLoader("002：LangChain读取文本.md", mode="elements")
docs = loader.load()
print(docs)

"""
mode="elements"
可以控制返回的结果是否拆分成多个分块，否则只会返回一整块信息：
[Document(metadata={'source': './002：LangChain读取文本.md'}, page_content='Test\n\n测试')]
"""