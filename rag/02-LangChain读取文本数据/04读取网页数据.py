from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader(web_path="https://www.baidu.com")
docs = loader.load()
print(docs)