from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("../99-doc-data/黑悟空/黑神话悟空.pdf")
data = loader.load()
for item in data:
    print(item.page_content)
