# 01
from langchain_community.document_loaders import UnstructuredCSVLoader
unstructLoader = UnstructuredCSVLoader("../99-doc-data/黑悟空/黑神话悟空.csv")
unstructDocuments = unstructLoader.load()
print(unstructDocuments)

# 02
from langchain_community.document_loaders import CSVLoader
loader = CSVLoader("../99-doc-data/黑悟空/黑神话悟空.csv")
documents = loader.load()
print(documents)

# 03
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import CSVLoader

loader = DirectoryLoader(
    path="../99-doc-data",  # Specify the directory containing your CSV files
    glob="**/*.csv",                # Use a glob pattern to match CSV files
    loader_cls=CSVLoader            # Specify CSVLoader as the loader class
)

docs = loader.load()
print(f"文档数：{len(docs)}")  # 输出文档总数
print(docs[0])