from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter,RecursiveCharacterTextSplitter

loader = TextLoader("../99-doc-data/黑悟空/黑悟空wiki.txt")
data = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=5)
# chunk_size: 每个块的大小
# chunk_overlap: 块之间的重叠大小
chunks = text_splitter.split_documents(data)
for chunk in chunks:
    print("====== 切块分页 ======01")
    print(chunk.page_content)
