from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter,RecursiveCharacterTextSplitter

loader = TextLoader("../99-doc-data/黑悟空/黑悟空wiki.txt")
data = loader.load()

# 定义分割符列表，按优先级依次使用
# RecursiveCharacterTextSplitter会按照这个顺序尝试分割
separators = [
    "\n\n",  # 双换行符（段落分隔）
    "\n",    # 单换行符（行分隔）
    "。",    # 中文句号
    ".",     # 英文句号
    "！",    # 中文感叹号
    "!",     # 英文感叹号
    "？",    # 中文问号
    "?",     # 英文问号
    "；",    # 中文分号
    ";",     # 英文分号
    "，",    # 中文逗号
    ",",     # 英文逗号
    " ",     # 空格
    ""       # 最后按字符分割
]

recursive_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=5,
    separators=separators,
    length_function=len
)

r_chunks = recursive_text_splitter.split_documents(data)
for chunk in r_chunks:
    print("====== 递归切块分页02 ======")
    print(chunk.page_content)