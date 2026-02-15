from langchain_community.document_loaders import DirectoryLoader, TextLoader

# 如果有 markdown 文件，需要执行：pip install "unstructured[md]"
directory_loader = DirectoryLoader(
    '',
    # glob="**/*.txt",  # 只加载 .txt 文件
    # loader_cls=TextLoader,
    loader_kwargs={'encoding': 'utf-8'},
    use_multithreading=True,  # 多线程加载
    show_progress=True,  # 显示进度条
    silent_errors=True,  # 跳过错误
)
documents = directory_loader.load()
print(documents)

"""
没有 glob="**/*.txt"：
DirectoryLoader 会尝试加载目录中的所有文件（.txt、.py、.md、init.py 等）

没有 loader_cls=TextLoader：
DirectoryLoader 会使用 UnstructuredFileLoader 作为默认加载器
这个加载器会根据文件类型自动选择处理方式
对于 .md 文件，它需要 unstructured[md] 依赖
对于 .py 文件，它需要相应的 Python 文件解析依赖
"""
