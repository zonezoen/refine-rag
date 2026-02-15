from langchain_unstructured import UnstructuredLoader
loader = UnstructuredLoader(
    file_path="../99-doc-data/山西文旅/云冈石窟-ch.pdf",  # PDF文件路径
    strategy="hi_res",    # 使用高分辨率策略进行文档处理
    # partition_via_api=True,  # 通过API进行文档分块
    # coordinates=True,     # 提取文本坐标信息
    languages=["chi_sim"]  # 简体中文-- chi_sim   英文--eng，
)

# 英文 PDF
# loader = UnstructuredLoader(
#     file_path="../99-doc-data/山西文旅/云冈石窟-en.pdf",  # PDF文件路径
#     strategy="hi_res",    # 使用高分辨率策略进行文档处理
#     # partition_via_api=True,  # 通过API进行文档分块
#     # coordinates=True,     # 提取文本坐标信息
# )

# # 下载简体中文语言包
# 方案一：
# brew install tesseract-lang
# 方案二：
# cd /opt/homebrew/share/tessdata
# sudo curl -L -O https://github.com/tesseract-ocr/tessdata/raw/main/chi_sim.traineddata

docs = []

# lazy_load() 是一种延迟加载方法
# 它不会一次性将所有文档加载到内存中，而是在需要时才逐个加载文档
# 这对于处理大型PDF文件时可以节省内存使用
for doc in loader.lazy_load():
    docs.append(doc)

print(docs)

# 输出结果是结构化的，再对结果进行结构化解析即可提取对应的内容
