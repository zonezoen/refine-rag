"""
unstructured 库在处理 Office 文档（PPT、Word、Excel）时，需要使用 LibreOffice 的命令行工具 soffice 来转换文档。

在 Debian/Ubuntu 系统中，可以使用以下命令安装：
sudo apt-get update && sudo apt-get install -y libreoffice

- Install instructions: https://www.libreoffice.org/get-help/install-howto/
- Mac: https://formulae.brew.sh/cask/libreoffice
- Debian: https://wiki.debian.org/LibreOffice

解决方案（macOS）：
brew install libreoffice
"""
from unstructured.partition.ppt import partition_ppt
# 解析 PPT 文件
ppt_elements = partition_ppt(filename="../99-doc-data/黑悟空/黑神话悟空.pptx")
print("PPT 内容：")
for element in ppt_elements:
    print("=====分页=====")
    print(element.text)
    
from langchain_core.documents import Document
# 转换为 Documents 数据结构
documents = [
Document(page_content=element.text, 
  	     metadata={"source": "data/黑神话悟空PPT.pptx"})
    for element in ppt_elements
]

# 输出转换后的 Documents
print(documents)


