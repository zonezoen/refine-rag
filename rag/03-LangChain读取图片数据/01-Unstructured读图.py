from langchain_community.document_loaders import UnstructuredImageLoader
# pip install pdfminer.six
# pip install "unstructured[all-docs]"
image_path = "../99-doc-data/黑悟空/黑悟空英文.jpg"
loader = UnstructuredImageLoader(image_path)

data = loader.load()
print(data)

# 结果：
"""
Warning: No languages specified, defaulting to English.
[Document(metadata={'source': '../99-doc-data/黑悟空/黑悟空英文.jpg'}, page_content='2\n\nPons\n\nBLACK MYTH. WUKONGY\n\n4')]
"""