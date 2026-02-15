"""
使用 unstructured 库进行 PDF 表格提取

【系统依赖安装】
macOS:
- brew install poppler
- brew install tesseract
- brew install tesseract-lang  # 中文支持

【验证安装】
- pdfinfo -v
- tesseract --version
"""

import os
import sys
from pathlib import Path
from unstructured.partition.pdf import partition_pdf

# 确保工作目录正确
script_dir = Path(__file__).parent.parent.parent
if script_dir.exists():
    os.chdir(script_dir)
    print(f"工作目录设置为: {os.getcwd()}")

file_path = "/Users/zonezone/Desktop/work/refine-rag/rag/99-doc-data/复杂PDF/billionaires_page-1-5.pdf"

# 检查文件是否存在
if not os.path.exists(file_path):
    print(f"错误: 文件不存在 - {file_path}")
    print(f"当前工作目录: {os.getcwd()}")
    print("请确保:")
    print("1. 在项目根目录运行脚本")
    print("2. PDF文件路径正确")
    sys.exit(1)

print(f"正在处理文件: {file_path}")

elements = partition_pdf(
    file_path,
    strategy="hi_res",  # 使用高精度策略
)

# 创建元素ID到元素的映射（用于查找父节点）
element_map = {element.id: element for element in elements if hasattr(element, 'id')}

# 遍历并打印表格信息
for element in elements:
    if element.category == "Table":
        print("\n表格数据:")
        print("表格元数据:", vars(element.metadata))
        print("表格内容:")
        print(element.text)
        
        # 获取并打印父节点信息
        parent_id = getattr(element.metadata, 'parent_id', None)
        if parent_id and parent_id in element_map:
            parent_element = element_map[parent_id]
            print("\n父节点信息:")
            print(f"类型: {parent_element.category}")
            print(f"内容: {parent_element.text}")
            if hasattr(parent_element, 'metadata'):
                print(f"父节点元数据: {vars(parent_element.metadata)}")
        else:
            print(f"未找到父节点 (ID: {parent_id})")
        print("-" * 50)


