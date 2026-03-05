"""
按段落和标题切块

这是最推荐的切块方式，特别适合：
- 说明书、论文、技术文档
- 有明确章节结构的文档
- Markdown 格式的文档

优势：
1. 保持逻辑完整性
2. 每个块都有明确的主题
3. 便于溯源和引用
"""

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# ============================================
# 方法1：Markdown 标题切块（推荐）
# ============================================

print("="*60)
print("方法1：Markdown 标题切块")
print("="*60)

# 示例 Markdown 文档
markdown_document = """
# 黑神话：悟空

## 游戏简介

《黑神话：悟空》是一款由游戏科学公司开发的动作角色扮演游戏。游戏基于中国古典名著《西游记》改编，讲述了孙悟空的传奇故事。

游戏采用虚幻引擎5开发，画面表现力极强。

## 游戏玩法

### 战斗系统

游戏的战斗系统流畅爽快，玩家可以使用金箍棒进行各种连招。

战斗中需要注意敌人的攻击模式，合理使用闪避和格挡。

### 技能系统

玩家可以学习72变等经典技能。每个技能都有独特的效果和使用场景。

技能可以通过游戏进程逐步解锁。

## 游戏世界

### 场景设计

游戏包含多个精心设计的场景，如黑风山、火焰山等。

每个场景都有独特的视觉风格和敌人类型。

### 探索要素

玩家可以在场景中探索，寻找隐藏的宝箱和秘密。

探索会获得额外的奖励和装备。
"""

# 定义要按哪些标题级别切块
headers_to_split_on = [
    ("#", "一级标题"),      # H1
    ("##", "二级标题"),     # H2
    ("###", "三级标题"),    # H3
]

# 创建 Markdown 标题切块器
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=False  # 保留标题在内容中
)

# 执行切块
md_header_splits = markdown_splitter.split_text(markdown_document)

print(f"\n生成的块数: {len(md_header_splits)}")
print("\n切块结果:")

for i, chunk in enumerate(md_header_splits, 1):
    print(f"\n--- 第 {i} 个块 ---")
    print(f"内容: {chunk.page_content}")
    print(f"元数据: {chunk.metadata}")
    print("-" * 50)

# ============================================
# 方法2：段落切块（双换行符分割）
# ============================================

print("\n" + "="*60)
print("方法2：段落切块（双换行符分割）")
print("="*60)

# 示例文本（段落之间用双换行符分隔）
paragraph_document = """
《黑神话：悟空》是一款由游戏科学公司开发的动作角色扮演游戏。游戏基于中国古典名著《西游记》改编。

游戏采用虚幻引擎5开发，画面表现力极强。战斗系统流畅爽快，深受玩家喜爱。

玩家在游戏中扮演孙悟空的转世，踏上寻找真相的旅程。游戏包含多个精心设计的场景。

每个场景都有独特的视觉风格和敌人类型。玩家需要不断提升自己的战斗技巧。
"""

# 使用递归切块器，优先按段落分割
paragraph_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", "。", " ", ""],  # 优先按双换行符（段落）分割
    chunk_size=500,
    chunk_overlap=50,
    length_function=len
)

paragraph_chunks = paragraph_splitter.split_text(paragraph_document)

print(f"\n生成的块数: {len(paragraph_chunks)}")
print("\n切块结果:")

for i, chunk in enumerate(paragraph_chunks, 1):
    print(f"\n--- 第 {i} 个段落块 ---")
    print(chunk)
    print("-" * 50)

# ============================================
# 方法3：Markdown + 段落混合切块（推荐）
# ============================================

print("\n" + "="*60)
print("方法3：Markdown + 段落混合切块（推荐）")
print("="*60)

# 先按标题切块
md_header_splits = markdown_splitter.split_text(markdown_document)

# 再对每个标题块进行段落切块（如果块太大）
final_chunks = []

for header_chunk in md_header_splits:
    # 如果块太大，再按段落切分
    if len(header_chunk.page_content) > 200:
        # 使用段落切块器
        sub_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", "。", " ", ""],
            chunk_size=200,
            chunk_overlap=20
        )
        
        sub_chunks = sub_splitter.split_text(header_chunk.page_content)
        
        # 为每个子块保留原始的标题元数据
        for sub_chunk in sub_chunks:
            from langchain_core.documents import Document
            final_chunks.append(Document(
                page_content=sub_chunk,
                metadata=header_chunk.metadata  # 保留标题信息
            ))
    else:
        final_chunks.append(header_chunk)

print(f"\n生成的块数: {len(final_chunks)}")
print("\n切块结果:")

for i, chunk in enumerate(final_chunks, 1):
    print(f"\n--- 第 {i} 个混合块 ---")
    print(f"内容: {chunk.page_content[:100]}...")
    print(f"元数据: {chunk.metadata}")
    print("-" * 50)
    
    if i >= 5:
        print(f"\n... 还有 {len(final_chunks) - 5} 个块 ...")
        break

# ============================================
# 方法4：从 Markdown 文件读取并切块
# ============================================

print("\n" + "="*60)
print("方法4：从 Markdown 文件读取并切块")
print("="*60)

# 创建一个示例 Markdown 文件
sample_md_content = """
# 黑神话：悟空攻略

## 第一章：黑风山

### 主线任务

在黑风山，你需要击败黑风大王。注意观察他的攻击模式。

使用闪避可以有效躲避他的重击。

### 支线任务

探索黑风山的隐藏区域，可以找到珍贵的装备。

记得与NPC对话，获取更多线索。

## 第二章：黄风岭

### 主线任务

黄风岭的敌人会使用风系攻击。建议提前准备抗风装备。

BOSS战需要注意走位，避免被风暴卷入。

### 收集要素

黄风岭有多个收集品，完成收集可以解锁成就。
"""

# 保存到文件
with open("sample_guide.md", "w", encoding="utf-8") as f:
    f.write(sample_md_content)

# 从文件加载
from langchain_community.document_loaders import UnstructuredMarkdownLoader

md_loader = UnstructuredMarkdownLoader("sample_guide.md")
md_docs = md_loader.load()

print(f"\n加载的文档数: {len(md_docs)}")

# 按标题切块
md_text = md_docs[0].page_content
md_chunks = markdown_splitter.split_text(md_text)

print(f"切块后的块数: {len(md_chunks)}")

for i, chunk in enumerate(md_chunks, 1):
    print(f"\n--- 第 {i} 个块 ---")
    print(f"内容: {chunk.page_content[:100]}...")
    print(f"元数据: {chunk.metadata}")
    print("-" * 50)

# 清理示例文件
import os
if os.path.exists("sample_guide.md"):
    os.remove("sample_guide.md")
    print("\n已清理示例文件")

# ============================================
# 方法5：HTML 标题切块
# ============================================

print("\n" + "="*60)
print("方法5：HTML 标题切块")
print("="*60)

from langchain_text_splitters import HTMLHeaderTextSplitter

html_string = """
<!DOCTYPE html>
<html>
<body>
    <h1>黑神话：悟空</h1>
    <p>《黑神话：悟空》是一款动作角色扮演游戏。</p>
    
    <h2>游戏特色</h2>
    <p>游戏采用虚幻引擎5开发，画面表现力极强。</p>
    
    <h3>战斗系统</h3>
    <p>战斗系统流畅爽快，玩家可以使用金箍棒进行各种连招。</p>
    
    <h3>技能系统</h3>
    <p>玩家可以学习72变等经典技能。</p>
    
    <h2>游戏世界</h2>
    <p>游戏包含多个精心设计的场景。</p>
</body>
</html>
"""

# 定义要按哪些 HTML 标签切块
headers_to_split_on = [
    ("h1", "一级标题"),
    ("h2", "二级标题"),
    ("h3", "三级标题"),
]

html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
html_header_splits = html_splitter.split_text(html_string)

print(f"\n生成的块数: {len(html_header_splits)}")

for i, chunk in enumerate(html_header_splits, 1):
    print(f"\n--- 第 {i} 个 HTML 块 ---")
    print(f"内容: {chunk.page_content}")
    print(f"元数据: {chunk.metadata}")
    print("-" * 50)

# ============================================
# 总结和建议
# ============================================

print("\n" + "="*80)
print("总结和建议")
print("="*80)

print("""
1. Markdown 标题切块（最推荐）：
   - 适合技术文档、说明书、论文
   - 保留完整的层级结构
   - 元数据包含标题信息，便于溯源

2. 段落切块：
   - 适合普通文章、博客
   - 按双换行符分割
   - 简单有效

3. 混合切块（推荐）：
   - 先按标题切块
   - 再对大块进行段落切分
   - 平衡了结构和大小

4. HTML 标题切块：
   - 适合网页内容
   - 与 Markdown 类似

5. 选择建议：
   - 有标题结构 → 使用标题切块
   - 纯文本 → 使用段落切块
   - 大型文档 → 使用混合切块
""")
