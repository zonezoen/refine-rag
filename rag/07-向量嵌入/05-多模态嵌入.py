"""
多模态嵌入演示：使用 Visualized-BGE 模型对图片和文本进行联合编码

什么是多模态嵌入？
- 将不同模态（图片、文本）映射到同一个向量空间
- 可以实现图片搜索文本、文本搜索图片、图片搜索图片
- 适用于图文检索、视觉问答等场景

Visualized-BGE 模型：
- 基于 BGE（BAAI General Embedding）扩展
- 支持图片和文本的联合编码
- 可以单独编码图片，也可以图文联合编码

安装依赖：
pip install visual_bge torch pillow numpy

下载模型权重：
wget https://huggingface.co/BAAI/bge-visualized/resolve/main/Visualized_base_en_v1.5.pth

参考文档：
https://github.com/FlagOpen/FlagEmbedding/tree/master/research/visual_bge
"""

import torch
from visual_bge.modeling import Visualized_BGE
from PIL import Image
import numpy as np
import os

def main():
    print("="*60)
    print("多模态嵌入演示")
    print("="*60)
    
    # ========== 1. 初始化模型 ==========
    print("\n【步骤1】初始化 Visualized-BGE 模型...")
    
    # BGE 文本编码器的模型名称
    # 这是预训练的文本嵌入模型，用于处理文本部分
    model_name = "BAAI/bge-base-en-v1.5"
    
    # 视觉编码器的权重文件路径
    # 这个权重文件包含了图片编码器的参数
    # 需要提前下载到本地
    model_path = "../99-doc-data/多模态/Visualized_base_en_v1.5.pth"
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误：模型文件不存在 - {model_path}")
        print("请先下载模型权重文件：")
        print("wget https://huggingface.co/BAAI/bge-visualized/resolve/main/Visualized_base_en_v1.5.pth")
        return
    
    # 创建 Visualized-BGE 模型实例
    # model_name_bge: 文本编码器（BGE）
    # model_weight: 视觉编码器权重
    model = Visualized_BGE(
        model_name_bge=model_name,
        model_weight=model_path
    )
    
    # 设置为评估模式（不进行训练，只做推理）
    # 这会关闭 dropout 等训练时的随机性
    model.eval()
    
    print("模型加载完成！")
    print(f"  - 文本编码器: {model_name}")
    print(f"  - 视觉编码器: {model_path}")
    
    # ========== 2. 准备测试图片 ==========
    print("\n【步骤2】准备测试图片...")
    
    # 图片路径（相对路径）
    image_path = "../99-doc-data/多模态/query_image.jpg"
    
    # 检查图片是否存在
    if not os.path.exists(image_path):
        print(f"错误：图片文件不存在 - {image_path}")
        print("请确保图片文件存在")
        return
    
    print(f"图片路径: {image_path}")
    
    # 可选：显示图片信息
    try:
        img = Image.open(image_path)
        print(f"图片尺寸: {img.size}")
        print(f"图片模式: {img.mode}")
    except Exception as e:
        print(f"无法读取图片: {e}")
        return
    
    # ========== 3. 编码图片（纯视觉） ==========
    print("\n【步骤3】编码图片（纯视觉模态）...")
    
    # 使用 torch.no_grad() 上下文管理器
    # 作用：禁用梯度计算，节省内存，加快推理速度
    # 因为我们只做推理，不需要反向传播
    with torch.no_grad():
        # 仅使用图片进行编码
        # 输入：图片路径
        # 输出：图片的向量表示（tensor）
        image_embedding = model.encode(image=image_path)
    
    print("图片编码完成！")
    print(f"  - 输入: 图片")
    print(f"  - 输出: 向量 (tensor)")
    
    # ========== 4. 编码图片+文本（多模态） ==========
    print("\n【步骤4】编码图片+文本（多模态）...")
    
    # 文本描述
    # 这段文本会和图片一起编码，生成融合了图文信息的向量
    text = "这是一张悟空战斗示例图片"
    
    with torch.no_grad():
        # 同时使用图片和文本进行编码
        # 输入：图片路径 + 文本描述
        # 输出：融合了图文信息的向量（tensor）
        multimodal_embedding = model.encode(
            image=image_path,
            text=text
        )
    
    print("多模态编码完成！")
    print(f"  - 输入: 图片 + 文本")
    print(f"  - 文本: {text}")
    print(f"  - 输出: 向量 (tensor)")
    
    # ========== 5. 转换为 NumPy 数组 ==========
    print("\n【步骤5】转换为 NumPy 数组...")
    
    # 将 PyTorch tensor 转换为 NumPy 数组
    # .cpu(): 将 tensor 从 GPU 移到 CPU（如果在 GPU 上）
    # .numpy(): 转换为 NumPy 数组（方便后续处理）
    image_embedding_np = image_embedding.cpu().numpy()
    multimodal_embedding_np = multimodal_embedding.cpu().numpy()
    
    print("转换完成！")
    
    # ========== 6. 分析嵌入向量 ==========
    print("\n" + "="*60)
    print("【结果分析】")
    print("="*60)
    
    # 分析纯图片嵌入
    print("\n1. 纯图片嵌入向量:")
    print(f"   - 形状: {image_embedding_np.shape}")
    print(f"   - 维度: {image_embedding_np.shape[1]}")
    print(f"   - 前10个元素: {image_embedding_np[0][:10]}")
    print(f"   - 向量范数: {np.linalg.norm(image_embedding_np[0]):.4f}")
    print(f"   - 说明: 只包含图片的视觉信息")
    
    # 分析多模态嵌入
    print("\n2. 多模态嵌入向量:")
    print(f"   - 形状: {multimodal_embedding_np.shape}")
    print(f"   - 维度: {multimodal_embedding_np.shape[1]}")
    print(f"   - 前10个元素: {multimodal_embedding_np[0][:10]}")
    print(f"   - 向量范数: {np.linalg.norm(multimodal_embedding_np[0]):.4f}")
    print(f"   - 说明: 融合了图片和文本的信息")
    
    # 计算两个向量的相似度
    print("\n3. 向量相似度分析:")
    
    # 余弦相似度 = 向量点积 / (向量范数的乘积)
    # 范围：[-1, 1]，越接近1表示越相似
    cosine_similarity = np.dot(
        image_embedding_np[0],
        multimodal_embedding_np[0]
    ) / (
        np.linalg.norm(image_embedding_np[0]) *
        np.linalg.norm(multimodal_embedding_np[0])
    )
    
    print(f"   - 余弦相似度: {cosine_similarity:.4f}")
    print(f"   - 说明: 两个向量的相似程度")
    print(f"   - 解释: 添加文本描述后，向量发生了变化")
    
    # ========== 7. 应用场景说明 ==========
    print("\n" + "="*60)
    print("【应用场景】")
    print("="*60)
    print("""
    1. 图文检索：
       - 用文本搜索图片："悟空战斗" → 找到相关图片
       - 用图片搜索文本：上传图片 → 找到相关描述
    
    2. 图片搜索图片：
       - 上传一张图片，找到相似的图片
       - 基于视觉相似度，不需要文本标签
    
    3. 视觉问答：
       - 问题："图片中的角色在做什么？"
       - 将问题和图片编码，生成答案
    
    4. 多模态 RAG：
       - 文档包含图片和文字
       - 用户提问，检索相关的图文内容
       - 生成包含图片信息的答案
    
    5. 电商搜索：
       - 用户上传商品图片
       - 找到相似的商品和描述
    """)
    
    # ========== 8. 技术细节说明 ==========
    print("\n" + "="*60)
    print("【技术细节】")
    print("="*60)
    print("""
    向量维度：
    - 通常是 768 或 1024 维
    - 维度越高，表达能力越强，但计算成本越高
    
    向量范数：
    - 向量的长度（欧几里得距离）
    - 归一化后的向量范数接近 1
    
    余弦相似度：
    - 衡量两个向量的方向相似度
    - 范围：[-1, 1]
    - 1: 完全相同
    - 0: 正交（无关）
    - -1: 完全相反
    
    为什么需要多模态嵌入？
    - 传统方法：图片和文本分别处理，无法关联
    - 多模态嵌入：统一向量空间，可以跨模态检索
    """)

if __name__ == '__main__':
    main() 