"""
多模态嵌入演示 - 使用 CLIP 模型

CLIP (Contrastive Language-Image Pre-training) 是什么？
- OpenAI 开发的多模态模型
- 可以理解图片和文本的关系
- 将图片和文本映射到同一个向量空间
- 支持零样本图像分类

优势：
- 安装简单：pip install transformers pillow
- 模型成熟稳定
- 支持多种语言（包括中文）
- 社区支持好

安装依赖：
pip install transformers pillow torch
"""

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
import os

def main():
    print("="*60)
    print("多模态嵌入演示 - CLIP 模型")
    print("="*60)
    
    # ========== 1. 加载 CLIP 模型 ==========
    print("\n【步骤1】加载 CLIP 模型...")
    print("首次运行会下载模型（约 600MB），请耐心等待...")
    
    # 使用中文优化的 CLIP 模型
    # 也可以使用原版：openai/clip-vit-base-patch32
    model_name = "OFA-Sys/chinese-clip-vit-base-patch16"
    
    try:
        # 加载模型和处理器
        model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)
        
        # 设置为评估模式
        model.eval()
        
        print("模型加载完成！")
        print(f"  - 模型: {model_name}")
        print(f"  - 支持: 图片和文本的联合编码")
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("\n尝试使用英文版 CLIP 模型...")
        model_name = "openai/clip-vit-base-patch32"
        model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)
        model.eval()
        print("英文版 CLIP 模型加载完成！")
    
    # ========== 2. 准备测试数据 ==========
    print("\n【步骤2】准备测试数据...")
    
    # 图片路径
    image_path = "../99-doc-data/多模态/query_image.jpg"
    
    # 检查图片是否存在
    if not os.path.exists(image_path):
        print(f"警告：图片不存在 - {image_path}")
        print("将创建一个测试图片...")
        
        # 创建一个简单的测试图片
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new('RGB', (224, 224), color='white')
        draw = ImageDraw.Draw(img)
        draw.text((50, 100), "Test Image", fill='black')
        
        # 保存测试图片
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        img.save(image_path)
        print(f"测试图片已创建: {image_path}")
    
    # 加载图片
    image = Image.open(image_path)
    print(f"图片路径: {image_path}")
    print(f"图片尺寸: {image.size}")
    
    # 准备文本
    texts = [
        "悟空在战斗",
        "孙悟空使用金箍棒",
        "一只猫在睡觉",
        "风景照片"
    ]
    
    print(f"\n测试文本:")
    for i, text in enumerate(texts, 1):
        print(f"  {i}. {text}")
    
    # ========== 3. 编码图片 ==========
    print("\n【步骤3】编码图片...")
    
    with torch.no_grad():
        # 处理图片
        # processor 会自动调整图片大小、归一化等
        inputs = processor(images=image, return_tensors="pt")
        
        # 获取图片的嵌入向量
        # vision_model 是 CLIP 的视觉编码器
        image_features = model.get_image_features(**inputs)
        
        # 归一化（使余弦相似度计算更准确）
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    print("图片编码完成！")
    print(f"  - 向量维度: {image_features.shape[1]}")
    print(f"  - 向量范数: {image_features.norm().item():.4f}")
    
    # ========== 4. 编码文本 ==========
    print("\n【步骤4】编码文本...")
    
    with torch.no_grad():
        # 处理文本
        # processor 会自动分词、添加特殊标记等
        text_inputs = processor(text=texts, return_tensors="pt", padding=True)
        
        # 获取文本的嵌入向量
        # text_model 是 CLIP 的文本编码器
        text_features = model.get_text_features(**text_inputs)
        
        # 归一化
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    print("文本编码完成！")
    print(f"  - 文本数量: {len(texts)}")
    print(f"  - 向量维度: {text_features.shape[1]}")
    
    # ========== 5. 计算图文相似度 ==========
    print("\n【步骤5】计算图文相似度...")
    
    # 计算图片和每个文本的相似度
    # 使用矩阵乘法：(1, 512) @ (512, 4) = (1, 4)
    similarities = (image_features @ text_features.T).squeeze(0)
    
    # 转换为 numpy 数组
    similarities_np = similarities.cpu().numpy()
    
    print("\n图片与文本的相似度:")
    for i, (text, sim) in enumerate(zip(texts, similarities_np), 1):
        print(f"  {i}. {text:20s} - 相似度: {sim:.4f}")
    
    # 找到最相似的文本
    best_match_idx = similarities_np.argmax()
    print(f"\n最匹配的文本: {texts[best_match_idx]} (相似度: {similarities_np[best_match_idx]:.4f})")
    
    # ========== 6. 零样本图像分类 ==========
    print("\n【步骤6】零样本图像分类...")
    
    # 定义类别
    categories = [
        "一张战斗场景的图片",
        "一张风景照片",
        "一张动物照片",
        "一张食物照片"
    ]
    
    with torch.no_grad():
        # 编码类别文本
        category_inputs = processor(text=categories, return_tensors="pt", padding=True)
        category_features = model.get_text_features(**category_inputs)
        category_features = category_features / category_features.norm(dim=-1, keepdim=True)
        
        # 计算相似度
        category_similarities = (image_features @ category_features.T).squeeze(0)
        
        # 转换为概率（使用 softmax）
        probs = torch.softmax(category_similarities * 100, dim=0)
    
    print("\n图片分类结果:")
    for category, prob in zip(categories, probs):
        print(f"  {category:30s} - 概率: {prob.item():.2%}")
    
    # ========== 7. 图片搜索图片示例 ==========
    print("\n【步骤7】图片搜索图片示例...")
    
    # 假设我们有多张图片
    print("\n说明：在实际应用中，可以这样实现图片搜索：")
    print("""
    1. 预先编码所有图片库中的图片
    2. 用户上传查询图片
    3. 编码查询图片
    4. 计算与图片库的相似度
    5. 返回最相似的图片
    
    示例代码：
    
    # 编码图片库
    image_database = []
    for img_path in image_paths:
        img = Image.open(img_path)
        inputs = processor(images=img, return_tensors="pt")
        features = model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        image_database.append(features)
    
    # 搜索
    query_features = encode_image(query_image)
    similarities = [
        (query_features @ db_features.T).item()
        for db_features in image_database
    ]
    
    # 返回最相似的图片
    top_k = sorted(
        zip(image_paths, similarities),
        key=lambda x: x[1],
        reverse=True
    )[:5]
    """)
    
    # ========== 8. 应用场景总结 ==========
    print("\n" + "="*60)
    print("【应用场景总结】")
    print("="*60)
    print("""
    1. 图文检索：
       ✅ 用文本搜索图片
       ✅ 用图片搜索文本
       ✅ 跨模态检索
    
    2. 零样本分类：
       ✅ 无需训练，直接分类
       ✅ 灵活定义类别
       ✅ 适合快速原型
    
    3. 图片相似度：
       ✅ 找到相似图片
       ✅ 图片去重
       ✅ 推荐系统
    
    4. 多模态 RAG：
       ✅ 文档包含图文
       ✅ 统一检索
       ✅ 更丰富的答案
    
    5. 电商应用：
       ✅ 以图搜图
       ✅ 商品推荐
       ✅ 视觉搜索
    """)
    
    # ========== 9. CLIP vs 其他模型 ==========
    print("\n" + "="*60)
    print("【CLIP vs 其他多模态模型】")
    print("="*60)
    print("""
    | 模型 | 优势 | 劣势 |
    |------|------|------|
    | CLIP | 安装简单、社区支持好 | 中文支持一般 |
    | Chinese-CLIP | 中文优化 | 模型较大 |
    | Visualized-BGE | 精度高 | 安装复杂 |
    | BLIP | 支持图像描述生成 | 计算成本高 |
    
    推荐：
    - 英文应用：openai/clip-vit-base-patch32
    - 中文应用：OFA-Sys/chinese-clip-vit-base-patch16
    - 高精度：Visualized-BGE（如果能安装）
    """)

if __name__ == '__main__':
    main()
