#!/bin/bash

# Visual-BGE 多模态检索安装脚本
# 使用方法: bash install.sh

echo "=========================================="
echo "Visual-BGE 多模态检索环境安装"
echo "=========================================="
echo ""

# 检查是否已安装 git
if ! command -v git &> /dev/null; then
    echo "错误: 未找到 git，请先安装 git"
    exit 1
fi

# 1. 克隆 FlagEmbedding 仓库
echo "步骤 1/4: 克隆 FlagEmbedding 仓库..."
if [ -d "FlagEmbedding" ]; then
    echo "FlagEmbedding 目录已存在，跳过克隆"
else
    git clone https://github.com/FlagOpen/FlagEmbedding.git
    if [ $? -ne 0 ]; then
        echo "错误: 克隆仓库失败"
        exit 1
    fi
fi

# 2. 安装 visual_bge
echo ""
echo "步骤 2/4: 安装 visual_bge 包..."
cd FlagEmbedding/research/visual_bge
pip install -e .
if [ $? -ne 0 ]; then
    echo "错误: 安装 visual_bge 失败"
    exit 1
fi
cd ../../..

# 3. 安装其他依赖
echo ""
echo "步骤 3/4: 安装其他依赖包..."
pip install torchvision timm einops ftfy
if [ $? -ne 0 ]; then
    echo "错误: 安装依赖包失败"
    exit 1
fi

# 4. 提示下载模型
echo ""
echo "步骤 4/4: 下载模型权重"
echo "=========================================="
echo "请选择要下载的模型："
echo "1) 英文模型 (bge-visualized-base-en-v1.5)"
echo "2) 多语言模型/中文 (bge-visualized-m3)"
echo "3) 两个都下载"
echo "4) 跳过（稍后手动下载）"
echo ""
read -p "请输入选项 (1-4): " choice

case $choice in
    1)
        echo "下载英文模型..."
        mkdir -p models
        huggingface-cli download BAAI/bge-visualized-base-en-v1.5 --local-dir ./models/bge-visualized-base-en-v1.5
        ;;
    2)
        echo "下载多语言模型..."
        mkdir -p models
        huggingface-cli download BAAI/bge-visualized-m3 --local-dir ./models/bge-visualized-m3
        ;;
    3)
        echo "下载两个模型..."
        mkdir -p models
        huggingface-cli download BAAI/bge-visualized-base-en-v1.5 --local-dir ./models/bge-visualized-base-en-v1.5
        huggingface-cli download BAAI/bge-visualized-m3 --local-dir ./models/bge-visualized-m3
        ;;
    4)
        echo "跳过模型下载"
        echo ""
        echo "请稍后手动下载模型："
        echo "英文模型: https://huggingface.co/BAAI/bge-visualized-base-en-v1.5"
        echo "多语言模型: https://huggingface.co/BAAI/bge-visualized-m3"
        ;;
    *)
        echo "无效选项，跳过模型下载"
        ;;
esac

echo ""
echo "=========================================="
echo "安装完成！"
echo "=========================================="
echo ""
echo "下一步："
echo "1. 如果还没下载模型，请使用以下命令："
echo "   huggingface-cli download BAAI/bge-visualized-m3 --local-dir ./models/bge-visualized-m3"
echo ""
echo "2. 运行示例程序："
echo "   python Milvus+Visual-BGE多模态检索-中文.py"
echo ""
echo "详细文档请查看: 安装说明.md"
