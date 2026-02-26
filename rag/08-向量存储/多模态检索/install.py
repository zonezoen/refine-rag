"""
Visual-BGE 多模态检索环境安装脚本
使用方法: python install.py
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """运行命令并显示进度"""
    print(f"\n{description}...")
    print(f"执行命令: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"错误: {description}失败")
        print(f"错误信息: {result.stderr}")
        return False
    
    print(f"✓ {description}成功")
    return True

def main():
    print("=" * 50)
    print("Visual-BGE 多模态检索环境安装")
    print("=" * 50)
    
    # 检查 Python 版本
    if sys.version_info < (3, 8):
        print("错误: 需要 Python 3.8 或更高版本")
        sys.exit(1)
    
    print(f"Python 版本: {sys.version}")
    
    # 1. 克隆 FlagEmbedding 仓库
    print("\n步骤 1/4: 克隆 FlagEmbedding 仓库")
    if Path("FlagEmbedding").exists():
        print("FlagEmbedding 目录已存在，跳过克隆")
    else:
        if not run_command(
            "git clone https://github.com/FlagOpen/FlagEmbedding.git",
            "克隆 FlagEmbedding 仓库"
        ):
            sys.exit(1)
    
    # 2. 安装 visual_bge
    print("\n步骤 2/4: 安装 visual_bge 包")
    original_dir = os.getcwd()
    try:
        os.chdir("FlagEmbedding/research/visual_bge")
        if not run_command(
            f"{sys.executable} -m pip install -e .",
            "安装 visual_bge"
        ):
            os.chdir(original_dir)
            sys.exit(1)
        os.chdir(original_dir)
    except Exception as e:
        print(f"错误: {e}")
        os.chdir(original_dir)
        sys.exit(1)
    
    # 3. 安装其他依赖
    print("\n步骤 3/4: 安装其他依赖包")
    dependencies = ["torchvision", "timm", "einops", "ftfy"]
    for dep in dependencies:
        if not run_command(
            f"{sys.executable} -m pip install {dep}",
            f"安装 {dep}"
        ):
            print(f"警告: {dep} 安装失败，请手动安装")
    
    # 4. 提示下载模型
    print("\n步骤 4/4: 下载模型权重")
    print("=" * 50)
    print("请选择要下载的模型：")
    print("1) 英文模型 (bge-visualized-base-en-v1.5)")
    print("2) 多语言模型/中文 (bge-visualized-m3) - 推荐")
    print("3) 两个都下载")
    print("4) 跳过（稍后手动下载）")
    
    choice = input("\n请输入选项 (1-4): ").strip()
    
    # 创建 models 目录
    Path("models").mkdir(exist_ok=True)
    
    if choice == "1":
        print("\n下载英文模型...")
        run_command(
            "huggingface-cli download BAAI/bge-visualized-base-en-v1.5 --local-dir ./models/bge-visualized-base-en-v1.5",
            "下载英文模型"
        )
    elif choice == "2":
        print("\n下载多语言模型...")
        run_command(
            "huggingface-cli download BAAI/bge-visualized-m3 --local-dir ./models/bge-visualized-m3",
            "下载多语言模型"
        )
    elif choice == "3":
        print("\n下载两个模型...")
        run_command(
            "huggingface-cli download BAAI/bge-visualized-base-en-v1.5 --local-dir ./models/bge-visualized-base-en-v1.5",
            "下载英文模型"
        )
        run_command(
            "huggingface-cli download BAAI/bge-visualized-m3 --local-dir ./models/bge-visualized-m3",
            "下载多语言模型"
        )
    else:
        print("\n跳过模型下载")
        print("\n请稍后手动下载模型：")
        print("英文模型: https://huggingface.co/BAAI/bge-visualized-base-en-v1.5")
        print("多语言模型: https://huggingface.co/BAAI/bge-visualized-m3")
        print("\n或使用命令：")
        print("huggingface-cli download BAAI/bge-visualized-m3 --local-dir ./models/bge-visualized-m3")
    
    # 完成
    print("\n" + "=" * 50)
    print("安装完成！")
    print("=" * 50)
    print("\n下一步：")
    print("1. 如果还没下载模型，请使用以下命令：")
    print("   huggingface-cli download BAAI/bge-visualized-m3 --local-dir ./models/bge-visualized-m3")
    print("\n2. 确保 Milvus 服务正在运行：")
    print("   cd ../")
    print("   docker-compose up -d")
    print("\n3. 运行示例程序：")
    print("   python Milvus+Visual-BGE多模态检索-中文.py")
    print("\n详细文档请查看: 安装说明.md")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n安装已取消")
        sys.exit(1)
    except Exception as e:
        print(f"\n错误: {e}")
        sys.exit(1)
