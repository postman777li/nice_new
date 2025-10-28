#!/bin/bash

# =============================================
# Hugging Face 镜像加速配置脚本
# =============================================

echo "================================"
echo "配置 Hugging Face 镜像加速"
echo "================================"

# 设置HF镜像环境变量
export HF_ENDPOINT=https://hf-mirror.com

echo ""
echo "✓ 已设置 HF_ENDPOINT=https://hf-mirror.com"
echo ""

# 验证设置
if [ -z "$HF_ENDPOINT" ]; then
    echo "❌ 设置失败"
    exit 1
else
    echo "✓ 当前 HF_ENDPOINT: $HF_ENDPOINT"
fi

echo ""
echo "================================"
echo "使用方法："
echo "================================"
echo ""
echo "方法1: 在当前shell中生效（临时）"
echo "  source setup_hf_mirror.sh"
echo ""
echo "方法2: 永久生效（添加到 ~/.bashrc 或 ~/.zshrc）"
echo "  echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc"
echo "  source ~/.bashrc"
echo ""
echo "方法3: 在 Python 代码中自动启用（推荐）"
echo "  # 已在 BERTScore 和 COMET 类中自动配置"
echo "  # use_hf_mirror=True （默认）"
echo ""
echo "================================"
echo "测试下载速度："
echo "================================"
echo ""
echo "运行以下命令测试："
echo "  python -c 'from transformers import AutoTokenizer; AutoTokenizer.from_pretrained(\"bert-base-chinese\")'"
echo ""

