#!/usr/bin/env python3
"""
测试 GPU 加速是否正常工作
"""
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.metrics import COMETMetric, BERTScoreMetric

print("="*60)
print("GPU 加速测试")
print("="*60)

# 检查 CUDA 可用性
print(f"\n1. CUDA 可用性检查:")
print(f"   torch.cuda.is_available(): {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU 设备: {torch.cuda.get_device_name(0)}")
    print(f"   GPU 数量: {torch.cuda.device_count()}")
else:
    print("   ⚠️  未检测到 GPU，将使用 CPU")
    sys.exit(0)

# 测试数据
source = "合同双方应当遵守本协议的所有条款。"
prediction = "The parties shall comply with all terms of this agreement."
reference = "Contracting parties must comply with all provisions of this agreement."

print(f"\n2. 测试 BERTScore GPU 加速:")
try:
    bertscore = BERTScoreMetric(device='cuda', use_hf_mirror=True)
    print(f"   ✓ BERTScore 初始化成功，设备: {bertscore.device}")
    
    # 第一次调用（会加载模型）
    print(f"   正在加载模型并计算...")
    scores = bertscore.compute([prediction], [reference])
    print(f"   ✓ BERTScore F1: {scores['f1']:.4f}")
    
    # 第二次调用（应该使用缓存的模型）
    print(f"   再次计算（使用缓存的模型）...")
    scores = bertscore.compute([prediction], [reference])
    print(f"   ✓ BERTScore F1: {scores['f1']:.4f} (模型已缓存)")
    
    # 检查模型是否在 GPU 上
    if bertscore._scorer is not None:
        model_device = next(bertscore._scorer._model.parameters()).device
        print(f"   ✓ 模型位于设备: {model_device}")
    
except Exception as e:
    print(f"   ✗ BERTScore 测试失败: {e}")

print(f"\n3. 测试 COMET GPU 加速:")
try:
    comet = COMETMetric(gpus=1, use_hf_mirror=True)
    print(f"   ✓ COMET 初始化成功，GPU 数: {comet.gpus}")
    
    # 第一次调用（会加载模型）
    print(f"   正在加载模型并计算...")
    result = comet.compute([source], [prediction], [reference])
    print(f"   ✓ COMET 分数: {result['mean']:.4f}")
    
    # 第二次调用（应该使用缓存的模型）
    print(f"   再次计算（使用缓存的模型）...")
    result = comet.compute([source], [prediction], [reference])
    print(f"   ✓ COMET 分数: {result['mean']:.4f} (模型已缓存)")
    
except Exception as e:
    print(f"   ✗ COMET 测试失败: {e}")

print(f"\n4. GPU 内存使用:")
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    print(f"   已分配: {allocated:.2f} GB")
    print(f"   已保留: {reserved:.2f} GB")

print(f"\n{'='*60}")
print("✅ GPU 加速测试完成")
print("="*60)

