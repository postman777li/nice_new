#!/usr/bin/env python3
"""
预下载评估所需的所有模型
避免在评估时下载，节省时间
"""
import os
import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def setup_hf_mirror(use_mirror=True):
    """设置 HF 镜像"""
    if use_mirror:
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        print("✓ 已启用 Hugging Face 镜像加速: https://hf-mirror.com")
    else:
        if 'HF_ENDPOINT' in os.environ:
            del os.environ['HF_ENDPOINT']
        print("✓ 使用官方 Hugging Face Hub")


def clean_model_cache(model_name):
    """清理指定模型的缓存"""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    
    # 转换模型名称为缓存目录格式
    # 例如: xlm-roberta-large -> models--xlm-roberta-large
    # Unbabel/wmt22-comet-da -> models--Unbabel--wmt22-comet-da
    cache_name = "models--" + model_name.replace("/", "--")
    cache_path = cache_dir / cache_name
    
    if cache_path.exists():
        try:
            print(f"   清理旧缓存: {cache_path}")
            shutil.rmtree(cache_path)
            print(f"   ✅ 缓存已清理")
            return True
        except Exception as e:
            print(f"   ⚠️  清理失败: {e}")
            return False
    else:
        print(f"   ℹ️  未找到缓存: {cache_name}")
        return True


def download_bertscore_model(clean_cache=False, use_mirror=True):
    """下载 BERTScore 模型"""
    print("="*60)
    print("1. 下载 BERTScore 模型")
    print("="*60)
    
    model_type = "xlm-roberta-large"
    
    if clean_cache:
        print("清理旧缓存...")
        clean_model_cache(model_type)
    
    try:
        setup_hf_mirror(use_mirror)
        
        from transformers import AutoModel, AutoTokenizer
        import torch
        
        print(f"模型: {model_type}")
        print(f"大小: ~1.4 GB")
        print("开始下载...\n")
        
        # 下载tokenizer和model
        print("   下载 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_type)
        print("   ✅ Tokenizer 下载完成")
        
        print("   下载 model...")
        model = AutoModel.from_pretrained(model_type)
        print("   ✅ Model 下载完成")
        
        # 使用 BERTScorer 确保完整下载
        print("\n   初始化 BERTScorer...")
        from bert_score import BERTScorer
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        scorer = BERTScorer(
            model_type=model_type,
            lang="zh",
            device=device,
            rescale_with_baseline=False
        )
        
        print(f"✅ BERTScore 模型下载完成")
        print(f"   模型: {scorer._model.config._name_or_path}")
        print(f"   设备: {device}")
        
        # 测试一下
        test_pred = ["这是一个测试"]
        test_ref = ["这是测试"]
        P, R, F1 = scorer.score(test_pred, test_ref)
        print(f"   测试分数: F1={F1.mean().item():.4f}")
        
        return True
    except Exception as e:
        print(f"❌ BERTScore 模型下载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def download_comet_model(clean_cache=False, use_mirror=True):
    """下载 COMET 模型"""
    print("\n" + "="*60)
    print("2. 下载 COMET 模型")
    print("="*60)
    
    model_name = "Unbabel/wmt22-comet-da"
    
    if clean_cache:
        print("清理旧缓存...")
        clean_model_cache(model_name)
    
    try:
        setup_hf_mirror(use_mirror)
        
        from comet import download_model, load_from_checkpoint
        
        print(f"模型: {model_name}")
        print(f"大小: ~2.3 GB")
        print("开始下载...\n")
        
        model_path = download_model(model_name)
        print(f"✅ COMET 模型下载完成")
        print(f"   模型路径: {model_path}")
        
        # 加载模型测试
        print("   加载模型进行测试...")
        model = load_from_checkpoint(model_path)
        
        # 测试一下
        data = [{
            "src": "合同双方应当遵守本协议的所有条款。",
            "mt": "The parties shall comply with all terms.",
            "ref": "The parties must comply with all terms."
        }]
        result = model.predict(data, batch_size=1, gpus=0)
        print(f"   测试分数: {result['scores'][0]:.4f}")
        
        return True
    except Exception as e:
        print(f"❌ COMET 模型下载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_disk_space():
    """检查磁盘空间"""
    print("="*60)
    print("磁盘空间检查")
    print("="*60)
    
    import shutil
    
    cache_dir = Path.home() / ".cache" / "huggingface"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    total, used, free = shutil.disk_usage(cache_dir)
    
    print(f"缓存目录: {cache_dir}")
    print(f"总空间: {total / (1024**3):.1f} GB")
    print(f"已使用: {used / (1024**3):.1f} GB")
    print(f"可用空间: {free / (1024**3):.1f} GB")
    
    required_gb = 4  # BERTScore (1.4GB) + COMET (2.3GB) + 余量
    if free / (1024**3) < required_gb:
        print(f"\n⚠️  警告: 磁盘空间不足 {required_gb} GB")
        print(f"   建议清理缓存或增加磁盘空间")
        return False
    else:
        print(f"\n✅ 磁盘空间充足 (需要 ~{required_gb} GB)")
        return True


def show_cache_info():
    """显示缓存信息"""
    print("\n" + "="*60)
    print("模型缓存信息")
    print("="*60)
    
    cache_dir = Path.home() / ".cache" / "huggingface"
    
    if cache_dir.exists():
        # 统计缓存大小
        total_size = 0
        file_count = 0
        for path in cache_dir.rglob("*"):
            if path.is_file():
                total_size += path.stat().st_size
                file_count += 1
        
        print(f"缓存目录: {cache_dir}")
        print(f"文件数量: {file_count}")
        print(f"总大小: {total_size / (1024**3):.2f} GB")
        
        # 列出主要模型
        hub_dir = cache_dir / "hub"
        if hub_dir.exists():
            print(f"\n已缓存的模型 (前20个):")
            model_dirs = [d for d in hub_dir.iterdir() if d.is_dir() and d.name.startswith('models--')]
            for model_dir in sorted(model_dirs)[:20]:
                model_name = model_dir.name.replace('models--', '').replace('--', '/')
                size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
                print(f"   - {model_name}: {size / (1024**3):.2f} GB")
    else:
        print(f"缓存目录不存在: {cache_dir}")
        print("将在下载模型时自动创建")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='预下载评估所需的模型')
    parser.add_argument('--clean', action='store_true', help='清理旧缓存后重新下载')
    parser.add_argument('--no-mirror', action='store_true', help='不使用 HF 镜像')
    parser.add_argument('--bert-only', action='store_true', help='仅下载 BERTScore 模型')
    parser.add_argument('--comet-only', action='store_true', help='仅下载 COMET 模型')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print(" "*15 + "模型预下载工具")
    print("="*60 + "\n")
    
    use_mirror = not args.no_mirror
    
    if args.clean:
        print("⚠️  清理模式: 将删除旧缓存后重新下载\n")
    
    if not (args.bert_only or args.comet_only):
        print("此脚本将下载以下模型:")
        print("  1. xlm-roberta-large (BERTScore) - ~1.4 GB")
        print("  2. wmt22-comet-da (COMET) - ~2.3 GB")
        print("  总计: ~3.7 GB\n")
    
    # 检查磁盘空间
    if not check_disk_space():
        print("\n❌ 磁盘空间不足，请清理后重试")
        return
    
    print("\n" + "="*60)
    print("开始下载模型")
    print("="*60 + "\n")
    
    # 下载模型
    results = []
    
    if not args.comet_only:
        results.append(("BERTScore", download_bertscore_model(
            clean_cache=args.clean, 
            use_mirror=use_mirror
        )))
    
    if not args.bert_only:
        results.append(("COMET", download_comet_model(
            clean_cache=args.clean,
            use_mirror=use_mirror
        )))
    
    # 显示缓存信息
    show_cache_info()
    
    # 总结
    print("\n" + "="*60)
    print("下载总结")
    print("="*60)
    
    for name, success in results:
        status = "✅ 成功" if success else "❌ 失败"
        print(f"{name:15s}: {status}")
    
    all_success = all(success for _, success in results)
    
    if all_success:
        print("\n🎉 所有模型下载完成！")
        print("\n现在可以运行评估脚本，无需等待下载：")
        print("  python evaluate_results.py outputs/experiment_results.json --metrics bleu chrf bertscore comet")
    else:
        print("\n⚠️  部分模型下载失败")
        print("\n建议:")
        print("  1. 检查网络连接")
        print("  2. 如果镜像有问题，尝试不使用镜像:")
        print("     python download_models.py --no-mirror --clean")
        print("  3. 分别下载各个模型:")
        print("     python download_models.py --bert-only --clean")
        print("     python download_models.py --comet-only --clean")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  下载已取消")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
