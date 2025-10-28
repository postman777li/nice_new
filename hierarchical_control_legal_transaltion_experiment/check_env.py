#!/usr/bin/env python3
"""
环境配置检查工具
运行实验前使用此脚本验证环境配置是否正确
"""
import os
import sys
from pathlib import Path


def check_env():
    """检查环境配置"""
    print("=" * 60)
    print("环境配置检查")
    print("=" * 60)
    print()
    
    all_ok = True
    
    # 1. 检查 OPENAI_API_KEY
    api_key = os.getenv('OPENAI_API_KEY', '').strip()
    if api_key:
        print(f"✓ OPENAI_API_KEY: {api_key[:8]}...{api_key[-4:]}")
    else:
        print("✗ OPENAI_API_KEY: 未设置")
        print("  请设置: export OPENAI_API_KEY='your-api-key-here'")
        all_ok = False
    print()
    
    # 2. 检查可选环境变量
    print("可选环境变量:")
    
    base_url = os.getenv('OPENAI_BASE_URL')
    if base_url:
        print(f"  ✓ OPENAI_BASE_URL: {base_url}")
    else:
        print(f"  - OPENAI_BASE_URL: 未设置 (使用默认)")
    
    model = os.getenv('OPENAI_API_MODEL', 'gpt-4o-mini')
    print(f"  ✓ OPENAI_API_MODEL: {model}")
    
    timeout = os.getenv('LLM_TIMEOUT', '300')
    print(f"  ✓ LLM_TIMEOUT: {timeout}s")
    
    max_retries = os.getenv('LLM_MAX_RETRIES', '3')
    print(f"  ✓ LLM_MAX_RETRIES: {max_retries}")
    
    max_concurrent = os.getenv('LLM_MAX_CONCURRENT', '10')
    print(f"  ✓ LLM_MAX_CONCURRENT: {max_concurrent}")
    
    print()
    
    # 3. 检查必需文件
    print("必需文件:")
    
    required_files = [
        'terms_zh_en.db',
        'dataset/processed/test_set_zh_en.json',
        'src/lib/llm_client.py',
        'run_experiment.py',
        'evaluate_results.py'
    ]
    
    for file_path in required_files:
        full_path = Path(__file__).parent / file_path
        if full_path.exists():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} (不存在)")
            if file_path == 'terms_zh_en.db':
                print(f"    提示: 请运行术语库导入脚本")
            elif 'test_set' in file_path:
                print(f"    提示: 请确保测试集已生成")
    
    print()
    
    # 4. 检查 Python 依赖
    print("Python 依赖:")
    
    required_packages = [
        ('openai', 'OpenAI SDK'),
        ('asyncio', '异步支持（内置）'),
        ('bert_score', 'BERTScore评估'),
        ('sacrebleu', 'BLEU评估'),
        ('comet', 'COMET评估（unbabel-comet）')
    ]
    
    for package, desc in required_packages:
        try:
            if package == 'asyncio':
                import asyncio
            elif package == 'openai':
                import openai
            elif package == 'bert_score':
                import bert_score
            elif package == 'sacrebleu':
                import sacrebleu
            elif package == 'comet':
                from comet import download_model
            print(f"  ✓ {package} - {desc}")
        except ImportError:
            print(f"  ✗ {package} - {desc} (未安装)")
            if package == 'comet':
                print(f"    安装: pip install unbabel-comet")
            else:
                print(f"    安装: pip install {package}")
            all_ok = False
    
    print()
    print("=" * 60)
    
    if all_ok:
        print("✅ 环境配置检查通过！可以运行实验。")
        print()
        print("运行示例:")
        print("  # 单条翻译测试")
        print("  python run_translation.py --source '测试文本'")
        print()
        print("  # 批量实验（小样本测试）")
        print("  python run_experiment.py --samples 5 --ablations baseline")
        print()
        print("  # 完整实验")
        print("  python run_experiment.py --ablations baseline terminology full")
        return 0
    else:
        print("❌ 环境配置不完整，请先修复上述问题。")
        return 1


if __name__ == "__main__":
    sys.exit(check_env())

