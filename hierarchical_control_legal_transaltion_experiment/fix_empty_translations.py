#!/usr/bin/env python3
"""
修复实验结果中的空翻译
自动识别空翻译样本并重新翻译
"""
import asyncio
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).parent))

from run_translation import SimpleTranslator
from datasets import TestSample


async def fix_empty_translations(
    result_file: Path,
    output_file: Path = None,
    verbose: bool = False,
    dry_run: bool = False
):
    """修复空翻译"""
    print("=" * 60)
    print("空翻译修复工具")
    print("=" * 60)
    
    # 加载结果文件
    print(f"\n加载结果文件: {result_file}")
    with open(result_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    print(f"找到 {len(results)} 个消融实验")
    
    # 统计空翻译
    total_empty = 0
    fixes_needed = {}
    
    for ablation_name, samples in results.items():
        empty_samples = [
            (i, s) for i, s in enumerate(samples)
            if not s.get('prediction', '').strip()
        ]
        if empty_samples:
            fixes_needed[ablation_name] = empty_samples
            total_empty += len(empty_samples)
            print(f"  {ablation_name}: {len(samples)} 个样本, {len(empty_samples)} 个空翻译")
    
    if total_empty == 0:
        print("\n✅ 没有发现空翻译，无需修复！")
        return 0
    
    print(f"\n共发现 {total_empty} 个空翻译需要修复")
    
    if dry_run:
        print("\n[预演模式] 以下样本将被重新翻译:")
        for ablation_name, empty_samples in fixes_needed.items():
            print(f"\n{ablation_name}:")
            for idx, sample in empty_samples[:5]:  # 只显示前5个
                print(f"  [{idx}] sample_id={sample.get('sample_id')}")
                print(f"      source: {sample.get('source', '')[:80]}...")
            if len(empty_samples) > 5:
                print(f"  ... 还有 {len(empty_samples) - 5} 个")
        print("\n运行时去掉 --dry-run 参数以执行实际修复")
        return 0
    
    # 执行修复
    print("\n开始修复...")
    fixed_count = 0
    
    for ablation_name, empty_samples in fixes_needed.items():
        print(f"\n{'='*60}")
        print(f"修复消融实验: {ablation_name}")
        print(f"{'='*60}")
        
        # 从ablation名称推断配置
        config = _get_config_from_ablation(ablation_name)
        translator = SimpleTranslator(config, verbose=verbose)
        
        for idx, sample in empty_samples:
            sample_id = sample.get('sample_id', 'unknown')
            source = sample.get('source', '')
            
            if not source:
                print(f"  ⚠️  跳过样本 {sample_id}: 源文本为空")
                continue
            
            print(f"  [{idx}] 重新翻译 sample_id={sample_id}...")
            
            try:
                # 重新翻译
                result = await translator.translate(
                    source=source,
                    src_lang='zh',
                    tgt_lang='en'
                )
                
                if result['success'] and result['final'].strip():
                    # 更新结果
                    results[ablation_name][idx]['prediction'] = result['final']
                    results[ablation_name][idx]['success'] = True
                    results[ablation_name][idx]['trace'] = result['trace']
                    
                    # 如果有trace中的metrics，也更新
                    if 'metrics' in sample and result['trace'].get('r1'):
                        # 保留原有metrics，只更新confidence
                        results[ablation_name][idx]['trace'] = result['trace']
                    
                    fixed_count += 1
                    print(f"      ✓ 修复成功: {result['final'][:80]}...")
                else:
                    print(f"      ✗ 修复失败: 翻译结果仍为空")
                    results[ablation_name][idx]['error'] = 'Retry translation failed'
                
            except Exception as e:
                print(f"      ✗ 修复失败: {e}")
                results[ablation_name][idx]['error'] = f'Retry failed: {str(e)}'
    
    # 保存修复后的结果
    if output_file is None:
        output_file = result_file.parent / f"{result_file.stem}_fixed.json"
    
    print(f"\n{'='*60}")
    print(f"保存修复后的结果到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 修复完成!")
    print(f"  总空翻译数: {total_empty}")
    print(f"  成功修复数: {fixed_count}")
    print(f"  失败数: {total_empty - fixed_count}")
    
    return 0


def _get_config_from_ablation(ablation_name: str) -> Dict[str, Any]:
    """从消融实验名称推断配置"""
    if 'baseline' in ablation_name.lower():
        return {
            'name': '基线（纯LLM）',
            'hierarchical': False,
            'useTermBase': False,
            'useRuleTable': False,
            'useTM': False,
            'max_rounds': 1
        }
    elif 'terminology_syntax' in ablation_name.lower():
        return {
            'name': '术语+句法控制',
            'hierarchical': True,
            'useTermBase': True,
            'useRuleTable': False,
            'useTM': False,
            'max_rounds': 2
        }
    elif 'terminology' in ablation_name.lower():
        return {
            'name': '术语控制',
            'hierarchical': True,
            'useTermBase': True,
            'useRuleTable': False,
            'useTM': False,
            'max_rounds': 1
        }
    else:  # full
        return {
            'name': '完整系统',
            'hierarchical': True,
            'useTermBase': True,
            'useRuleTable': False,
            'useTM': False,
            'max_rounds': 3
        }


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='修复实验结果中的空翻译')
    parser.add_argument('result_file', type=str, help='实验结果文件路径（JSON）')
    parser.add_argument('--output', '-o', type=str, help='输出文件路径（默认: 原文件名_fixed.json）')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
    parser.add_argument('--dry-run', action='store_true', help='预演模式，不执行实际修复')
    
    args = parser.parse_args()
    
    result_file = Path(args.result_file)
    if not result_file.exists():
        print(f"❌ 错误：文件不存在: {result_file}")
        return 1
    
    output_file = Path(args.output) if args.output else None
    
    return await fix_empty_translations(
        result_file=result_file,
        output_file=output_file,
        verbose=args.verbose,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

