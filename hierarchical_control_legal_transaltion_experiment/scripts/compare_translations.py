#!/usr/bin/env python3
"""
对比两个实验结果的翻译差异
"""
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

def load_results(filepath: str) -> List[Dict[str, Any]]:
    """加载实验结果"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 获取第一个键的值（可能是 terminology_syntax 或 full）
    key = list(data.keys())[0]
    return data[key]

def calculate_bleu_diff(ts_results: List[Dict], full_results: List[Dict]) -> List[tuple]:
    """计算 BLEU 分数差异"""
    diffs = []
    
    for i, (ts, full) in enumerate(zip(ts_results, full_results)):
        # 从 evaluation 中获取 BLEU
        ts_eval = ts.get('evaluation', {})
        full_eval = full.get('evaluation', {})
        
        # 尝试不同的 BLEU 位置
        ts_bleu = ts_eval.get('bleu', ts.get('metrics', {}).get('bleu', 0))
        full_bleu = full_eval.get('bleu', full.get('metrics', {}).get('bleu', 0))
        
        diff = ts_bleu - full_bleu
        
        diffs.append({
            'index': i,
            'sample_id': ts.get('sample_id', ''),
            'diff': diff,
            'ts_bleu': ts_bleu,
            'full_bleu': full_bleu,
            'source': ts.get('source', ''),
            'target': ts.get('target', ''),
            'ts_pred': ts.get('prediction', ''),
            'full_pred': full.get('prediction', ''),
            'domain': ts.get('metadata', {}).get('domain', '未知'),
            'law': ts.get('metadata', {}).get('law', '未知')
        })
    
    return diffs

def print_case_analysis(case: Dict, rank: int):
    """打印单个案例分析"""
    print(f"\n{'='*80}")
    print(f"案例 {rank}: 样本 ID={case['sample_id']}")
    print(f"{'='*80}")
    print(f"领域: {case['law']} ({case['domain']})")
    print(f"BLEU 差异: {case['diff']:.2f} (TS={case['ts_bleu']:.2f} vs Full={case['full_bleu']:.2f})")
    print(f"\n【源文】")
    print(case['source'][:200])
    if len(case['source']) > 200:
        print(f"... (共 {len(case['source'])} 字符)")
    
    print(f"\n【参考译文】")
    print(case['target'][:200])
    if len(case['target']) > 200:
        print(f"... (共 {len(case['target'])} 字符)")
    
    print(f"\n【Terminology+Syntax 译文】(BLEU={case['ts_bleu']:.2f})")
    print(case['ts_pred'][:200])
    if len(case['ts_pred']) > 200:
        print(f"... (共 {len(case['ts_pred'])} 字符)")
    
    print(f"\n【Full 译文】(BLEU={case['full_bleu']:.2f})")
    print(case['full_pred'][:200])
    if len(case['full_pred']) > 200:
        print(f"... (共 {len(case['full_pred'])} 字符)")
    
    # 分析差异
    print(f"\n【差异分析】")
    ts_words = set(case['ts_pred'].split())
    full_words = set(case['full_pred'].split())
    
    only_in_ts = ts_words - full_words
    only_in_full = full_words - ts_words
    
    if only_in_ts:
        print(f"仅在 TS 中: {', '.join(list(only_in_ts)[:10])}")
        if len(only_in_ts) > 10:
            print(f"  ... 还有 {len(only_in_ts) - 10} 个词")
    
    if only_in_full:
        print(f"仅在 Full 中: {', '.join(list(only_in_full)[:10])}")
        if len(only_in_full) > 10:
            print(f"  ... 还有 {len(only_in_full) - 10} 个词")

def analyze_by_domain(diffs: List[Dict]):
    """按领域分析"""
    by_domain = defaultdict(list)
    for case in diffs:
        by_domain[case['law']].append(case['diff'])
    
    print(f"\n{'='*80}")
    print("各法律领域的 BLEU 差异统计")
    print(f"{'='*80}")
    
    for law, diff_list in sorted(by_domain.items()):
        avg_diff = sum(diff_list) / len(diff_list)
        max_diff = max(diff_list)
        min_diff = min(diff_list)
        worse_count = sum(1 for d in diff_list if d < 0)
        
        print(f"\n{law}:")
        print(f"  样本数: {len(diff_list)}")
        print(f"  平均差异: {avg_diff:.2f}")
        print(f"  最大差异: {max_diff:.2f}")
        print(f"  最小差异: {min_diff:.2f}")
        print(f"  Full 更差的比例: {worse_count}/{len(diff_list)} ({worse_count/len(diff_list)*100:.1f}%)")

def main():
    # 文件路径
    ts_file = "outputs/translation-terminology_syntax/experiment_results_1760162012.json"
    full_file = "outputs/translation-full/experiment_results_1760172546.json"
    
    print("加载实验结果...")
    ts_results = load_results(ts_file)
    full_results = load_results(full_file)
    
    print(f"TS 结果: {len(ts_results)} 个样本")
    print(f"Full 结果: {len(full_results)} 个样本")
    
    # 计算差异
    print("\n计算 BLEU 差异...")
    diffs = calculate_bleu_diff(ts_results, full_results)
    
    # 按差异排序
    diffs.sort(key=lambda x: x['diff'], reverse=True)
    
    # 统计
    print(f"\n{'='*80}")
    print("整体统计")
    print(f"{'='*80}")
    avg_diff = sum(d['diff'] for d in diffs) / len(diffs)
    worse_count = sum(1 for d in diffs if d['diff'] < 0)
    better_count = sum(1 for d in diffs if d['diff'] > 0)
    
    print(f"平均 BLEU 差异: {avg_diff:.2f}")
    print(f"Full 更差的案例: {worse_count}/{len(diffs)} ({worse_count/len(diffs)*100:.1f}%)")
    print(f"Full 更好的案例: {better_count}/{len(diffs)} ({better_count/len(diffs)*100:.1f}%)")
    
    # 按领域分析
    analyze_by_domain(diffs)
    
    # 显示差异最大的案例（TS 更好）
    print(f"\n{'='*80}")
    print("Full 模式表现最差的 5 个案例 (TS 明显更好)")
    print(f"{'='*80}")
    for i, case in enumerate(diffs[:5], 1):
        print_case_analysis(case, i)
    
    # 显示差异最大的案例（Full 更好）
    print(f"\n\n{'='*80}")
    print("Full 模式表现最好的 5 个案例 (Full 明显更好)")
    print(f"{'='*80}")
    for i, case in enumerate(diffs[-5:][::-1], 1):
        print_case_analysis(case, i)
    
    # 保存详细对比
    output_file = "outputs/detailed_comparison.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(diffs, f, ensure_ascii=False, indent=2)
    print(f"\n\n详细对比已保存到: {output_file}")

if __name__ == "__main__":
    main()

