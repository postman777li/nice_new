#!/usr/bin/env python3
"""
从已有的full实验结果的trace中提取中间层翻译
"""
import json
import sys
from pathlib import Path

input_file = sys.argv[1] if len(sys.argv) > 1 else 'outputs/experiment_results_1760192230.json'
input_path = Path(input_file)

print(f"读取: {input_path}")
with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

if 'full' not in data:
    print("❌ 没有找到full实验结果")
    sys.exit(1)

full_results = data['full']
print(f"找到 {len(full_results)} 个full样本")

# 提取术语层
terminology = []
for sample in full_results:
    trace = sample.get('trace', {})
    if 'r1' in trace and trace['r1'].get('output'):
        terminology.append({
            'sample_id': sample['sample_id'],
            'source': sample['source'],
            'target': sample['target'],
            'prediction': trace['r1']['output'],
            'success': True,
            'metrics': {},
            'metadata': sample.get('metadata', {})
        })

print(f"✓ 提取了 {len(terminology)} 个术语层结果")

# 提取术语+句法层
syntax = []
for sample in full_results:
    trace = sample.get('trace', {})
    if 'r2' in trace and trace['r2'].get('output'):
        syntax.append({
            'sample_id': sample['sample_id'],
            'source': sample['source'],
            'target': sample['target'],
            'prediction': trace['r2']['output'],
            'success': True,
            'metrics': {},
            'metadata': sample.get('metadata', {})
        })

print(f"✓ 提取了 {len(syntax)} 个术语+句法层结果")

# 添加到结果中
if terminology:
    data['terminology'] = terminology
if syntax:
    data['terminology_syntax'] = syntax

# 保存
output_file = input_path.parent / f"{input_path.stem}_with_layers.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"\n✅ 保存到: {output_file}")
print(f"现在包含: {', '.join(data.keys())}")

