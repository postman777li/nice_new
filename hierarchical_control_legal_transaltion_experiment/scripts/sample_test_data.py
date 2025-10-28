"""
从原始数据中随机抽取100个双语句子对作为测试数据
Sample 100 bilingual sentence pairs from raw data as test data
"""
import json
import random
from pathlib import Path

def sample_bilingual_pairs(input_json_files, output_file, sample_size=100, seed=42, lang_pair=('zh', 'en')):
    """
    从多个JSON文件中随机抽取双语句子对
    
    Args:
        input_json_files: 输入JSON文件列表
        output_file: 输出文件路径
        sample_size: 抽样数量
        seed: 随机种子
        lang_pair: 语言对，例如 ('zh', 'en') 或 ('zh', 'ja')
    """
    random.seed(seed)
    src_lang, tgt_lang = lang_pair
    
    all_pairs = []
    
    # 读取所有JSON文件
    for json_file in input_json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 提取句子对（支持新格式：texts.zh, texts.en, texts.ja）
            if 'entries' in data:
                metadata = data.get('metadata', {})
                law_name = metadata.get('law_name', '')
                domain = metadata.get('domain', '')
                year = metadata.get('year', '')
                
                for entry in data['entries']:
                    texts = entry.get('texts', {})
                    source_text = texts.get(src_lang, entry.get(src_lang, ''))
                    target_text = texts.get(tgt_lang, entry.get(tgt_lang, ''))
                    
                    # 使用train数据相同的格式：直接使用语言代码作为字段名
                    pair = {
                        'law': law_name,
                        'domain': domain,
                        'year': year,
                        'id': entry.get('id', ''),
                        src_lang: source_text,
                        tgt_lang: target_text
                    }
                    if pair[src_lang] and pair[tgt_lang]:
                        all_pairs.append(pair)
            
            print(f"✓ 读取 {json_file.name}: {len(data.get('entries', []))} 条")
            
        except Exception as e:
            print(f"✗ 读取 {json_file.name} 失败: {e}")
    
    print(f"\n总共收集到 {len(all_pairs)} 个句子对")
    
    # 随机抽样
    if len(all_pairs) > sample_size:
        sampled_pairs = random.sample(all_pairs, sample_size)
    else:
        sampled_pairs = all_pairs
        print(f"⚠️ 可用数据少于 {sample_size}，使用全部 {len(all_pairs)} 个")
    
    # 统计领域信息
    domains = list(set(pair['law'] for pair in sampled_pairs if pair.get('law')))
    
    # 保存（格式与train_set一致）
    output_data = {
        'metadata': {
            'pair': f"{src_lang}-{tgt_lang}",
            'total_entries': len(sampled_pairs),
            'domains': sorted(domains),
            'seed': seed,
            'sample_from': len(all_pairs),
            'source_files': [f.name for f in input_json_files],
            'created_at': __import__('datetime').datetime.now().isoformat()
        },
        'entries': sampled_pairs
    }
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 保存 {len(sampled_pairs)} 个样本到: {output_file}")
    
    return len(sampled_pairs)


def main():
    # 数据集目录
    dataset_dir = Path(__file__).parent.parent / 'dataset' / 'processed'
    output_base = Path('/home/hao/deepling.tech/bilingual_term_extractor/examples/test_data')
    
    print("=" * 60)
    print("创建测试数据样本")
    print("=" * 60)
    
    # 1. 中英数据
    print("\n[1/2] 抽取中英双语数据...")
    zh_en_files = list(dataset_dir.glob('中华人民共和国*.json'))[:5]  # 选择前5个文件
    if zh_en_files:
        output_zh_en = output_base / 'sample_zh_en_100.json'
        count = sample_bilingual_pairs(zh_en_files, output_zh_en, sample_size=100, lang_pair=('zh', 'en'))
        print(f"✓ 中英样本创建完成: {count} 对")
    else:
        print("✗ 未找到中英数据文件")
    
    # 2. 中日数据
    print("\n[2/2] 抽取中日双语数据...")
    # 使用相同的文件，但提取中日对
    if zh_en_files:
        output_zh_ja = output_base / 'sample_zh_ja_100.json'
        count = sample_bilingual_pairs(zh_en_files, output_zh_ja, sample_size=100, lang_pair=('zh', 'ja'))
        print(f"✓ 中日样本创建完成: {count} 对")
    else:
        print("✗ 未找到数据文件")
    
    print("\n" + "=" * 60)
    print("测试数据创建完成！")
    print("=" * 60)
    print(f"\n输出位置: {output_base}")
    print(f"  - sample_zh_en_100.json (中英对)")
    print(f"  - sample_zh_ja_100.json (中日对)")


if __name__ == '__main__':
    main()

