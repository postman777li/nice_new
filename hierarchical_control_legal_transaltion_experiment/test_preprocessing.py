#!/usr/bin/env python3
"""
术语预处理快速测试脚本
"""
import asyncio
from pathlib import Path
from datasets import TestSample
from src.agents.terminology import (
    MonoExtractAgent,
    DeduplicateAgent,
    BatchTranslateAgent,
    TerminologyPreprocessor
)


async def test_basic_workflow():
    """测试基本工作流"""
    print("="*60)
    print("术语预处理基本功能测试")
    print("="*60)
    
    # 创建测试样本
    samples = [
        TestSample(
            id="1",
            source="人民法院应当依法保护当事人的合法权益。诉讼代理人可以代为承认、放弃、变更诉讼请求。",
            target="The people's court shall protect the legitimate rights and interests of the parties according to law.",
            src_lang="zh",
            tgt_lang="en",
            document_id="test",
            article_id="1",
            metadata={"law": "civil_procedure", "domain": "litigation"}
        ),
        TestSample(
            id="2",
            source="最高人民法院管辖全国重大案件。人民法院可以裁定先予执行。",
            target="The Supreme People's Court has jurisdiction over major cases nationwide.",
            src_lang="zh",
            tgt_lang="en",
            document_id="test",
            article_id="2",
            metadata={"law": "civil_procedure", "domain": "litigation"}
        )
    ]
    
    print(f"\n创建了 {len(samples)} 个测试样本\n")
    
    # 测试1: 单语提取
    print("-"*60)
    print("测试1: 单语术语提取")
    print("-"*60)
    mono_extract = MonoExtractAgent(locale='zh')
    
    extracted_terms = []
    contexts = []
    for sample in samples:
        terms = await mono_extract.execute({'text': sample.source})
        extracted_terms.append(terms)
        contexts.append(sample.source)
        print(f"样本{sample.id}: 提取了 {len(terms)} 个术语")
        for term in terms[:3]:  # 只显示前3个
            print(f"  - {term.term} (分数: {term.score:.2f}, 类别: {term.category})")
    
    # 测试2: 去重
    print(f"\n{'-'*60}")
    print("测试2: 术语去重")
    print("-"*60)
    dedup = DeduplicateAgent(locale='zh')
    deduplicated = await dedup.execute({
        'extracted_terms': extracted_terms,
        'contexts': contexts,
        'max_contexts': 2
    })
    
    total_before = sum(len(terms) for terms in extracted_terms)
    print(f"去重前: {total_before} 个术语")
    print(f"去重后: {len(deduplicated)} 个不重复术语")
    print("\n高频术语:")
    for term in deduplicated[:5]:
        print(f"  - {term.term} (出现{term.count}次, 分数{term.score:.2f})")
    
    print(f"\n{'='*60}")
    print("✅ 基本功能测试通过！")
    print("="*60)
    print("\n提示：")
    print("1. 完整预处理请使用: python run_experiment.py --preprocess-only")
    print("2. 查看详细文档: TERM_PREPROCESSING_GUIDE.md")


if __name__ == "__main__":
    asyncio.run(test_basic_workflow())

