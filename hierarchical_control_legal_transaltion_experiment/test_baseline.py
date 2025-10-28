#!/usr/bin/env python3
"""
测试基线翻译智能体
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.agents.baseline_translation import BaselineTranslationAgent


async def test_baseline():
    """测试基线翻译"""
    print("="*60)
    print("测试基线翻译智能体（纯LLM，无控制策略）")
    print("="*60)
    
    # 创建基线翻译智能体
    agent = BaselineTranslationAgent(locale='en')
    
    # 测试文本
    test_text = "合同双方应当遵守本协议的所有条款。"
    
    print(f"\n源文本: {test_text}")
    print("翻译中...\n")
    
    # 执行翻译
    result = await agent.execute({
        'source_text': test_text,
        'source_lang': 'zh',
        'target_lang': 'en'
    }, None)
    
    print(f"译文: {result.translated_text}")
    print(f"置信度: {result.confidence:.2f}")
    
    print("\n" + "="*60)
    print("✅ 基线翻译测试完成")
    print("="*60)
    
    return result


if __name__ == "__main__":
    asyncio.run(test_baseline())

