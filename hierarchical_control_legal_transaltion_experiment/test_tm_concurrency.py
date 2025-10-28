#!/usr/bin/env python3
"""
测试 TM 在并发场景下的初始化
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.agents.discourse.discourse_query import DiscourseQueryAgent

async def test_concurrent_tm():
    """测试并发 TM 初始化"""
    print("测试并发 TM 初始化...")
    print("="*60)
    
    # 创建 10 个 agent 实例（模拟并发场景）
    agents = [DiscourseQueryAgent(locale='zh') for _ in range(10)]
    
    # 并发执行查询
    async def query(agent, idx):
        result = await agent.execute({
            'text': '公司应当制作股东名册',
            'source_lang': 'zh',
            'target_lang': 'en',
            'top_k': 3
        })
        return idx, len(result)
    
    tasks = [query(agent, i) for i, agent in enumerate(agents)]
    results = await asyncio.gather(*tasks)
    
    print("\n并发查询结果:")
    for idx, count in results:
        status = "✅" if count > 0 else "⚠️"
        print(f"{status} Agent {idx}: 找到 {count} 个 TM 匹配")
    
    success_count = sum(1 for _, count in results if count > 0)
    print(f"\n总结: {success_count}/{len(results)} 个 agent 成功检索到 TM")
    
    if success_count == len(results):
        print("✅ 并发 TM 初始化测试通过！")
        return True
    else:
        print("❌ 部分 agent 未能检索到 TM")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_concurrent_tm())
    sys.exit(0 if result else 1)

