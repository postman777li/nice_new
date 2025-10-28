"""
单语术语提取Agent - 从源文本中提取关键法律术语
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

from ..base import BaseAgent, AgentConfig, AgentRunContext

logger = logging.getLogger(__name__)


@dataclass
class MonoExtractItem:
    term: str
    score: float
    context: str
    category: str


class MonoExtractAgent(BaseAgent):
    def __init__(self, locale: str = 'zh'):
        super().__init__(AgentConfig(
            name='terminology:mono-extract',
            role='terminology_extractor',
            domain='terminology',
            specialty='单语术语提取',
            quality='review',
            locale=locale
        ))

    async def execute(self, input_data: Dict[str, Any], ctx: Optional[AgentRunContext] = None) -> List[MonoExtractItem]:
        """从源文本中提取关键法律术语"""
        text = input_data.get('text', '')
        if not text:
            return []
        
        # 使用LLM提取关键法律术语
        messages = [
            {
                "role": "system",
                "content": f"""你是一个专业的法律术语提取专家。请从给定的法律文本中提取关键的法律术语。

提取标准：
1. 法律概念和术语
2. 专业法律词汇
3. 需要标准化翻译的术语
4. 具有法律意义的复合词

请为每个术语提供：
- 术语文本
- 重要性评分（0-1）
- 术语类别

返回JSON格式：
{{
    "terms": [
        {{
            "term": "术语文本",
            "score": 0.9,
            "category": "术语类别"
        }}
    ]
}}"""
            },
            {
                "role": "user", 
                "content": f"请从以下法律文本中提取关键术语：\n\n{text}"
            }
        ]
        
        try:
            result = await self.call_llm_json(messages)
            terms_data = result.get('terms', [])
            
            return [
                MonoExtractItem(
                    term=item.get('term', ''),
                    score=item.get('score', 0.0),
                    context=item.get('context', ''),
                    category=item.get('category', '')
                )
                for item in terms_data
            ]
        except Exception as e:
            logger.error(f"Mono extract failed: {e}")
            return []
