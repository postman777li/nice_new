"""
简化的文档级术语提取Agent
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re
import logging

from ..base import BaseAgent, AgentConfig, AgentRunContext

logger = logging.getLogger(__name__)


@dataclass
class DocumentTerm:
    """文档术语"""
    term: str
    score: float
    count: int


@dataclass
class DocumentTermExtractOptions:
    """术语提取选项"""
    max_terms: int = 100
    chunk_size: int = 5000
    overlap: int = 300
    prompt: Optional[str] = None


class DocumentTermExtractAgent(BaseAgent):
    """简化的文档级术语提取Agent"""
    
    def __init__(self, locale: str = 'zh'):
        super().__init__(AgentConfig(
            name='preprocess:doc-term-extract',
            role='terminology_extractor',
            domain='terminology',
            specialty='文档术语识别',
            quality='review',
            locale=locale
        ))
    
    async def execute(self, input_data: Dict[str, Any], ctx: Optional[AgentRunContext] = None) -> List[DocumentTerm]:
        """执行术语提取"""
        text = input_data.get('text', '').strip()
        if not text:
            return []
        
        options = input_data.get('options', DocumentTermExtractOptions())
        max_terms = max(1, min(200, options.max_terms))
        
        # 使用LLM提取术语
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
- 出现次数

返回JSON格式：
{{
    "terms": [
        {{
            "term": "术语文本",
            "score": 0.9,
            "count": 3
        }}
    ]
}}"""
            },
            {
                "role": "user", 
                "content": f"请从以下法律文本中提取最多{max_terms}个关键术语：\n\n{text}"
            }
        ]
        
        try:
            result = await self.call_llm_json(messages)
            terms_data = result.get('terms', [])
            
            return [
                DocumentTerm(
                    term=item.get('term', ''),
                    score=item.get('score', 0.0),
                    count=item.get('count', 1)
                )
                for item in terms_data
            ]
        except Exception as e:
            logger.error(f"Term extraction failed: {e}")
            # 回退到简单的统计方法
            return self._fallback_extraction(text, max_terms)
    
    def _fallback_extraction(self, text: str, max_terms: int) -> List[DocumentTerm]:
        """回退的统计提取方法"""
        # 简单的词频统计
        words = re.findall(r'\b\w+\b', text)
        word_count = {}
        
        for word in words:
            if len(word) > 2:  # 过滤短词
                word_count[word] = word_count.get(word, 0) + 1
        
        # 按频率排序
        sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        
        terms = []
        for word, count in sorted_words[:max_terms]:
            score = min(1.0, count / len(words))  # 简单的评分
            terms.append(DocumentTerm(term=word, score=score, count=count))
        
        return terms
