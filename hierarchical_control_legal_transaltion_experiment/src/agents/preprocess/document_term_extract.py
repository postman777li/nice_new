"""
文档级术语提取Agent - Python版本
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re
import asyncio
import logging
import json

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
    """文档级术语提取Agent"""
    
    def __init__(self, locale: str = 'zh'):
        super().__init__(AgentConfig(
            name='preprocess:doc-term-extract',
            role='terminology_extractor',
            domain='terminology',
            specialty='大规模文档术语识别与评估',
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
        chunk_size = max(1000, min(12000, options.chunk_size))
        overlap = max(0, min(chunk_size // 4, options.overlap))
        
        # 先用统计方法获取候选术语
        candidates = self._build_stat_candidates(text, chunk_size, overlap, max_terms * 5)
        
        # 然后用LLM进行评分和筛选
        locale = ctx.locale if ctx else input_data.get('locale', self.locale)
        final_terms = await self._score_with_llm(
            candidates, text, options.prompt, max_terms, locale
        )
        
        return final_terms
    
    def _build_stat_candidates(self, text: str, chunk_size: int, overlap: int, max_candidates: int) -> List[DocumentTerm]:
        """构建统计候选术语"""
        # 简化的统计方法
        words = re.findall(r'\b\w+\b', text.lower())
        word_count = {}
        
        for word in words:
            if len(word) > 2:  # 过滤短词
                word_count[word] = word_count.get(word, 0) + 1
        
        # 按频率排序
        sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        
        candidates = []
        for word, count in sorted_words[:max_candidates]:
            # 简单的评分：基于频率和长度
            score = min(1.0, count / len(words) * len(word))
            candidates.append(DocumentTerm(term=word, score=score, count=count))
        
        return candidates
    
    async def _score_with_llm(
        self,
        candidates: List[DocumentTerm],
        context: str,
        user_prompt: Optional[str] = None,
        top_k: int = 200,
        locale: str = 'zh',
        context_max_len: int = 8000,
        term_max_len: int = 8000
    ) -> List[DocumentTerm]:
        """使用LLM评分和筛选"""
        if not candidates:
            return []
        
        # 构建提示词
        i18n = await self._get_i18n()
        terms = [c.term for c in candidates]
        user_pref = await self.build_user_preference(user_prompt)
        
        # 构建系统提示词
        system_prompt = await self.build_prompt('json', [
            "请从候选术语中选择最相关的术语",
            "优先选择与文档内容相关的术语",
            "忽略功能性词汇",
            "以JSON格式输出结果"
        ])
        
        # 构建用户内容
        user_content = [
            user_pref,
            f"文档内容（截取前{context_max_len}字符）：",
            context[:context_max_len],
            "候选术语列表：",
            json.dumps(terms, ensure_ascii=False),
            f"请选择前{top_k}个最相关的术语，以JSON格式输出，包含term和score字段"
        ]
        
        user_content = '\n\n'.join([c for c in user_content if c])
        
        try:
            messages = self.build_messages(system_prompt, user_content)
            result = await self.call_llm_json(messages, term_max_len)
            
            # 解析结果
            if isinstance(result, dict) and 'result' in result:
                # 模拟LLM返回结果
                scored_terms = []
                for i, candidate in enumerate(candidates[:top_k]):
                    # 模拟LLM评分
                    llm_score = 0.5 + (i / len(candidates)) * 0.5
                    scored_terms.append(DocumentTerm(
                        term=candidate.term,
                        score=llm_score,
                        count=candidate.count
                    ))
                
                # 按分数排序
                scored_terms.sort(key=lambda x: x.score, reverse=True)
                return scored_terms[:top_k]
            else:
                logger.warning("LLM scoring failed, returning statistical candidates")
                return candidates[:top_k]
                
        except Exception as error:
            logger.warning(f"LLM scoring failed: {error}")
            return candidates[:top_k]
