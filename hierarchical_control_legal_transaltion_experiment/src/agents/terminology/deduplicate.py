"""
术语去重Agent - 合并和去重提取的术语
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import logging

from ..base import BaseAgent, AgentConfig, AgentRunContext
from .mono_extract import MonoExtractItem

logger = logging.getLogger(__name__)


@dataclass
class DeduplicatedTerm:
    """去重后的术语"""
    term: str
    count: int
    score: float  # 取最高分
    contexts: List[str]  # 1-2个上下文句子
    category: str


class DeduplicateAgent(BaseAgent):
    """术语去重Agent"""
    
    def __init__(self, locale: str = 'zh'):
        super().__init__(AgentConfig(
            name='terminology:deduplicate',
            role='terminology_deduplicator',
            domain='terminology',
            specialty='术语去重与合并',
            quality='review',
            locale=locale
        ))
    
    async def execute(self, input_data: Dict[str, Any], ctx: Optional[AgentRunContext] = None) -> List[DeduplicatedTerm]:
        """
        去重和合并提取的术语
        
        Args:
            input_data: {
                'extracted_terms': List[List[MonoExtractItem]], # 多个文本提取的术语列表
                'contexts': List[str], # 对应的原始文本
                'max_contexts': int  # 每个术语保留的上下文数量（默认2）
            }
        
        Returns:
            List[DeduplicatedTerm]: 去重后的术语列表
        """
        extracted_terms_list = input_data.get('extracted_terms', [])
        contexts = input_data.get('contexts', [])
        max_contexts = input_data.get('max_contexts', 2)
        
        if not extracted_terms_list:
            return []
        
        # 使用字典存储术语，键为术语文本
        term_dict = defaultdict(lambda: {
            'count': 0,
            'max_score': 0.0,
            'contexts': [],
            'categories': set()
        })
        
        # 合并所有提取的术语
        for idx, terms in enumerate(extracted_terms_list):
            context = contexts[idx] if idx < len(contexts) else ''
            
            for item in terms:
                if not item.term or not item.term.strip():
                    continue
                
                term_key = item.term.strip()
                term_data = term_dict[term_key]
                
                # 更新统计
                term_data['count'] += 1
                term_data['max_score'] = max(term_data['max_score'], item.score)
                
                # 添加上下文（去重，限制数量）
                if context and context not in term_data['contexts']:
                    if len(term_data['contexts']) < max_contexts:
                        term_data['contexts'].append(context)
                
                # 收集类别
                if item.category:
                    term_data['categories'].add(item.category)
        
        # 转换为DeduplicatedTerm列表
        deduplicated = []
        for term, data in term_dict.items():
            # 选择最常见的类别，如果有多个则选第一个
            category = list(data['categories'])[0] if data['categories'] else ''
            
            deduplicated.append(DeduplicatedTerm(
                term=term,
                count=data['count'],
                score=data['max_score'],
                contexts=data['contexts'],
                category=category
            ))
        
        # 按出现次数和分数排序
        deduplicated.sort(key=lambda x: (x.count, x.score), reverse=True)
        
        logger.info(f"去重前: {sum(len(terms) for terms in extracted_terms_list)} 个术语")
        logger.info(f"去重后: {len(deduplicated)} 个不重复术语")
        
        return deduplicated

