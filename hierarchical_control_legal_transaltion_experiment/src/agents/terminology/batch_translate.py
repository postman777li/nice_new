"""
批量术语翻译Agent - 批量翻译术语，复用SearchAgent查询已有翻译
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import asyncio
import logging

from ..base import BaseAgent, AgentConfig, AgentRunContext
from .deduplicate import DeduplicatedTerm
from .search import SearchAgent

logger = logging.getLogger(__name__)


@dataclass
class BatchTranslationResult:
    """批量翻译结果"""
    term: str
    translation: str
    source: str  # 'database' or 'llm'
    confidence: float
    contexts: List[str]


class BatchTranslateAgent(BaseAgent):
    """批量术语翻译Agent"""
    
    def __init__(self, locale: str = 'zh', db_path: str = 'backend/terms.db'):
        super().__init__(AgentConfig(
            name='terminology:batch-translate',
            role='batch_translator',
            domain='terminology',
            specialty='批量术语翻译',
            quality='review',
            locale=locale
        ))
        self.db_path = db_path
        self.search_agent = SearchAgent(locale=locale, db_path=db_path)
    
    async def execute(self, input_data: Dict[str, Any], ctx: Optional[AgentRunContext] = None) -> List[BatchTranslationResult]:
        """
        批量翻译术语，优先使用数据库中已有的翻译
        
        Args:
            input_data: {
                'terms': List[DeduplicatedTerm],
                'src_lang': str,
                'tgt_lang': str,
                'domain': str,
                'batch_size': int  # 每批翻译的术语数量（默认20）
            }
        
        Returns:
            List[BatchTranslationResult]: 翻译结果列表
        """
        terms = input_data.get('terms', [])
        src_lang = input_data.get('src_lang', 'zh')
        tgt_lang = input_data.get('tgt_lang', 'en')
        domain = input_data.get('domain', 'law')
        batch_size = input_data.get('batch_size', 20)
        
        if not terms:
            return []
        
        logger.info(f"开始批量翻译 {len(terms)} 个术语")
        
        # 步骤1: 查询数据库中已有的翻译
        logger.info("步骤1: 查询数据库中已有的翻译...")
        db_results = await self._search_in_database(terms, src_lang, tgt_lang, domain)
        
        found_terms = {r.term for r in db_results}
        logger.info(f"  从数据库找到 {len(found_terms)} 个术语的翻译")
        
        # 步骤2: 对未找到翻译的术语进行批量翻译
        terms_to_translate = [t for t in terms if t.term not in found_terms]
        logger.info(f"  需要翻译 {len(terms_to_translate)} 个新术语")
        
        llm_results = []
        if terms_to_translate:
            logger.info("步骤2: 批量翻译新术语...")
            llm_results = await self._batch_translate_terms(
                terms_to_translate, 
                src_lang, 
                tgt_lang, 
                batch_size
            )
        
        # 合并结果
        all_results = db_results + llm_results
        logger.info(f"✓ 完成翻译: 数据库{len(db_results)}个 + LLM{len(llm_results)}个 = 共{len(all_results)}个")
        
        return all_results
    
    async def _search_in_database(self, 
                                  terms: List[DeduplicatedTerm],
                                  src_lang: str,
                                  tgt_lang: str,
                                  domain: str) -> List[BatchTranslationResult]:
        """在数据库中查询术语翻译"""
        # 批量查询
        term_list = [t.term for t in terms]
        search_results = await self.search_agent.execute({
            'terms': term_list,
            'source_lang': src_lang,
            'target_lang': tgt_lang,
            'domain': domain,
            'exact_match': True
        })
        
        # 转换为BatchTranslationResult
        results = []
        for search_result in search_results:
            # 找到对应的DeduplicatedTerm以获取contexts
            matching_term = next((t for t in terms if t.term == search_result.term), None)
            contexts = matching_term.contexts if matching_term else []
            
            results.append(BatchTranslationResult(
                term=search_result.term,
                translation=search_result.translation,
                source='database',
                confidence=search_result.confidence,
                contexts=contexts
            ))
        
        return results
    
    async def _batch_translate_terms(self,
                                     terms: List[DeduplicatedTerm],
                                     src_lang: str,
                                     tgt_lang: str,
                                     batch_size: int) -> List[BatchTranslationResult]:
        """批量翻译术语（分批处理）"""
        all_results = []
        
        for i in range(0, len(terms), batch_size):
            batch = terms[i:i + batch_size]
            logger.info(f"  翻译批次 {i//batch_size + 1}/{(len(terms)-1)//batch_size + 1} ({len(batch)} 个术语)")
            
            results = await self._translate_batch(batch, src_lang, tgt_lang)
            all_results.extend(results)
        
        return all_results
    
    async def _translate_batch(self,
                               terms: List[DeduplicatedTerm],
                               src_lang: str,
                               tgt_lang: str) -> List[BatchTranslationResult]:
        """翻译一批术语"""
        # 构建术语列表，包含上下文
        term_items = []
        for term in terms:
            item = {
                'term': term.term,
                'category': term.category,
                'contexts': term.contexts[:2]  # 最多2个上下文
            }
            term_items.append(item)
        
        # 调用LLM批量翻译
        messages = [
            {
                "role": "system",
                "content": f"""你是专业的法律术语翻译专家。请将以下中文法律术语翻译成英文。

翻译要求：
1. 使用标准的法律英语表达
2. 保持术语的专业性和准确性
3. 参考提供的上下文确定最佳翻译
4. 遵循法律翻译的行业标准

返回JSON格式：
{{
    "translations": [
        {{
            "term": "源术语",
            "translation": "Translated Term",
            "confidence": 0.9
        }}
    ]
}}"""
            },
            {
                "role": "user",
                "content": f"请翻译以下{len(terms)}个法律术语：\n\n```json\n{self._format_terms_for_llm(term_items)}\n```"
            }
        ]
        
        try:
            response = await self.call_llm_json(messages, temperature=0.1)
            translations = response.get('translations', [])
            
            # 转换为BatchTranslationResult
            results = []
            for item in translations:
                term_text = item.get('term', '')
                translation = item.get('translation', '')
                
                if not term_text or not translation:
                    continue
                
                # 找到对应的DeduplicatedTerm以获取完整信息
                matching_term = next((t for t in terms if t.term == term_text), None)
                contexts = matching_term.contexts if matching_term else []
                
                results.append(BatchTranslationResult(
                    term=term_text,
                    translation=translation,
                    source='llm',
                    confidence=item.get('confidence', 0.8),
                    contexts=contexts
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"批量翻译失败: {e}")
            # 返回空翻译，但保留术语信息
            return [
                BatchTranslationResult(
                    term=t.term,
                    translation='',
                    source='llm',
                    confidence=0.0,
                    contexts=t.contexts
                )
                for t in terms
            ]
    
    def _format_terms_for_llm(self, term_items: List[Dict]) -> str:
        """格式化术语列表供LLM使用"""
        import json
        formatted = []
        for item in term_items:
            formatted.append({
                'term': item['term'],
                'category': item.get('category', ''),
                'example_context': item['contexts'][0] if item['contexts'] else ''
            })
        return json.dumps(formatted, ensure_ascii=False, indent=2)

