"""
双语术语质量检验Agent - 过滤和验证双语术语对齐质量
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

from ..base import BaseAgent, AgentConfig, AgentRunContext

logger = logging.getLogger(__name__)


@dataclass
class QualityCheckResult:
    """质量检验结果"""
    is_valid: bool
    quality_score: float
    issues: List[str]
    suggestions: List[str]


@dataclass
class FilteredBilingualTerm:
    """过滤后的双语术语"""
    source_term: str
    target_term: str
    confidence: float
    category: str
    source_context: str
    target_context: str
    quality_score: float
    is_valid: bool
    law: str = ""
    domain: str = ""
    year: str = ""
    entry_id: str = ""


class BilingualTermQualityCheckAgent(BaseAgent):
    """双语术语质量检验智能体"""
    
    def __init__(self, locale: str = 'zh'):
        super().__init__(AgentConfig(
            name='preprocess:bilingual-term-quality-check',
            role='bilingual_terminology_quality_checker',
            domain='preprocess',
            specialty='双语术语质量检验',
            quality='review',
            locale=locale
        ))

    async def execute(self, input_data: Dict[str, Any], ctx: Optional[AgentRunContext] = None) -> List[FilteredBilingualTerm]:
        """对双语术语进行质量检验和过滤（支持批量处理）"""
        terms = input_data.get('terms', [])
        source_text = input_data.get('source_text', '')
        target_text = input_data.get('target_text', '')
        src_lang = input_data.get('src_lang', 'zh')
        tgt_lang = input_data.get('tgt_lang', 'en')
        batch_mode = input_data.get('batch_mode', True)  # 默认启用批量模式
        # 默认批次大小：10 个术语（推荐 8-15）
        default_batch_size = input_data.get('batch_size', 10)
        
        if not terms:
            return []
        
        # 如果启用批量模式且术语数量大于1，使用批量处理
        # 注意：如果术语数量超过 batch_size，需要分批处理以避免 LLM 响应过长
        if batch_mode and len(terms) > 1:
            logger.info(f"使用批量模式处理 {len(terms)} 个术语，批次大小: {default_batch_size}")
            
            # 如果术语数量超过 batch_size，分批处理
            if len(terms) > default_batch_size:
                all_filtered_terms = []
                for i in range(0, len(terms), default_batch_size):
                    batch_terms = terms[i:i + default_batch_size]
                    logger.info(f"处理批次 {i//default_batch_size + 1}/{(len(terms)-1)//default_batch_size + 1}: {len(batch_terms)} 个术语")
                    filtered_batch = await self._batch_check_quality(batch_terms, source_text, target_text, src_lang, tgt_lang)
                    all_filtered_terms.extend(filtered_batch)
                logger.info(f"批量质量检验完成: {len(terms)} -> {len(all_filtered_terms)} 个术语")
                return all_filtered_terms
            else:
                # 术语数量不超过 batch_size，一次性处理
                return await self._batch_check_quality(terms, source_text, target_text, src_lang, tgt_lang)
        
        # 否则使用逐个处理
        logger.info(f"使用逐个模式处理 {len(terms)} 个术语")
        filtered_terms = []
        for term in terms:
            try:
                quality_result = await self._check_term_quality(term, source_text, target_text, src_lang, tgt_lang)
                
                filtered_term = FilteredBilingualTerm(
                    source_term=term.get('source_term', ''),
                    target_term=term.get('target_term', ''),
                    confidence=term.get('confidence', 0.0),
                    category=term.get('category', ''),
                    source_context=term.get('source_context', ''),
                    target_context=term.get('target_context', ''),
                    quality_score=quality_result.quality_score,
                    is_valid=quality_result.is_valid,
                    law=term.get('law', ''),
                    domain=term.get('domain', ''),
                    year=term.get('year', ''),
                    entry_id=term.get('entry_id', '')
                )
                
                if quality_result.is_valid:
                    filtered_terms.append(filtered_term)
                else:
                    logger.info(f"过滤掉低质量术语: {term.get('source_term', '')} -> {term.get('target_term', '')}")
                    logger.info(f"质量问题: {', '.join(quality_result.issues)}")
                
            except Exception as e:
                logger.error(f"质量检验失败: {e}")
                continue
        
        logger.info(f"质量检验完成: {len(terms)} -> {len(filtered_terms)} 个术语")
        return filtered_terms
    
    async def _batch_check_quality(self, terms: List[Dict[str, Any]], source_text: str, target_text: str, src_lang: str, tgt_lang: str) -> List[FilteredBilingualTerm]:
        """批量检查多个术语的质量"""
        # 构建术语列表文本
        terms_text = "\n".join([
            f"{i+1}. 源术语: {term.get('source_term', '')} | 目标术语: {term.get('target_term', '')} | 置信度: {term.get('confidence', 0.0)}"
            for i, term in enumerate(terms)
        ])
        
        # 使用LLM进行批量质量检验
        messages = [
            {
                "role": "system",
                "content": f"""你是一个专业的法律术语质量检验专家。请批量评估给定的双语术语对齐对的质量。

评估标准：
1. 语义一致性：源语言和目标语言术语是否表达相同的法律概念
2. 术语准确性：翻译是否准确、专业
3. 法律相关性：术语是否具有法律意义
4. 完整性：术语是否完整，不是片段
5. 一致性：术语在上下文中的使用是否一致

请为每个术语对提供：
- 是否有效（true/false）
- 质量评分（0-1）
- 发现的问题列表
- 改进建议列表

返回JSON格式（必须严格按照此格式）：
{{
    "results": [
        {{
            "index": 0,
            "is_valid": true,
            "quality_score": 0.9,
            "issues": [],
            "suggestions": []
        }},
        {{
            "index": 1,
            "is_valid": false,
            "quality_score": 0.3,
            "issues": ["语义不一致"],
            "suggestions": ["需要重新翻译"]
        }}
    ]
}}"""
            },
            {
                "role": "user",
                "content": f"""请批量评估以下双语术语对齐对的质量，以json格式返回：

术语列表（共{len(terms)}个）：
{terms_text}

上下文：
源语言文本（{src_lang}）：{source_text[:500]}...
目标语言文本（{tgt_lang}）：{target_text[:500]}...

请评估这些术语对齐对的质量，并严格按照上面的JSON格式返回评估结果。每个术语都必须有对应的评估结果。"""
            }
        ]
        
        try:
            result = await self.call_llm_json(messages)
            
            if 'error' in result:
                logger.warning(f"批量质量检验LLM调用失败: {result['error']}")
                return []
            
            if 'raw' in result:
                try:
                    import json
                    result = json.loads(result['raw'])
                except json.JSONDecodeError as e:
                    logger.warning(f"批量质量检验LLM返回内容不是有效JSON: {e}")
                    return []
            
            quality_results = result.get('results', [])
            
            if not quality_results:
                logger.warning("批量质量检验未返回任何结果")
                return []
            
            # 创建过滤后的术语列表
            filtered_terms = []
            for i, term in enumerate(terms):
                # 找到对应的质量检验结果
                quality_result = next((r for r in quality_results if r.get('index') == i), None)
                if not quality_result:
                    logger.warning(f"术语 {i} 没有对应的质量检验结果，跳过")
                    continue
                
                is_valid = quality_result.get('is_valid', False)
                quality_score = quality_result.get('quality_score', 0.0)
                
                filtered_term = FilteredBilingualTerm(
                    source_term=term.get('source_term', ''),
                    target_term=term.get('target_term', ''),
                    confidence=term.get('confidence', 0.0),
                    category=term.get('category', ''),
                    source_context=term.get('source_context', ''),
                    target_context=term.get('target_context', ''),
                    quality_score=quality_score,
                    is_valid=is_valid,
                    law=term.get('law', ''),
                    domain=term.get('domain', ''),
                    year=term.get('year', ''),
                    entry_id=term.get('entry_id', '')
                )
                
                if is_valid:
                    filtered_terms.append(filtered_term)
                else:
                    logger.info(f"过滤掉低质量术语: {term.get('source_term', '')} -> {term.get('target_term', '')}")
                    logger.info(f"质量问题: {', '.join(quality_result.get('issues', []))}")
            
            logger.info(f"批量质量检验完成: {len(terms)} -> {len(filtered_terms)} 个术语")
            return filtered_terms
            
        except Exception as e:
            logger.error(f"批量质量检验失败: {e}")
            return []
    
    async def _check_term_quality(self, term: Dict[str, Any], source_text: str, target_text: str, src_lang: str, tgt_lang: str) -> QualityCheckResult:
        """检查单个术语的质量"""
        source_term = term.get('source_term', '')
        target_term = term.get('target_term', '')
        confidence = term.get('confidence', 0.0)
        
        # 使用LLM进行质量检验
        messages = [
            {
                "role": "system",
                "content": f"""你是一个专业的法律术语质量检验专家。请评估给定的双语术语对齐对的质量。

评估标准：
1. 语义一致性：源语言和目标语言术语是否表达相同的法律概念
2. 术语准确性：翻译是否准确、专业
3. 法律相关性：术语是否具有法律意义
4. 完整性：术语是否完整，不是片段
5. 一致性：术语在上下文中的使用是否一致

请为每个术语对提供：
- 是否有效（true/false）
- 质量评分（0-1）
- 发现的问题
- 改进建议

返回JSON格式：
{{
    "is_valid": true/false,
    "quality_score": 0.9,
    "issues": ["问题1", "问题2"],
    "suggestions": ["建议1", "建议2"]
}}"""
            },
            {
                "role": "user",
                "content": f"""请评估以下双语术语对齐对的质量，以json格式返回：

源语言术语（{src_lang}）：{source_term}
目标语言术语（{tgt_lang}）：{target_term}
置信度：{confidence}

上下文：
源语言文本：{source_text}
目标语言文本：{target_text}

请评估这个术语对齐对的质量，并以json格式提供详细的评估结果。"""
            }
        ]
        
        try:
            result = await self.call_llm_json(messages)
            
            return QualityCheckResult(
                is_valid=result.get('is_valid', False),
                quality_score=result.get('quality_score', 0.0),
                issues=result.get('issues', []),
                suggestions=result.get('suggestions', [])
            )
        except Exception as e:
            logger.error(f"质量检验LLM调用失败: {e}")
            # 返回默认的低质量结果
            return QualityCheckResult(
                is_valid=False,
                quality_score=0.0,
                issues=["质量检验失败"],
                suggestions=["需要人工审核"]
            )
