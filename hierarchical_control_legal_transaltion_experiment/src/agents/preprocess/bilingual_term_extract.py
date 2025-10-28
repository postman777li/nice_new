"""
双语术语提取Agent - 从双语文本中提取平行术语翻译
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

from ..base import BaseAgent, AgentConfig, AgentRunContext

logger = logging.getLogger(__name__)


@dataclass
class BilingualTermItem:
    """双语术语提取结果项"""
    source_term: str
    target_term: str
    confidence: float
    category: str
    source_context: str
    target_context: str


class BilingualTermExtractAgent(BaseAgent):
    """双语术语提取智能体"""
    
    def __init__(self, locale: str = 'zh'):
        super().__init__(AgentConfig(
            name='preprocess:bilingual-term-extract',
            role='bilingual_terminology_extractor',
            domain='preprocess',
            specialty='双语术语提取',
            quality='review',
            locale=locale
        ))

    async def execute(self, input_data: Dict[str, Any], ctx: Optional[AgentRunContext] = None) -> List[BilingualTermItem]:
        """从双语文本中提取平行术语翻译（支持批量处理）"""
        # 检查是否为批量模式
        text_pairs = input_data.get('text_pairs', None)
        batch_mode = input_data.get('batch_mode', False)
        # 默认批次大小：3 个法条对（推荐 2-5）
        default_batch_size = input_data.get('batch_size', 3)
        
        # 如果提供了 text_pairs，使用批量模式
        if text_pairs and batch_mode:
            logger.info(f"使用批量模式处理 {len(text_pairs)} 个文本对")
            return await self._batch_extract_terms(text_pairs, 
                                                   input_data.get('src_lang', 'zh'),
                                                   input_data.get('tgt_lang', 'en'))
        
        # 否则使用单个文本对模式
        source_text = input_data.get('source_text', '')
        target_text = input_data.get('target_text', '')
        src_lang = input_data.get('src_lang', 'zh')
        tgt_lang = input_data.get('tgt_lang', 'en')
        
        if not source_text or not target_text:
            return []
        
        # 使用LLM提取双语术语
        messages = [
            {
                "role": "system",
                "content": f"""你是一个专业的法律术语提取专家。请从给定的双语法律文本中提取对应的术语翻译对。

提取标准：
1. 法律概念和术语的对应翻译
2. 专业法律词汇的平行翻译
3. 具有法律意义的复合词翻译
4. 需要标准化翻译的术语对

请为每个术语对提供：
- 源语言术语
- 目标语言术语
- 置信度评分（0-1）
- 术语类别
- 源语言上下文信息
- 目标语言上下文信息

返回JSON格式：
{{
    "terms": [
        {{
            "source_term": "源语言术语",
            "target_term": "目标语言术语", 
            "confidence": 0.9,
            "category": "术语类别",
            "source_context": "源语言上下文",
            "target_context": "目标语言上下文"
        }}
    ]
}}"""
            },
            {
                "role": "user", 
                "content": f"""请从以下双语法律文本中提取术语翻译对，以json格式返回：

源语言文本（{src_lang}）：
{source_text}

目标语言文本（{tgt_lang}）：
{target_text}

请提取对应的术语翻译对，确保术语在两种语言中具有相同的法律含义。返回json格式的结果。"""
            }
        ]
        
        try:
            result = await self.call_llm_json(messages)
            terms_data = result.get('terms', [])
            
            return [
                BilingualTermItem(
                    source_term=item.get('source_term', ''),
                    target_term=item.get('target_term', ''),
                    confidence=item.get('confidence', 0.0),
                    category=item.get('category', ''),
                    source_context=item.get('source_context', ''),
                    target_context=item.get('target_context', '')
                )
                for item in terms_data
            ]
        except Exception as e:
            logger.error(f"Bilingual term extract failed: {e}")
            return []
    
    async def _batch_extract_terms(self, text_pairs: List[Dict[str, str]], src_lang: str, tgt_lang: str) -> List[BilingualTermItem]:
        """批量从多个双语文本对中提取平行术语翻译"""
        if not text_pairs:
            return []
        
        # 构建批量文本对列表
        text_pairs_text = "\n\n".join([
            f"=== 文本对 {i+1} ===\n源语言文本（{src_lang}）：\n{pair.get('source_text', '')}\n\n目标语言文本（{tgt_lang}）：\n{pair.get('target_text', '')}"
            for i, pair in enumerate(text_pairs)
        ])
        
        # 使用LLM进行批量术语提取
        messages = [
            {
                "role": "system",
                "content": f"""你是一个专业的法律术语提取专家。请从给定的多个双语法律文本对中批量提取对应的术语翻译对。

提取标准：
1. 法律概念和术语的对应翻译
2. 专业法律词汇的平行翻译
3. 具有法律意义的复合词翻译
4. 需要标准化翻译的术语对

请为每个术语对提供：
- 源语言术语
- 目标语言术语
- 置信度评分（0-1）
- 术语类别
- 源语言上下文信息
- 目标语言上下文信息
- 文本对索引（从0开始）

返回JSON格式（必须严格按照此格式）：
{{
    "results": [
        {{
            "text_pair_index": 0,
            "terms": [
                {{
                    "source_term": "源语言术语",
                    "target_term": "目标语言术语", 
                    "confidence": 0.9,
                    "category": "术语类别",
                    "source_context": "源语言上下文",
                    "target_context": "目标语言上下文"
                }}
            ]
        }},
        {{
            "text_pair_index": 1,
            "terms": [...]
        }}
    ]
}}"""
            },
            {
                "role": "user", 
                "content": f"""请从以下{len(text_pairs)}个双语法律文本对中批量提取术语翻译对，以json格式返回：

{text_pairs_text}

请为每个文本对提取对应的术语翻译对，确保术语在两种语言中具有相同的法律含义。严格按照上面的JSON格式返回结果。"""
            }
        ]
        
        try:
            result = await self.call_llm_json(messages)
            
            if 'error' in result:
                logger.warning(f"批量术语提取LLM调用失败: {result['error']}")
                return []
            
            if 'raw' in result:
                try:
                    import json
                    result = json.loads(result['raw'])
                except json.JSONDecodeError as e:
                    logger.warning(f"批量术语提取LLM返回内容不是有效JSON: {e}")
                    return []
            
            extraction_results = result.get('results', [])
            
            if not extraction_results:
                logger.warning("批量术语提取未返回任何结果")
                return []
            
            # 收集所有提取的术语
            all_terms = []
            for extraction_result in extraction_results:
                text_pair_index = extraction_result.get('text_pair_index', -1)
                terms_data = extraction_result.get('terms', [])
                
                for item in terms_data:
                    term = BilingualTermItem(
                        source_term=item.get('source_term', ''),
                        target_term=item.get('target_term', ''),
                        confidence=item.get('confidence', 0.0),
                        category=item.get('category', ''),
                        source_context=item.get('source_context', ''),
                        target_context=item.get('target_context', '')
                    )
                    all_terms.append(term)
            
            logger.info(f"批量术语提取完成: 从 {len(text_pairs)} 个文本对中提取了 {len(all_terms)} 个术语")
            return all_terms
            
        except Exception as e:
            logger.error(f"批量术语提取失败: {e}")
            return []
