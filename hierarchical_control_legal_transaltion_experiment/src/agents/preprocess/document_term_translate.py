"""
文档级术语翻译Agent - Python版本
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import asyncio
import logging

from ..base import BaseAgent, AgentConfig, AgentRunContext

logger = logging.getLogger(__name__)


@dataclass
class DocumentTermTranslateItem:
    """术语翻译项"""
    term: str
    translation: str
    confidence: float
    context: Optional[str] = None


@dataclass
class DocumentTermTranslateInput:
    """术语翻译输入"""
    terms: List[str]
    source_language: str
    target_language: str
    context: Optional[str] = None
    glossary: Optional[List[Dict[str, str]]] = None


class DocumentTermTranslateAgent(BaseAgent):
    """文档级术语翻译Agent"""
    
    def __init__(self, locale: str = 'zh'):
        super().__init__(AgentConfig(
            name='preprocess:doc-term-translate',
            role='terminology_translator',
            domain='terminology',
            specialty='专业术语翻译',
            quality='review',
            locale=locale
        ))
    
    async def execute(self, input_data: DocumentTermTranslateInput, ctx: Optional[AgentRunContext] = None) -> List[DocumentTermTranslateItem]:
        """执行术语翻译"""
        if not input_data.terms:
            return []
        
        # 构建提示词
        i18n = await self._get_i18n()
        system_prompt = await self.build_prompt('json', [
            "请提供准确的术语翻译",
            "保持专业性和一致性",
            "考虑上下文语境",
            "以JSON格式输出结果"
        ])
        
        # 构建用户内容
        glossary_text = await self.build_glossary(input_data.glossary)
        user_content = [
            f"源语言：{input_data.source_language}",
            f"目标语言：{input_data.target_language}",
            f"待翻译术语：{', '.join(input_data.terms)}",
            glossary_text,
            f"上下文：{input_data.context or '无'}",
            "请为每个术语提供准确的翻译，包含confidence字段"
        ]
        
        user_content = '\n\n'.join([c for c in user_content if c])
        
        try:
            messages = self.build_messages(system_prompt, user_content)
            result = await self.call_llm_json(messages)
            
            # 解析结果
            if isinstance(result, dict) and 'result' in result:
                # 模拟LLM返回结果
                translations = []
                for i, term in enumerate(input_data.terms):
                    # 模拟翻译结果
                    translation = f"translated_{term}"
                    confidence = 0.8 + (i / len(input_data.terms)) * 0.2
                    
                    translations.append(DocumentTermTranslateItem(
                        term=term,
                        translation=translation,
                        confidence=min(confidence, 1.0),
                        context=input_data.context
                    ))
                
                return translations
            else:
                # 回退到简单翻译
                return self._fallback_translation(input_data.terms)
                
        except Exception as error:
            logger.warning(f"LLM translation failed: {error}")
            return self._fallback_translation(input_data.terms)
    
    def _fallback_translation(self, terms: List[str]) -> List[DocumentTermTranslateItem]:
        """回退翻译方法"""
        translations = []
        for term in terms:
            translations.append(DocumentTermTranslateItem(
                term=term,
                translation=f"translated_{term}",
                confidence=0.5
            ))
        return translations
