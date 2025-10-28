"""
基线翻译Agent - 纯LLM直接翻译，不使用任何控制策略
"""
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

from .base import BaseAgent, AgentConfig, AgentRunContext

logger = logging.getLogger(__name__)


@dataclass
class BaselineTranslationResult:
    source_text: str
    translated_text: str
    confidence: float


class BaselineTranslationAgent(BaseAgent):
    """纯基线翻译Agent - 直接使用LLM翻译，无任何控制策略"""
    
    def __init__(self, locale: str = 'zh'):
        super().__init__(AgentConfig(
            name='baseline:translation',
            role='baseline_translator',
            domain='general',
            specialty='纯LLM翻译（无控制策略）',
            quality='baseline',
            locale=locale
        ))
    
    async def execute(self, input_data: Dict[str, Any], ctx: Optional[AgentRunContext] = None) -> BaselineTranslationResult:
        """直接翻译，不使用任何术语库、规则或控制策略"""
        source_text = input_data.get('source_text', '')
        source_lang = input_data.get('source_lang', 'zh')
        target_lang = input_data.get('target_lang', 'en')
        
        if not source_text:
            return BaselineTranslationResult(
                source_text=source_text,
                translated_text='',
                confidence=0.0
            )
        
        # 构建纯翻译prompt（无任何约束）
        messages = [
            {
                "role": "system",
                "content": f"""You are a professional translator. Translate the following text from {source_lang} to {target_lang}.

Requirements:
- Provide only the translation, no explanations
- Maintain the meaning and tone of the original text
- Use natural and fluent language

Return JSON format:
{{
    "translated_text": "翻译文本",
    "confidence": 0.9
}}"""
            },
            {
                "role": "user",
                "content": f"""Translate the following {source_lang} text to {target_lang}:

{source_text}"""
            }
        ]
        
        try:
            result = await self.call_llm_json(messages)
            
            return BaselineTranslationResult(
                source_text=source_text,
                translated_text=result.get('translated_text', source_text),
                confidence=result.get('confidence', 0.0)
            )
        except Exception as e:
            logger.error(f"Baseline translation failed: {e}")
            return BaselineTranslationResult(
                source_text=source_text,
                translated_text=source_text,
                confidence=0.0
            )

