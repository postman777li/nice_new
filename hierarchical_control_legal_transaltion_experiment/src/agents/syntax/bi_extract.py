"""
双语句法提取Agent - 从双语文本对中提取模态和连接词映射表
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

from ..base import BaseAgent, AgentConfig, AgentRunContext

logger = logging.getLogger(__name__)


@dataclass
class BiExtractItem:
    source_pattern: str
    target_pattern: str
    modality_type: str
    confidence: float
    context: str


class BiExtractAgent(BaseAgent):
    def __init__(self, locale: str = 'zh'):
        super().__init__(AgentConfig(
            name='syntax:bi-extract',
            role='syntax_extractor',
            domain='syntax',
            specialty='双语句法模式提取',
            quality='review',
            locale=locale
        ))

    async def execute(self, input_data: Dict[str, Any], ctx: Optional[AgentRunContext] = None) -> List[BiExtractItem]:
        """从双语文本对中提取模态和连接词映射表"""
        source_text = input_data.get('source_text', '')
        target_text = input_data.get('target_text', '')
        source_lang = input_data.get('source_lang', 'zh')
        target_lang = input_data.get('target_lang', 'en')
        source_lang_text = "中文" if source_lang == 'zh' else source_lang
        target_lang_text = "英语" if target_lang == 'en' else "日语"
        if not source_text or not target_text:
            return []
        
        # 使用LLM提取双语句法模式
        messages = [
            {
                "role": "system",
                "content": f"""你是一个**极其苛刻**的法律领域翻译句法分析专家。请从给定的双语法律文本对中提取关键的句法模式映射。

分析重点：
1. 情态动词映射（如 "shall" → "必须"）
2. 逻辑连接词模式
3. 条件句结构
4. 被动语态

请为每个模式提供：
- 源语言模式
- 目标语言模式
- 模态类型
- 置信度评分
- 上下文

返回JSON格式：
{{
    "patterns": [
        {{
            "source_pattern": "源语言模式",
            "target_pattern": "目标语言模式",
            "modality_type": "模态类型",
            "confidence": 0.9,
            "context": "使用上下文"
        }}
    ]
}}"""
            },
            {
                "role": "user",
                "content": f"""请分析以下{source_lang_text}到{target_lang_text}的法律文本对，提取句法模式映射：

源文本：{source_text}

目标文本：{target_text}"""
            }
        ]
        
        try:
            result = await self.call_llm_json(messages)
            patterns_data = result.get('patterns', [])
            
            return [
                BiExtractItem(
                    source_pattern=item.get('source_pattern', ''),
                    target_pattern=item.get('target_pattern', ''),
                    modality_type=item.get('modality_type', ''),
                    confidence=item.get('confidence', 0.0),
                    context=item.get('context', '')
                )
                for item in patterns_data
            ]
        except Exception as e:
            logger.error(f"Bi extract failed: {e}")
            return []
