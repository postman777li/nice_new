"""
篇章评估Agent - 评估篇章适当性
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

from ..base import BaseAgent, AgentConfig, AgentRunContext

logger = logging.getLogger(__name__)


@dataclass
class DiscourseEvaluateResult:
    terminology_consistency: float  # 用词一致性
    syntax_consistency: float       # 句法一致性
    style_consistency: float        # 风格一致性
    overall_score: float
    terminology_differences: List[str]  # 用词差异
    syntax_differences: List[str]       # 句法差异
    recommendations: List[str]


class DiscourseEvaluateAgent(BaseAgent):
    def __init__(self, locale: str = 'zh'):
        super().__init__(AgentConfig(
            name='discourse:evaluate',
            role='discourse_evaluator',
            domain='discourse',
            specialty='篇章适当性评估',
            quality='review',
            locale=locale
        ))

    async def execute(self, input_data: Dict[str, Any], ctx: Optional[AgentRunContext] = None) -> DiscourseEvaluateResult:
        """分析当前翻译与参考翻译的差异"""
        current_text = input_data.get('text', '')
        references = input_data.get('references', [])  # 应该是最佳的3个参考
        source_text = input_data.get('source_text', '')
        target_lang = input_data.get('target_lang', 'en')
        
        if not current_text or not references:
            return DiscourseEvaluateResult(
                terminology_consistency=1.0,
                syntax_consistency=1.0,
                style_consistency=1.0,
                overall_score=1.0,
                terminology_differences=[],
                syntax_differences=[],
                recommendations=[]
            )
        
        # 使用LLM分析当前翻译与参考翻译的差异
        messages = [
            {
                "role": "system",
                "content": f"""你是一个专业的法律翻译一致性分析专家。你的任务是比较当前翻译与参考翻译例子，找出用词和句法上的差异。

分析维度：
1. **用词一致性（Terminology Consistency）**：
   - 比较当前翻译与参考翻译在法律术语、专业词汇的选择上的差异
   - 识别用词不一致的地方（如同一概念使用了不同的翻译）
   - 评估术语选择是否符合参考翻译的风格

2. **句法一致性（Syntax Consistency）**：
   - 比较句式结构、语序安排的差异
   - 识别情态动词、连接词使用上的不同
   - 评估句法风格是否与参考翻译保持一致

3. **风格一致性（Style Consistency）**：
   - 比较整体语言风格、正式程度
   - 识别表达方式、修辞手法的差异
   - 评估是否符合参考翻译的篇章风格

输出要求：
- 明确指出具体的用词差异（列出对比）
- 明确指出具体的句法差异（列出对比）
- 给出改进建议使当前翻译与参考翻译风格一致

返回JSON格式：
{{
    "terminology_consistency": 0.85,
    "syntax_consistency": 0.80,
    "style_consistency": 0.90,
    "overall_score": 0.85,
    "terminology_differences": [
        "当前翻译使用'agreement'，参考翻译使用'contract'",
        "当前翻译使用'must'，参考翻译统一使用'shall'"
    ],
    "syntax_differences": [
        "当前翻译使用主动语态，参考翻译多用被动语态",
        "当前翻译条件句使用'if...then'，参考翻译使用'where'"
    ],
    "recommendations": [
        "建议将'agreement'改为'contract'以保持一致",
        "建议使用被动语态以符合参考风格"
    ]
}}"""
            },
            {
                "role": "user",
                "content": f"""请分析以下当前翻译与参考翻译例子的差异：

源文本：{source_text}

当前翻译：{current_text}

参考翻译例子（按相似度排序，前3个最佳匹配）：
{self._format_references(references)}

请详细比较当前翻译与这些参考例子在用词和句法上的差异，帮助我们调整当前翻译使其风格与参考保持一致。"""
            }
        ]
        
        try:
            result = await self.call_llm_json(messages)
            
            return DiscourseEvaluateResult(
                terminology_consistency=result.get('terminology_consistency', 1.0),
                syntax_consistency=result.get('syntax_consistency', 1.0),
                style_consistency=result.get('style_consistency', 1.0),
                overall_score=result.get('overall_score', 1.0),
                terminology_differences=result.get('terminology_differences', []),
                syntax_differences=result.get('syntax_differences', []),
                recommendations=result.get('recommendations', [])
            )
        except Exception as e:
            logger.error(f"Discourse evaluate failed: {e}")
            return DiscourseEvaluateResult(
                terminology_consistency=1.0,
                syntax_consistency=1.0,
                style_consistency=1.0,
                overall_score=1.0,
                terminology_differences=[],
                syntax_differences=[],
                recommendations=[]
            )
    
    def _format_references(self, references: List[Dict[str, Any]]) -> str:
        """格式化参考翻译"""
        if not references:
            return "无参考翻译"
        
        formatted = []
        for i, ref in enumerate(references, 1):
            formatted.append(f"\n参考例子 {i} (相似度: {ref.get('similarity_score', 0.0):.2f}):")
            formatted.append(f"  源文本: {ref.get('source_text', '')}")
            formatted.append(f"  译文: {ref.get('target_text', '')}")
            if ref.get('legal_domain'):
                formatted.append(f"  法律领域: {ref.get('legal_domain', '')}")
        
        return "\n".join(formatted)