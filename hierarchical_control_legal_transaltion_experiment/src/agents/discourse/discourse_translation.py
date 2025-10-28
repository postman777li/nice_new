"""
篇章翻译Agent - 综合多个输入生成最终翻译
支持多候选生成和COMET-Kiwi质量评估选择
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

from ..base import BaseAgent, AgentConfig, AgentRunContext

logger = logging.getLogger(__name__)


@dataclass
class DiscourseTranslationResult:
    source_text: str
    final_text: str
    integrated_references: List[Dict[str, Any]]
    confidence: float
    memory_updates: List[Dict[str, Any]]
    candidates: Optional[List[str]] = None  # 多个候选翻译（如果生成了）


class DiscourseTranslationAgent(BaseAgent):
    def __init__(self, locale: str = 'zh', generate_candidates: bool = False, num_candidates: int = 3):
        """
        初始化篇章翻译Agent
        
        Args:
            locale: 语言地区
            generate_candidates: 是否生成多个候选翻译（默认False，只生成一个）
            num_candidates: 生成的候选数量（仅当generate_candidates=True时有效）
        """
        super().__init__(AgentConfig(
            name='discourse:discourse-translation',
            role='discourse_translator',
            domain='discourse',
            specialty='篇章综合翻译',
            quality='review',
            locale=locale
        ))
        self.generate_candidates = generate_candidates
        self.num_candidates = num_candidates
        
        if self.generate_candidates:
            logger.info(f"✓ 篇章翻译Agent配置为生成{num_candidates}个候选翻译")

    async def execute(self, input_data: Dict[str, Any], ctx: Optional[AgentRunContext] = None) -> DiscourseTranslationResult:
        """综合多个输入生成最终翻译（支持多候选选择）"""
        source_text = input_data.get('source_text', '')
        current_translation = input_data.get('current_translation', '')
        selected_references = input_data.get('selected_references', [])
        evaluation = input_data.get('evaluation', None)  # 差异分析结果
        syntactic_suggestions = input_data.get('syntactic_suggestions', [])
        source_lang = input_data.get('source_lang', 'zh')
        target_lang = input_data.get('target_lang', 'en')
        
        if not source_text or not current_translation:
            return DiscourseTranslationResult(
                source_text=source_text,
                final_text=current_translation,
                integrated_references=[],
                confidence=0.0,
                memory_updates=[]
            )
        
        # 根据配置选择生成方式
        if self.generate_candidates:
            candidates = await self._generate_discourse_candidates(
                source_text, current_translation, selected_references,
                evaluation, syntactic_suggestions, source_lang, target_lang
            )
            if not candidates:
                # 如果候选生成失败，fallback到单次翻译
                logger.warning("候选生成失败，fallback到单次翻译")
                return await self._execute_single(
                    source_text, current_translation, selected_references,
                    evaluation, syntactic_suggestions, source_lang, target_lang
                )
            # 返回第一个候选作为final_text，同时保留所有候选
            return DiscourseTranslationResult(
                source_text=source_text,
                final_text=candidates[0],
                integrated_references=[{"reference": "LLM生成候选", "applied": True}],
                confidence=0.8,  # 默认置信度，实际会由选择器Agent决定
                memory_updates=[],
                candidates=candidates
            )
        else:
            return await self._execute_single(
                source_text, current_translation, selected_references,
                evaluation, syntactic_suggestions, source_lang, target_lang
            )
    
    async def _execute_single(
        self,
        source_text: str,
        current_translation: str,
        selected_references: List[Dict[str, Any]],
        evaluation: Any,
        syntactic_suggestions: List[Dict[str, Any]],
        source_lang: str,
        target_lang: str
    ) -> DiscourseTranslationResult:
        """原有的单次整合逻辑（向后兼容）"""
        # 注释：移除了所有早期退出逻辑，强制尝试篇章层改进  
        
        # 使用LLM综合生成最终翻译
        messages = [
            {
                "role": "system",
                "content": f"""你是一个专业的法律篇章翻译专家。你的任务是**审查当前翻译的质量，仅在发现明确问题时进行修正**。

**核心原则**：
1. **保守修改**：当前翻译已经过术语层和句法层优化，质量通常已经很好
2. **仅修正明确问题**：只在发现明显错误或不一致时才修改
3. **参考仅供参考**：参考译文可能来自不同版本或风格，不应强制对齐

**什么情况下应该修改**：
✅ 当前翻译有明显的语法错误
✅ 当前翻译有术语不一致（同一术语在不同地方翻译不同）
✅ 当前翻译有明显的语义错误或遗漏
✅ 评估明确指出严重的篇章问题

**什么情况下不应修改**：
❌ 仅仅因为参考译文使用了不同的用词（如consider vs. take into consideration）
❌ 仅仅因为参考译文使用了不同的句式（如主动vs.被动）
❌ 仅仅为了"风格对齐"而改写已经正确的翻译
❌ 当前翻译已经准确流畅，没有明显问题

**决策流程**：
1. 首先评估：当前翻译是否准确、流畅、完整？
2. 如果是，保持原样，返回当前翻译
3. 如果否，识别具体问题，进行有针对性的最小修改

返回JSON格式：
{{
    "final_text": "最终翻译（如无问题则保持原样）",
    "integrated_references": [
        {{"reference": "参考内容", "applied": true/false, "reason": "是否应用及原因"}}
    ],
    "confidence": 0.9,
    "memory_updates": [
        {{"segment": "记忆片段", "quality": 0.9}}
    ]
}}"""
            },
            {
                "role": "user",
                "content": f"""请审查以下{source_lang}到{target_lang}的法律翻译，仅在发现明确问题时修正：

源文本：{source_text}

当前翻译：{current_translation}

【参考译文】（仅供参考，不要强制对齐）：
{self._format_references(selected_references)}

{self._format_evaluation(evaluation)}

【任务】：
1. **分析评估建议**：仔细阅读evaluation中的recommendations
2. **应用合理建议**：对于有助于提升翻译质量的建议，积极应用
3. **保持核心准确**：在改进的同时保持语义准确
4. **风格对齐参考**：适度向参考译文的风格靠拢

⚠️ 重要提醒：
- 参考译文可能来自不同翻译版本，不要为了对齐参考而改写已经正确的翻译
- 用词差异（如consider vs. take into consideration）通常不是问题
- 只有当发现明确的错误或不一致时才修改

请返回审查结果。"""
            }
        ]
        
        try:
            result = await self.call_llm_json(messages)
            
            return DiscourseTranslationResult(
                source_text=source_text,
                final_text=result.get('final_text', current_translation),
                integrated_references=result.get('integrated_references', []),
                confidence=result.get('confidence', 0.0),
                memory_updates=result.get('memory_updates', [])
            )
        except Exception as e:
            logger.error(f"Discourse translation failed: {e}")
            return DiscourseTranslationResult(
                source_text=source_text,
                final_text=current_translation,
                integrated_references=[],
                confidence=0.0,
                memory_updates=[]
            )
    
    def _format_references(self, references: List[Dict[str, Any]]) -> str:
        """格式化参考翻译"""
        if not references:
            return "无参考翻译"
        
        formatted = []
        for i, ref in enumerate(references, 1):
            formatted.append(f"{i}. {ref.get('reference', '')} (相似度: {ref.get('weight', 0.0):.2f})")
        
        return "\n".join(formatted)
    
    def _format_evaluation(self, evaluation: Any) -> str:
        """格式化差异分析结果"""
        if not evaluation:
            return "无差异分析"
        
        parts = []
        parts.append("【差异分析结果】")
        parts.append(f"总分: {evaluation.overall_score:.2f}")
        parts.append(f"- 用词一致性: {evaluation.terminology_consistency:.2f}")
        parts.append(f"- 句法一致性: {evaluation.syntax_consistency:.2f}")
        parts.append(f"- 风格一致性: {evaluation.style_consistency:.2f}")
        
        if evaluation.terminology_differences:
            parts.append("\n发现的用词差异：")
            for diff in evaluation.terminology_differences:
                parts.append(f"- {diff}")
        
        if evaluation.syntax_differences:
            parts.append("\n发现的句法差异：")
            for diff in evaluation.syntax_differences:
                parts.append(f"- {diff}")
        
        if evaluation.recommendations:
            parts.append("\n改进建议：")
            for rec in evaluation.recommendations:
                parts.append(f"- {rec}")
        
        return "\n".join(parts)
    
    def _format_suggestions(self, suggestions: List[Dict[str, Any]]) -> str:
        """格式化句法建议"""
        if not suggestions:
            return "无句法建议"
        
        formatted = []
        for suggestion in suggestions:
            formatted.append(f"- {suggestion.get('suggestion', '')}")
        
        return "\n".join(formatted)
    
    async def _generate_discourse_candidates(
        self,
        source_text: str,
        current_translation: str,
        selected_references: List[Dict[str, Any]],
        evaluation: Any,
        syntactic_suggestions: List[Dict[str, Any]],
        source_lang: str,
        target_lang: str
    ) -> List[str]:
        """生成多个篇章整合候选"""
        messages = [
            {
                "role": "system",
                "content": f"""你是一个专业的法律篇章翻译专家。你的任务是**审查当前翻译并生成{self.num_candidates}个候选，采用保守的修改策略**。

**核心原则**：
- 当前翻译已经过术语层和句法层优化，质量通常已经很好
- 仅在发现明确问题时修改，不做不必要的改写
- 参考译文仅供参考，不强制对齐风格

**候选生成策略**：
1. **候选1（最保守）**：保持当前翻译不变或做最小修改
2. **候选2（适度修正）**：修正明确的错误或不一致
3. **候选3（参考风格）**：适度参考译文风格进行调整

输出要求：请严格以json格式输出：
{{
    "candidates": [
        {{"final_text": "候选1 - 保守版本", "confidence": 0.9}},
        {{"final_text": "候选2 - 适度修正", "confidence": 0.85}},
        {{"final_text": "候选3 - 风格调整", "confidence": 0.8}}
    ]
}}
"""
            },
            {
                "role": "user",
                "content": f"""请审查以下{source_lang}到{target_lang}的法律翻译，生成{self.num_candidates}个候选（从保守到适度）：

源文本：{source_text}

当前翻译：{current_translation}

【参考译文】（仅供参考，不要强制对齐）：
{self._format_references(selected_references)}

{self._format_evaluation(evaluation)}

请返回 {self.num_candidates} 个候选（从最保守到适度修改）。"""
            }
        ]
        
        try:
            # 使用稍高的temperature增加多样性
            result = await self.call_llm_json(messages, temperature=0.4)
            
            candidates_data = result.get('candidates', [])
            candidates = [c.get('final_text', '') for c in candidates_data if c.get('final_text', '').strip()]
            
            # 过滤掉过短的候选
            candidates = [c for c in candidates if len(c.strip()) >= len(current_translation.strip()) * 0.5]
            
            # 🚪 门控机制：将原文（当前翻译）作为第一个候选
            # 让LLM选择器判断是否真的需要修改
            eval_score = evaluation.overall_score if evaluation and hasattr(evaluation, 'overall_score') else 0.0
            candidates.insert(0, current_translation)
            logger.info(f"🚪 门控：原文已加入候选列表（位置0），评估分数: {eval_score:.2f}")
            
            if len(candidates) >= self.num_candidates:
                logger.info(f"成功生成 {len(candidates)} 个篇章整合候选")
                return candidates[:self.num_candidates]
            elif candidates:
                logger.warning(f"只生成了 {len(candidates)}/{self.num_candidates} 个有效候选，补充生成")
                # 补充生成
                additional = await self._generate_candidates_by_multiple_calls(
                    source_text, current_translation, selected_references,
                    evaluation, syntactic_suggestions, source_lang, target_lang,
                    num_needed=self.num_candidates - len(candidates)
                )
                candidates.extend(additional)
                return candidates[:self.num_candidates]
            else:
                logger.warning("LLM未返回有效候选，降级为多次调用")
                return await self._generate_candidates_by_multiple_calls(
                    source_text, current_translation, selected_references,
                    evaluation, syntactic_suggestions, source_lang, target_lang,
                    num_needed=self.num_candidates
                )
                
        except Exception as e:
            logger.error(f"生成篇章候选失败: {e}，降级为多次调用")
            return await self._generate_candidates_by_multiple_calls(
                source_text, current_translation, selected_references,
                evaluation, syntactic_suggestions, source_lang, target_lang,
                num_needed=self.num_candidates
            )
    
    async def _generate_candidates_by_multiple_calls(
        self,
        source_text: str,
        current_translation: str,
        selected_references: List[Dict[str, Any]],
        evaluation: Any,
        syntactic_suggestions: List[Dict[str, Any]],
        source_lang: str,
        target_lang: str,
        num_needed: int = None
    ) -> List[str]:
        """通过多次调用LLM生成不同候选（备用方案）"""
        if num_needed is None:
            num_needed = self.num_candidates
            
        # 🚪 门控机制：原文作为第一个候选
        candidates = [current_translation]
        logger.info("🚪 门控：备用方案也将原文作为第一个候选")
        temperatures = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        for i, temp in enumerate(temperatures[:num_needed]):
            try:
                result = await self._execute_single(
                    source_text, current_translation, selected_references,
                    evaluation, syntactic_suggestions, source_lang, target_lang
                )
                final = result.final_text.strip()
                if final and len(final) >= len(current_translation.strip()) * 0.5:
                    candidates.append(final)
                    logger.debug(f"温度{temp}生成候选{i+1}: 成功")
            except Exception as e:
                logger.warning(f"温度{temp}生成候选失败: {e}")
        
        if not candidates:
            # 最后的保底：返回当前翻译
            candidates.append(current_translation)
        
        logger.info(f"通过多次调用生成了 {len(candidates)} 个候选")
        return candidates
