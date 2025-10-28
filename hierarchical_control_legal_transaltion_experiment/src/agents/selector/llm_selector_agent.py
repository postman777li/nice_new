#!/usr/bin/env python3
"""
LLM选择器Agent：独立的翻译候选评估和选择智能体
"""

from typing import List, Optional, Dict, Any
import logging
from dataclasses import dataclass

from ..base import BaseAgent, AgentConfig, AgentRunContext

logger = logging.getLogger(__name__)


@dataclass
class SelectorResult:
    """选择器结果"""
    best_candidate: str
    best_candidate_index: int
    confidence: float
    reasoning: str
    candidate_analysis: List[Dict[str, Any]]
    all_scores: Optional[List[float]] = None


class LLMSelectorAgent(BaseAgent):
    """LLM选择器Agent：独立的智能体，用于评估和选择最佳翻译候选"""
    
    def __init__(self, locale: str = 'zh'):
        """初始化LLM选择器Agent"""
        super().__init__(AgentConfig(
            name='selector:llm-selector',
            role='translation_selector',
            domain='quality_assessment',
            specialty='翻译候选评估与选择',
            quality='review',
            locale=locale
        ))
        logger.info("✓ LLM选择器Agent已初始化")
    
    async def execute(self, input_data: Dict[str, Any], ctx: Optional[AgentRunContext] = None) -> SelectorResult:
        """
        执行候选选择
        
        Args:
            input_data: 输入数据，包含：
                - source_text: 源文本
                - candidates: 候选翻译列表
                - context: 可选的上下文信息（术语表、句法规则等）
                - layer_type: 翻译层类型 ('terminology', 'syntax', 'discourse', 'general')
                
        Returns:
            SelectorResult: 选择结果
        """
        source_text = input_data.get('source_text', '')
        candidates = input_data.get('candidates', [])
        context = input_data.get('context')
        layer_type = input_data.get('layer_type', 'general')
        
        if not candidates:
            logger.error("未提供任何候选翻译")
            return SelectorResult(
                best_candidate="",
                best_candidate_index=0,
                confidence=0.0,
                reasoning="无候选翻译",
                candidate_analysis=[],
                all_scores=[]
            )
        
        if len(candidates) == 1:
            logger.info("只有一个候选，直接返回")
            return SelectorResult(
                best_candidate=candidates[0],
                best_candidate_index=0,
                confidence=1.0,
                reasoning="只有一个候选",
                candidate_analysis=[{"index": 1, "score": 1.0}],
                all_scores=[1.0]
            )
        
        # 调用LLM进行选择
        return await self._select_best_candidate(
            source_text=source_text,
            candidates=candidates,
            context=context,
            layer_type=layer_type
        )
    
    async def _select_best_candidate(
        self,
        source_text: str,
        candidates: List[str],
        context: Optional[str],
        layer_type: str
    ) -> SelectorResult:
        """使用LLM选择最佳候选"""
        
        messages = [
            {
                "role": "system",
                "content": self._get_system_prompt(layer_type, context)
            },
            {
                "role": "user",
                "content": self._get_user_prompt(source_text, candidates)
            }
        ]
        
        try:
            response = await self.call_llm_json(messages, temperature=0.0)
            
            best_candidate_idx = response.get("best_candidate", 1) - 1
            confidence = response.get("confidence", 0.7)
            reasoning = response.get("reasoning", "LLM未提供具体理由")
            candidate_analysis = response.get("candidate_analysis", [])
            
            # 验证索引
            if not (0 <= best_candidate_idx < len(candidates)):
                logger.warning(f"LLM返回的best_candidate_idx {best_candidate_idx}无效，使用第一个候选")
                best_candidate_idx = 0
            
            best_translation = candidates[best_candidate_idx]
            
            # 构建分数列表
            scores = [0.0] * len(candidates)
            if candidate_analysis:
                for i, analysis in enumerate(candidate_analysis):
                    if i < len(scores):
                        scores[i] = analysis.get("score", 0.0)
            else:
                # 如果没有详细分析，给选中的一个高分
                scores[best_candidate_idx] = confidence
                remaining_score = (1.0 - confidence) / (len(candidates) - 1) if len(candidates) > 1 else 0
                for i in range(len(candidates)):
                    if i != best_candidate_idx:
                        scores[i] = remaining_score
            
            logger.info(f"LLM选择器: 选择候选#{best_candidate_idx+1}/{len(candidates)} (置信度: {confidence:.4f})")
            logger.debug(f"选择理由: {reasoning}")
            if candidate_analysis:
                logger.debug(f"候选分析: {candidate_analysis}")
            
            return SelectorResult(
                best_candidate=best_translation,
                best_candidate_index=best_candidate_idx,
                confidence=float(confidence),
                reasoning=reasoning,
                candidate_analysis=candidate_analysis,
                all_scores=scores
            )
        
        except Exception as e:
            logger.error(f"LLM选择器调用失败: {e}")
            # 失败时默认选择第一个候选
            return SelectorResult(
                best_candidate=candidates[0],
                best_candidate_index=0,
                confidence=0.5,
                reasoning=f"选择失败，使用第一个候选: {str(e)}",
                candidate_analysis=[],
                all_scores=[0.5] + [0.0] * (len(candidates) - 1)
            )
    
    def _get_system_prompt(self, layer_type: str, context: Optional[str]) -> str:
        """生成系统提示词"""
        
        base_prompt = """你是一个专业的翻译质量评估专家。你的任务是评估多个翻译候选，并选择最佳的一个。

评估标准：
1. **准确性**: 是否准确传达了源文的含义
2. **流畅性**: 译文是否自然流畅
3. **专业性**: 是否符合法律文本的专业表达
4. **一致性**: 术语使用是否一致"""
        
        # 根据层级添加特定关注点
        if layer_type == 'terminology':
            base_prompt += "\n\n**当前层级**: 术语层\n**重点关注**: 术语准确性、专业术语的正确使用"
        elif layer_type == 'syntax':
            base_prompt += "\n\n**当前层级**: 句法层\n**重点关注**: 语法正确性、句子结构合理性、表达规范性"
        elif layer_type == 'discourse':
            base_prompt += "\n\n**当前层级**: 篇章层\n**重点关注**: 上下文连贯性、整体质量、风格一致性"
        
        if context:
            base_prompt += f"\n\n**参考信息**:\n{context}"
        
        base_prompt += """

请以JSON格式返回评估结果：
{
    "best_candidate": <最佳候选编号，从1开始>,
    "confidence": <置信度，0.0-1.0>,
    "reasoning": "<为什么选择这个候选的详细理由>",
    "candidate_analysis": [
        {
            "index": 1,
            "score": 0.85,
            "strengths": ["优点1", "优点2"],
            "weaknesses": ["缺点1", "缺点2"]
        },
        ...
    ]
}"""
        
        return base_prompt
    
    def _get_user_prompt(self, source_text: str, candidates: List[str]) -> str:
        """生成用户提示词"""
        
        prompt = f"""**源文本**:
{source_text}

**候选翻译**:
"""
        
        for i, candidate in enumerate(candidates, 1):
            prompt += f"\n候选{i}:\n{candidate}\n"
        
        prompt += f"\n请评估以上{len(candidates)}个候选翻译，选择最佳的一个。"
        
        return prompt


# 全局单例（可选，用于避免重复创建）
_global_selector_agent: Optional[LLMSelectorAgent] = None


def get_global_selector_agent(locale: str = 'zh') -> LLMSelectorAgent:
    """获取全局LLM选择器Agent实例"""
    global _global_selector_agent
    if _global_selector_agent is None:
        _global_selector_agent = LLMSelectorAgent(locale=locale)
        logger.info("创建全局LLM选择器Agent实例")
    return _global_selector_agent

