"""
基于LLM的候选翻译选择器
使用LLM评估和选择最佳翻译候选，无需加载额外的评估模型
"""
from typing import List, Dict, Any, Optional, Tuple
import logging
import asyncio

logger = logging.getLogger(__name__)


class LLMSelector:
    """使用LLM评估并选择最佳翻译候选"""
    
    def __init__(self):
        """初始化LLM选择器"""
        pass
    
    async def select_best_candidate(
        self,
        source_text: str,
        candidates: List[str],
        context: Optional[str] = None,
        layer_type: str = "general",
        return_reasoning: bool = True
    ) -> Tuple[str, float, Optional[Dict]]:
        """
        从多个候选翻译中选择最佳的一个
        
        Args:
            source_text: 源文本
            candidates: 候选翻译列表
            context: 可选上下文信息（如术语表、句法规则等）
            layer_type: 层级类型 ('terminology', 'syntax', 'discourse', 'general')
            return_reasoning: 是否返回选择理由
            
        Returns:
            (最佳候选, 置信度分数0-1, 理由字典或None)
        """
        if not candidates:
            logger.error("候选列表为空")
            return "", 0.0, None
        
        if len(candidates) == 1:
            logger.debug("只有一个候选，直接返回")
            return candidates[0], 1.0, {"reasoning": "只有一个候选"}
        
        # 构建评估prompt
        evaluation_prompt = self._build_evaluation_prompt(
            source_text, candidates, context, layer_type
        )
        
        try:
            # 调用LLM进行评估
            from ..base import BaseAgent
            temp_agent = BaseAgent()
            
            result = await temp_agent.call_llm_json(
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个专业的翻译质量评估专家，擅长评估法律翻译的准确性、流畅性和专业性。"
                    },
                    {
                        "role": "user",
                        "content": evaluation_prompt
                    }
                ],
                temperature=0.3  # 使用较低temperature确保稳定输出
            )
            
            # 解析结果
            best_idx = result.get("best_candidate", 1) - 1  # 转换为0-based索引
            confidence = result.get("confidence", 0.8)
            reasoning = result.get("reasoning", "")
            candidate_analysis = result.get("candidate_analysis", [])
            
            # 验证索引有效性
            if 0 <= best_idx < len(candidates):
                best_translation = candidates[best_idx]
                logger.info(f"LLM选择器: 从{len(candidates)}个候选中选择#{best_idx+1} (置信度: {confidence:.2f})")
                if reasoning:
                    logger.debug(f"选择理由: {reasoning[:100]}...")
            else:
                logger.warning(f"LLM返回的索引{best_idx+1}无效，使用第一个候选")
                best_translation = candidates[0]
                confidence = 0.7
            
            # 构建理由字典
            reasoning_dict = None
            if return_reasoning:
                reasoning_dict = {
                    "best_candidate_index": best_idx + 1,
                    "confidence": confidence,
                    "reasoning": reasoning,
                    "candidate_analysis": candidate_analysis,
                    "all_candidates": candidates
                }
            
            return best_translation, float(confidence), reasoning_dict
            
        except Exception as e:
            logger.error(f"LLM选择失败: {e}，使用第一个候选")
            return candidates[0], 0.7, {"error": str(e)} if return_reasoning else None
    
    async def select_best_candidate_async(
        self,
        source_text: str,
        candidates: List[str],
        context: Optional[str] = None,
        layer_type: str = "general",
        return_scores: bool = False
    ) -> Tuple[str, float, Optional[List[float]]]:
        """
        异步版本，兼容COMET选择器的接口
        
        Args:
            source_text: 源文本
            candidates: 候选翻译列表
            context: 可选上下文信息
            layer_type: 层级类型
            return_scores: 是否返回所有候选的分数（用于兼容）
            
        Returns:
            (最佳候选, 置信度分数, 所有分数列表或None)
        """
        best_translation, confidence, reasoning = await self.select_best_candidate(
            source_text, candidates, context, layer_type, return_reasoning=True
        )
        
        # 如果需要返回所有分数，构造一个简化的分数列表
        all_scores = None
        if return_scores and reasoning:
            best_idx = reasoning.get("best_candidate_index", 1) - 1
            # 生成一个简单的分数分布：最佳候选得高分，其他候选得略低分
            all_scores = [0.7] * len(candidates)
            if 0 <= best_idx < len(candidates):
                all_scores[best_idx] = confidence
        
        return best_translation, confidence, all_scores
    
    def _build_evaluation_prompt(
        self,
        source_text: str,
        candidates: List[str],
        context: Optional[str],
        layer_type: str
    ) -> str:
        """构建评估prompt"""
        
        # 根据层级类型定制评估标准
        if layer_type == "terminology":
            evaluation_criteria = """
评估标准（按优先级排序）：
1. **术语准确性** (最重要)：是否正确使用了术语表中的翻译
2. **术语一致性**：同一术语在不同位置是否翻译一致
3. **流畅性**：翻译是否自然流畅
4. **完整性**：是否完整翻译了所有内容
"""
        elif layer_type == "syntax":
            evaluation_criteria = """
评估标准（按优先级排序）：
1. **语法正确性** (最重要)：是否符合目标语言的语法规范
2. **句式规范性**：是否符合法律文本的句式要求
3. **术语一致性**：术语翻译是否保持一致
4. **流畅性**：句子是否通顺易读
"""
        elif layer_type == "discourse":
            evaluation_criteria = """
评估标准（按优先级排序）：
1. **篇章连贯性** (最重要)：上下文是否连贯，逻辑是否清晰
2. **术语一致性**：整个篇章的术语翻译是否统一
3. **语篇风格**：是否符合法律文本的正式风格
4. **整体质量**：翻译的整体专业度和可读性
"""
        else:
            evaluation_criteria = """
评估标准（按优先级排序）：
1. **准确性** (最重要)：是否准确传达了原文意思
2. **流畅性**：翻译是否自然流畅
3. **专业性**：是否符合专业领域的表达习惯
4. **完整性**：是否完整翻译了所有内容
"""
        
        # 构建候选列表
        candidates_text = ""
        for i, candidate in enumerate(candidates, 1):
            candidates_text += f"\n候选 {i}:\n{candidate}\n"
        
        # 构建上下文信息
        context_text = ""
        if context:
            context_text = f"\n【参考信息】\n{context}\n"
        
        prompt = f"""请评估以下翻译候选的质量，选出最佳的一个。

【源文本】
{source_text}

【候选翻译】{candidates_text}
{context_text}
{evaluation_criteria}

**评估任务**：
1. 逐个分析每个候选的优缺点
2. 根据评估标准选择最佳候选
3. 给出选择理由和置信度评分

**输出要求**：严格按照以下JSON格式输出：
{{
    "best_candidate": 1,  // 最佳候选的编号（1、2、3等）
    "confidence": 0.9,    // 置信度评分（0.0-1.0）
    "reasoning": "选择候选X的理由：...",  // 简要说明选择理由（1-2句话）
    "candidate_analysis": [
        {{"candidate": 1, "strengths": ["优点1", "优点2"], "weaknesses": ["缺点1"]}},
        {{"candidate": 2, "strengths": ["优点1"], "weaknesses": ["缺点1", "缺点2"]}},
        ...
    ]
}}

请仔细评估后输出JSON："""
        
        return prompt


# 为了保持向后兼容，提供一个获取选择器的函数
def get_llm_selector() -> LLMSelector:
    """获取LLM选择器实例（无需单例，每次创建新实例）"""
    return LLMSelector()

