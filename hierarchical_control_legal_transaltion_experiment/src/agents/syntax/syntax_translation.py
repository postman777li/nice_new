"""
句法翻译Agent - 整合句法建议和翻译输出
支持多候选生成和COMET-Kiwi质量评估选择
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

from ..base import BaseAgent, AgentConfig, AgentRunContext

logger = logging.getLogger(__name__)


@dataclass
class SyntaxTranslationResult:
    source_text: str
    refined_text: str
    applied_corrections: List[Dict[str, Any]]
    confidence: float
    rule_updates: List[Dict[str, Any]]
    candidates: Optional[List[str]] = None  # 多个候选翻译（如果生成了）


class SyntaxTranslationAgent(BaseAgent):
    def __init__(self, locale: str = 'zh', generate_candidates: bool = False, num_candidates: int = 3):
        """
        初始化句法翻译Agent
        
        Args:
            locale: 语言地区
            generate_candidates: 是否生成多个候选翻译（默认False，只生成一个）
            num_candidates: 生成的候选数量（仅当generate_candidates=True时有效）
        """
        super().__init__(AgentConfig(
            name='syntax:syntax-translation',
            role='syntax_translator',
            domain='syntax',
            specialty='句法修正翻译',
            quality='review',
            locale=locale
        ))
        
        
        self.generate_candidates = generate_candidates
        self.num_candidates = num_candidates
        
        if self.generate_candidates:
            logger.info(f"✓ 句法翻译Agent配置为生成{num_candidates}个候选翻译")

    async def execute(self, input_data: Dict[str, Any], ctx: Optional[AgentRunContext] = None) -> SyntaxTranslationResult:
        """整合句法建议和翻译输出（支持多候选选择）"""
        source_text = input_data.get('source_text', '')
        target_text = input_data.get('target_text', '')
        patterns = input_data.get('patterns', [])
        evaluation = input_data.get('evaluation', None)
        source_lang = input_data.get('source_lang', 'zh')
        target_lang = input_data.get('target_lang', 'en')
        term_table = input_data.get('term_table', [])  # 术语表保护
        focus_patterns = input_data.get('focus_patterns', [])  # 重点关注的低置信度模式（兼容）
        low_confidence_patterns = input_data.get('low_confidence_patterns', [])  # BiExtract的低置信度模式
        low_score_dimensions = input_data.get('low_score_dimensions', [])  # SyntaxEval的低分维度
        refinement_mode = input_data.get('refinement_mode', 'full')  # full/targeted
        
        if not source_text or not target_text:
            return SyntaxTranslationResult(
                source_text=source_text,
                refined_text=target_text,
                applied_corrections=[],
                confidence=0.0,
                rule_updates=[]
            )
        
        # 根据配置选择生成方式
        if self.generate_candidates:
            candidates = await self._generate_refinement_candidates(
                source_text, target_text, patterns, evaluation,
                source_lang, target_lang, term_table, 
                low_confidence_patterns, low_score_dimensions, refinement_mode
            )
            if not candidates:
                # 如果候选生成失败，fallback到单次翻译
                logger.warning("候选生成失败，fallback到单次翻译")
                return await self._execute_single(
                    source_text, target_text, patterns, evaluation,
                    source_lang, target_lang, term_table, 
                    low_confidence_patterns, low_score_dimensions, refinement_mode
                )
            # 返回第一个候选作为refined_text，同时保留所有候选
            return SyntaxTranslationResult(
                source_text=source_text,
                refined_text=candidates[0],
                applied_corrections=[{"correction": "LLM生成候选", "applied": True}],
                confidence=0.8,  # 默认置信度，实际会由选择器Agent决定
                rule_updates=[],
                candidates=candidates
            )
        else:
            return await self._execute_single(
                source_text, target_text, patterns, evaluation,
                source_lang, target_lang, term_table, 
                low_confidence_patterns, low_score_dimensions, refinement_mode
            )
    
    async def _execute_single(
        self,
        source_text: str,
        target_text: str,
        patterns: List[Any],
        evaluation: Any,
        source_lang: str,
        target_lang: str,
        term_table: List[Dict[str, Any]],
        low_confidence_patterns: List[Any] = None,
        low_score_dimensions: List[Dict[str, Any]] = None,
        refinement_mode: str = 'full'
    ) -> SyntaxTranslationResult:
        """原有的单次改进逻辑（支持针对性改进）"""
        # 注释：移除了早期退出逻辑，强制尝试句法改进
        # 即使评估分数高或没有明确问题，也让LLM尝试改进
        
        if low_confidence_patterns is None:
            low_confidence_patterns = []
        if low_score_dimensions is None:
            low_score_dimensions = []
        
        # 构建术语保护提示
        term_protection = ""
        if term_table:
            term_pairs = [f"「{t.get('source', '')}」→「{t.get('target', '')}」" for t in term_table[:10]]
            term_protection = f"""
**术语表保护（严格遵守）**：
以下术语翻译已经验证，**绝对不可改动**：
{', '.join(term_pairs)}

修改时必须保持这些术语的原有译法。
"""
        
        # 根据refinement_mode和问题列表构建不同的指导信息
        if refinement_mode == 'targeted' and (low_confidence_patterns or low_score_dimensions):
            problem_summary = []
            if low_confidence_patterns:
                problem_summary.append(f"{len(low_confidence_patterns)} 个低置信度句法模式（< 0.9）")
            if low_score_dimensions:
                problem_summary.append(f"{len(low_score_dimensions)} 个低分维度（< 0.9）")
            
            mode_instruction = f"""
**🎯 针对性改进模式（Targeted Refinement）**

检测到需要重点改进的问题：{', '.join(problem_summary)}
其他部分质量良好，无需调整。

**优先级**：
1. 🔴 **首要任务**：重点改进下列标记的问题
2. ✅ 其他部分：保持原样（已经足够好）
3. 🔒 术语保护：绝对不改动已验证的术语

**需要重点关注的问题**：
"""
            # 添加低置信度模式详情
            if low_confidence_patterns:
                mode_instruction += f"\n📌 **BiExtract识别的低置信度模式** ({len(low_confidence_patterns)}个):\n"
                for i, p in enumerate(low_confidence_patterns[:5], 1):
                    sp = p.source_pattern if hasattr(p, 'source_pattern') else p.get('source_pattern', '')
                    tp = p.target_pattern if hasattr(p, 'target_pattern') else p.get('target_pattern', '')
                    conf = p.confidence if hasattr(p, 'confidence') else p.get('confidence', 0.0)
                    mode_instruction += f"   {i}. {sp} → {tp} (置信度: {conf:.2f}) - 需要验证改进\n"
                if len(low_confidence_patterns) > 5:
                    mode_instruction += f"   ... 还有 {len(low_confidence_patterns) - 5} 个\n"
            
            # 添加低分维度详情
            if low_score_dimensions:
                mode_instruction += f"\n📌 **SyntaxEval识别的低分维度** ({len(low_score_dimensions)}个):\n"
                for dim in low_score_dimensions:
                    dim_name = dim['dimension']
                    dim_score = dim['score']
                    dim_issues = dim['issues']
                    
                    dim_cn = {
                        'modality': '情态动词',
                        'connective': '连接词',
                        'conditional': '条件句',
                        'passive_voice': '被动语态'
                    }.get(dim_name, dim_name)
                    
                    mode_instruction += f"   - {dim_cn}: {dim_score:.2f}\n"
                    if dim_issues:
                        for issue in dim_issues[:2]:
                            problem = issue.get('problem', '')
                            if problem:
                                mode_instruction += f"      ⚠️  {problem}\n"
                        if len(dim_issues) > 2:
                            mode_instruction += f"      ... 还有 {len(dim_issues) - 2} 个问题\n"
        else:
            mode_instruction = """
**全面改进模式（Full Refinement）**

需要全面检查并改进翻译的句法表达。
"""
        
        # 使用LLM整合句法修正（应用句法模式）
        messages = [
            {
                "role": "system",
                "content": f"""你是一个专业的法律句法翻译专家。你的任务是**应用提取的句法模式来改进翻译**。

{mode_instruction}

**改进原则**：
1. **应用句法模式**：优先应用提取出的句法模式来优化表达
2. **修正句法问题**：修正评估中指出的句法不一致（issues）
3. **保持术语不变**：绝对不改动术语表中已验证的译法
4. **保持语义准确**：改进句法的同时保持原意

**可以改进的方面**：
- 根据模式调整情态动词的表达（如shall/must/should的选择）
- 根据模式优化连接词的使用（如and/or/but的表达）
- 根据模式调整条件逻辑的表达方式
- 根据模式统一句法结构的风格

{term_protection}

返回JSON格式：
{{
    "refined_text": "改进后的翻译（应用句法模式后的结果）",
    "applied_corrections": [
        {{"correction": "应用的句法改进", "applied": true}}
    ],
    "confidence": 0.9,
    "rule_updates": []
}}"""
            },
            {
                "role": "user",
                "content": f"""请对以下{source_lang}到{target_lang}的法律翻译进行句法改进：

源文本：{source_text}

当前翻译：{target_text}

{self._format_evaluation_and_patterns(patterns, evaluation, low_confidence_patterns)}

**任务**：
1. 仔细分析提取的句法模式（patterns）
{"2. 🔴 **重点关注**：优先改进上面标记的【低置信度模式】和【低分维度问题】" if (low_confidence_patterns or low_score_dimensions) else "2. 根据这些模式改进当前翻译的句法表达"}
3. 修正评估中指出的句法问题（issues）
4. 严格保护术语表中的译法

请返回改进后的翻译。"""
            }
        ]
        
        try:
            result = await self.call_llm_json(messages)
            
            refined_text = result.get('refined_text', target_text)
            
            # 安全检查：若改写后为空或过短，回退到原译文
            if not refined_text or not refined_text.strip():
                logger.warning("Refined text is empty, falling back to original")
                refined_text = target_text
            elif len(refined_text.strip()) < len(target_text.strip()) * 0.5:
                logger.warning(f"Refined text too short ({len(refined_text)} vs {len(target_text)}), falling back")
                refined_text = target_text
            
            return SyntaxTranslationResult(
                source_text=source_text,
                refined_text=refined_text,
                applied_corrections=result.get('applied_corrections', []),
                confidence=result.get('confidence', 0.0),
                rule_updates=result.get('rule_updates', [])
            )
        except Exception as e:
            logger.error(f"Syntax translation failed: {e}")
            return SyntaxTranslationResult(
                source_text=source_text,
                refined_text=target_text,
                applied_corrections=[],
                confidence=0.0,
                rule_updates=[]
            )
    
    def _format_evaluation_and_patterns(self, patterns: List[Any], evaluation: Any, focus_patterns: List[Any] = None) -> str:
        """格式化评估结果和句法模式（支持标记低置信度模式）"""
        if focus_patterns is None:
            focus_patterns = []
        
        # 构建focus_patterns的ID集合（用于快速查找）
        focus_ids = set()
        for fp in focus_patterns:
            if hasattr(fp, 'source_pattern'):
                focus_ids.add(fp.source_pattern)
            elif isinstance(fp, dict):
                focus_ids.add(fp.get('source_pattern', ''))
        
        parts = []
        
        # 1. 评估结果
        if evaluation:
            parts.append("【评估结果】")
            parts.append(f"总分: {evaluation.overall_score:.2f}")
            parts.append(f"- 情态动词保真度: {evaluation.modality_preservation:.2f}")
            parts.append(f"- 连接词一致性: {evaluation.connective_consistency:.2f}")
            parts.append(f"- 条件逻辑维护: {evaluation.conditional_logic:.2f}")
            
            if evaluation.issues:
                parts.append("\n发现的问题：")
                for issue in evaluation.issues:
                    parts.append(f"- {issue}")
            
            if evaluation.recommendations:
                parts.append("\n改进建议：")
                for rec in evaluation.recommendations:
                    parts.append(f"- {rec}")
        
        # 2. 句法模式（重点标记低置信度模式）
        if patterns:
            if focus_patterns:
                parts.append("\n【识别的句法模式】（🔴 = 低置信度，需重点改进）")
            else:
                parts.append("\n【识别的句法模式】")
                
            for pattern in patterns[:10]:  # 最多显示10个模式
                source_pattern = pattern.source_pattern if hasattr(pattern, 'source_pattern') else pattern.get('source_pattern', '')
                target_pattern = pattern.target_pattern if hasattr(pattern, 'target_pattern') else pattern.get('target_pattern', '')
                modality_type = pattern.modality_type if hasattr(pattern, 'modality_type') else pattern.get('modality_type', '')
                confidence = pattern.confidence if hasattr(pattern, 'confidence') else pattern.get('confidence', 0.0)
                
                # 检查是否是需要重点关注的模式
                is_focus = source_pattern in focus_ids
                prefix = "🔴 【低置信度】" if is_focus else ""
                
                parts.append(f"{prefix}- {source_pattern} → {target_pattern}")
                parts.append(f"  类型: {modality_type}, 置信度: {confidence:.2f}")
            
            if len(patterns) > 10:
                parts.append(f"... 还有 {len(patterns) - 10} 个模式")
        
        if not parts:
            return "无评估结果和句法模式"
        
        return "\n".join(parts)
    
    async def _generate_refinement_candidates(
        self,
        source_text: str,
        target_text: str,
        patterns: List[Any],
        evaluation: Any,
        source_lang: str,
        target_lang: str,
        term_table: List[Dict[str, Any]],
        low_confidence_patterns: List[Any] = None,
        low_score_dimensions: List[Dict[str, Any]] = None,
        refinement_mode: str = 'full'
    ) -> List[str]:
        """生成多个句法改进候选（支持针对性改进）"""
        if low_confidence_patterns is None:
            low_confidence_patterns = []
        if low_score_dimensions is None:
            low_score_dimensions = []
        # 构建术语保护提示
        term_protection = ""
        if term_table:
            term_pairs = [f"「{t.get('source', '')}」→「{t.get('target', '')}」" for t in term_table[:10]]
            term_protection = f"""
**术语表保护（严格遵守）**：
以下术语翻译已经验证，**绝对不可改动**：
{', '.join(term_pairs)}

修改时必须保持这些术语的原有译法。
"""
        
        # 根据refinement_mode构建不同的指导信息
        if refinement_mode == 'targeted' and (low_confidence_patterns or low_score_dimensions):
            problem_summary = []
            if low_confidence_patterns:
                problem_summary.append(f"{len(low_confidence_patterns)} 个低置信度句法模式（< 0.9）")
            if low_score_dimensions:
                problem_summary.append(f"{len(low_score_dimensions)} 个低分维度（< 0.9）")
            
            mode_instruction = f"""
**🎯 针对性改进模式（Targeted Refinement）**

检测到需要重点改进的问题：{', '.join(problem_summary)}

**优先级**：
1. 🔴 **首要任务**：在每个候选中，重点改进标记的问题
2. ✅ 其他部分：保持原样（已经足够好）
3. 🔒 术语保护：绝对不改动已验证的术语

**需要重点关注的问题**：
"""
            # 添加低置信度模式详情
            if low_confidence_patterns:
                mode_instruction += f"\n📌 **BiExtract识别的低置信度模式** ({len(low_confidence_patterns)}个):\n"
                for i, p in enumerate(low_confidence_patterns[:5], 1):
                    sp = p.source_pattern if hasattr(p, 'source_pattern') else p.get('source_pattern', '')
                    tp = p.target_pattern if hasattr(p, 'target_pattern') else p.get('target_pattern', '')
                    conf = p.confidence if hasattr(p, 'confidence') else p.get('confidence', 0.0)
                    mode_instruction += f"   {i}. {sp} → {tp} (置信度: {conf:.2f})\n"
            
            # 添加低分维度详情
            if low_score_dimensions:
                mode_instruction += f"\n📌 **SyntaxEval识别的低分维度** ({len(low_score_dimensions)}个):\n"
                for dim in low_score_dimensions:
                    dim_name = dim['dimension']
                    dim_score = dim['score']
                    dim_cn = {
                        'modality': '情态动词',
                        'connective': '连接词',
                        'conditional': '条件句',
                        'passive_voice': '被动语态'
                    }.get(dim_name, dim_name)
                    mode_instruction += f"   - {dim_cn}: {dim_score:.2f}\n"
            
            mode_instruction += f"""
**生成策略**：
- 候选1：保守改进（只改标记的问题）
- 候选2：适度改进（标记问题+微调相邻表达）
- 候选3：积极改进（标记问题+优化整体流畅性）
"""
        else:
            mode_instruction = f"""
**全面改进模式（Full Refinement）**

需要生成 {self.num_candidates} 个不同的改进候选，每个候选应该：
- 在保持术语不变的前提下
- 尝试不同的句法优化策略
- 尝试不同的情态动词/连接词选择
- 保持语义但表达略有差异
"""
        
        messages = [
            {
                "role": "system",
                "content": f"""你是一个专业的法律句法翻译专家。你的任务是**应用提取的句法模式来改进翻译**。

{mode_instruction}

{term_protection}

输出要求：请严格以json格式输出：
{{
    "candidates": [
        {{"refined_text": "候选1", "confidence": 0.9}},
        {{"refined_text": "候选2", "confidence": 0.85}},
        {{"refined_text": "候选3", "confidence": 0.88}}
    ]
}}
"""
            },
            {
                "role": "user",
                "content": f"""请对以下{source_lang}到{target_lang}的法律翻译进行句法改进（生成{self.num_candidates}个候选）：

源文本：{source_text}

当前翻译：{target_text}

{self._format_evaluation_and_patterns(patterns, evaluation, low_confidence_patterns)}

{"🔴 **重点任务**：针对上面标记的【低置信度模式】和【低分维度问题】进行改进" if (low_confidence_patterns or low_score_dimensions) else ""}

请返回 {self.num_candidates} 个不同的改进候选。"""
            }
        ]
        
        try:
            # 使用稍高的temperature增加多样性
            result = await self.call_llm_json(messages, temperature=0.4)
            
            candidates_data = result.get('candidates', [])
            candidates = [c.get('refined_text', '') for c in candidates_data if c.get('refined_text', '').strip()]
            
            # 过滤掉过短的候选
            candidates = [c for c in candidates if len(c.strip()) >= len(target_text.strip()) * 0.5]
            
            # 🚪 门控机制：将原文作为第一个候选
            # 让LLM选择器判断是否真的需要修改
            eval_score = evaluation.overall_score if evaluation and hasattr(evaluation, 'overall_score') else 0.0
            candidates.insert(0, target_text)
            logger.info(f"🚪 门控：原文已加入候选列表（位置0），评估分数: {eval_score:.2f}")
            
            if len(candidates) >= self.num_candidates:
                logger.info(f"成功生成 {len(candidates)} 个改进候选")
                return candidates[:self.num_candidates]
            elif candidates:
                logger.warning(f"只生成了 {len(candidates)}/{self.num_candidates} 个有效候选，补充生成")
                # 补充生成
                additional = await self._generate_candidates_by_multiple_calls(
                    source_text, target_text, patterns, evaluation,
                    source_lang, target_lang, term_table,
                    low_confidence_patterns, low_score_dimensions,
                    num_needed=self.num_candidates - len(candidates)
                )
                candidates.extend(additional)
                return candidates[:self.num_candidates]
            else:
                logger.warning("LLM未返回有效候选，降级为多次调用")
                return await self._generate_candidates_by_multiple_calls(
                    source_text, target_text, patterns, evaluation,
                    source_lang, target_lang, term_table,
                    low_confidence_patterns, low_score_dimensions,
                    num_needed=self.num_candidates
                )
                
        except Exception as e:
            logger.error(f"生成改进候选失败: {e}，降级为多次调用")
            return await self._generate_candidates_by_multiple_calls(
                source_text, target_text, patterns, evaluation,
                source_lang, target_lang, term_table,
                low_confidence_patterns, low_score_dimensions,
                num_needed=self.num_candidates
            )
    
    async def _generate_candidates_by_multiple_calls(
        self,
        source_text: str,
        target_text: str,
        patterns: List[Any],
        evaluation: Any,
        source_lang: str,
        target_lang: str,
        term_table: List[Dict[str, Any]],
        low_confidence_patterns: List[Any] = None,
        low_score_dimensions: List[Dict[str, Any]] = None,
        num_needed: int = None
    ) -> List[str]:
        """通过多次调用LLM生成不同候选（备用方案）"""
        if num_needed is None:
            num_needed = self.num_candidates
        if low_confidence_patterns is None:
            low_confidence_patterns = []
        if low_score_dimensions is None:
            low_score_dimensions = []
            
        # 🚪 门控机制：原文作为第一个候选
        candidates = [target_text]
        logger.info("🚪 门控：备用方案也将原文作为第一个候选")
        temperatures = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        for i, temp in enumerate(temperatures[:num_needed]):
            try:
                result = await self._execute_single(
                    source_text, target_text, patterns, evaluation,
                    source_lang, target_lang, term_table,
                    low_confidence_patterns, low_score_dimensions, 'full'
                )
                refined = result.refined_text.strip()
                if refined and len(refined) >= len(target_text.strip()) * 0.5:
                    candidates.append(refined)
                    logger.debug(f"温度{temp}生成候选{i+1}: 成功")
            except Exception as e:
                logger.warning(f"温度{temp}生成候选失败: {e}")
        
        if not candidates:
            # 最后的保底：返回原文本
            candidates.append(target_text)
        
        logger.info(f"通过多次调用生成了 {len(candidates)} 个候选")
        return candidates
