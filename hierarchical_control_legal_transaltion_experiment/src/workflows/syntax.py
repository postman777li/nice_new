"""
句法一致性控制工作流 - 三步句法分析-修正过程
1. Bi-Extract Agent: 提取双语句法模式
2. Evaluate Agent: 评估句法翻译
3. Translation Agent: 基于评估结果生成改进翻译
"""
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from ..agents.syntax.bi_extract import BiExtractAgent
from ..agents.syntax.syntax_evaluate import SyntaxEvaluateAgent
from ..agents.syntax.syntax_translation import SyntaxTranslationAgent
from ..agents.utils import get_global_control_config

if TYPE_CHECKING:
    from ..agents.utils import TranslationControlConfig


async def run_syntactic_workflow(
    source_text: str,
    target_text: str,
    src_lang: str,
    tgt_lang: str,
    term_table: Optional[List[Dict[str, Any]]] = None,
    verbose: bool = False,
    selection_config: Optional['TranslationControlConfig'] = None
) -> Dict[str, Any]:
    """运行句法工作流：三步句法分析-修正过程
    
    Args:
        source_text: 源文本（原始输入）
        target_text: 目标文本（第一轮翻译结果）
        src_lang: 源语言
        tgt_lang: 目标语言
        term_table: 术语表（用于保护术语）
        verbose: 是否显示详细信息
    
    Returns:
        包含改进后的翻译和句法分析结果的字典
    """
    try:
        # 1) Bi-Extract Agent: 提取双语句法模式
        if verbose:
            print(f"   📝 提取句法模式...")
        
        bi_extract_agent = BiExtractAgent(locale=src_lang)
        patterns = await bi_extract_agent.execute({
            "source_text": source_text,
            "target_text": target_text,
            "source_lang": src_lang,
            "target_lang": tgt_lang
        }, None)
        
        if verbose:
            print(f"   提取到 {len(patterns)} 个句法模式")
            if patterns:
                print(f"\n   【详细】句法模式:")
                for i, pattern in enumerate(patterns[:10], 1):
                    print(f"   {i}. {pattern.source_pattern} → {pattern.target_pattern}")
                    print(f"      类型: {pattern.modality_type}, 置信度: {pattern.confidence:.2f}")
        
        # 2) Evaluate Agent: 评估句法保真度
        evaluation = None
        if patterns:
            if verbose:
                print(f"   ✓ 评估句法保真度...")
            
            evaluate_agent = SyntaxEvaluateAgent(locale=tgt_lang)
            evaluation = await evaluate_agent.execute({
                "source_text": source_text,
                "target_text": target_text,
                "patterns": patterns,
                "source_lang": src_lang,
                "target_lang": tgt_lang
            }, None)
            
            if verbose and evaluation:
                print(f"   评估分数: {evaluation.overall_score:.2f}")
                
                # 显示各维度评分
                if hasattr(evaluation, 'modality_preservation'):
                    print(f"   - 情态动词准确性: {evaluation.modality_preservation:.2f}")
                if hasattr(evaluation, 'connective_consistency'):
                    print(f"   - 连接词逻辑: {evaluation.connective_consistency:.2f}")
                if hasattr(evaluation, 'conditional_logic'):
                    print(f"   - 条件句规范性: {evaluation.conditional_logic:.2f}")
                if hasattr(evaluation, 'passive_voice_appropriateness'):
                    print(f"   - 被动语态适当性: {evaluation.passive_voice_appropriateness:.2f}")
                
                # 显示具体问题
                total_issues = 0
                if hasattr(evaluation, 'modality_issues') and evaluation.modality_issues:
                    total_issues += len(evaluation.modality_issues)
                if hasattr(evaluation, 'connective_issues') and evaluation.connective_issues:
                    total_issues += len(evaluation.connective_issues)
                if hasattr(evaluation, 'conditional_issues') and evaluation.conditional_issues:
                    total_issues += len(evaluation.conditional_issues)
                if hasattr(evaluation, 'passive_issues') and evaluation.passive_issues:
                    total_issues += len(evaluation.passive_issues)
                
                if total_issues > 0:
                    print(f"\n   【详细】发现 {total_issues} 个具体句法问题:")
                    
                    # 情态动词问题
                    if hasattr(evaluation, 'modality_issues') and evaluation.modality_issues:
                        print(f"   🔴 情态动词问题 ({len(evaluation.modality_issues)}个):")
                        for issue in evaluation.modality_issues[:2]:
                            print(f"      - {issue.get('source', '')} → {issue.get('target', '')}: {issue.get('problem', '')}")
                    
                    # 连接词问题
                    if hasattr(evaluation, 'connective_issues') and evaluation.connective_issues:
                        print(f"   🟡 连接词问题 ({len(evaluation.connective_issues)}个):")
                        for issue in evaluation.connective_issues[:2]:
                            print(f"      - {issue.get('source', '')} → {issue.get('target', '')}: {issue.get('problem', '')}")
                    
                    # 条件句问题
                    if hasattr(evaluation, 'conditional_issues') and evaluation.conditional_issues:
                        print(f"   🟠 条件句问题 ({len(evaluation.conditional_issues)}个):")
                        for issue in evaluation.conditional_issues[:2]:
                            print(f"      - {issue.get('problem', '')}")
                    
                    # 被动语态问题
                    if hasattr(evaluation, 'passive_issues') and evaluation.passive_issues:
                        print(f"   🔵 被动语态问题 ({len(evaluation.passive_issues)}个):")
                        for issue in evaluation.passive_issues[:2]:
                            print(f"      - {issue.get('problem', '')}")
                
                # 显示发现的问题（总结）
                if hasattr(evaluation, 'issues') and evaluation.issues:
                    print(f"\n   【总结】句法问题:")
                    for i, issue in enumerate(evaluation.issues[:5], 1):
                        print(f"   {i}. {issue}")
                    if len(evaluation.issues) > 5:
                        print(f"   ... 还有 {len(evaluation.issues) - 5} 个问题")
                
                # 显示改进建议
                if hasattr(evaluation, 'recommendations') and evaluation.recommendations:
                    print(f"\n   【详细】改进建议:")
                    for i, rec in enumerate(evaluation.recommendations[:3], 1):
                        print(f"   {i}. {rec}")
                    if len(evaluation.recommendations) > 3:
                        print(f"   ... 还有 {len(evaluation.recommendations) - 3} 条建议")
        
        # 🚪 门控：基于模式置信度和评估问题的细粒度门控
        control_config = get_global_control_config()
        should_skip_refinement = False
        low_confidence_patterns = []  # BiExtract中置信度<0.9的模式
        low_score_dimensions = []  # SyntaxEval中评分<0.9的维度
        
        # 步骤1：识别BiExtract中的低置信度模式（置信度<0.9）
        if patterns:
            pattern_confidence_threshold = 0.9
            low_confidence_patterns = [
                p for p in patterns
                if hasattr(p, 'confidence') and p.confidence < pattern_confidence_threshold
            ]
            
            if low_confidence_patterns and verbose:
                print(f"\n   ⚠️  BiExtract识别出 {len(low_confidence_patterns)} 个低置信度模式（< {pattern_confidence_threshold}）:")
                for i, p in enumerate(low_confidence_patterns[:3], 1):
                    print(f"      {i}. {p.source_pattern} → {p.target_pattern} (置信度: {p.confidence:.2f})")
                if len(low_confidence_patterns) > 3:
                    print(f"      ... 还有 {len(low_confidence_patterns) - 3} 个")
        
        # 步骤2：识别SyntaxEval中的低分维度和具体问题（评分<0.9）
        if evaluation:
            eval_score_threshold = 0.9
            
            if hasattr(evaluation, 'modality_preservation') and evaluation.modality_preservation < eval_score_threshold:
                low_score_dimensions.append({
                    'dimension': 'modality',
                    'score': evaluation.modality_preservation,
                    'issues': getattr(evaluation, 'modality_issues', [])
                })
            
            if hasattr(evaluation, 'connective_consistency') and evaluation.connective_consistency < eval_score_threshold:
                low_score_dimensions.append({
                    'dimension': 'connective',
                    'score': evaluation.connective_consistency,
                    'issues': getattr(evaluation, 'connective_issues', [])
                })
            
            if hasattr(evaluation, 'conditional_logic') and evaluation.conditional_logic < eval_score_threshold:
                low_score_dimensions.append({
                    'dimension': 'conditional',
                    'score': evaluation.conditional_logic,
                    'issues': getattr(evaluation, 'conditional_issues', [])
                })
            
            if hasattr(evaluation, 'passive_voice_appropriateness') and evaluation.passive_voice_appropriateness < eval_score_threshold:
                low_score_dimensions.append({
                    'dimension': 'passive_voice',
                    'score': evaluation.passive_voice_appropriateness,
                    'issues': getattr(evaluation, 'passive_issues', [])
                })
            
            if low_score_dimensions and verbose:
                print(f"\n   ⚠️  SyntaxEval识别出 {len(low_score_dimensions)} 个低分维度（< {eval_score_threshold}）:")
                for dim in low_score_dimensions:
                    print(f"      - {dim['dimension']}: {dim['score']:.2f} (问题数: {len(dim['issues'])})")
        
        # 步骤3：门控决策
        if control_config and control_config.is_gating_enabled('syntax') and evaluation:
            # 如果有低置信度模式或低分维度，需要改进
            if low_confidence_patterns or low_score_dimensions:
                if verbose:
                    print(f"\n   🚪 句法门控：发现需要改进的内容")
                    if low_confidence_patterns:
                        print(f"      - {len(low_confidence_patterns)} 个低置信度模式")
                    if low_score_dimensions:
                        print(f"      - {len(low_score_dimensions)} 个低分维度")
                    print(f"      → 将进行针对性改进")
                should_skip_refinement = False
            # 如果整体评分也高于阈值，且没有低置信度/低分问题，可以跳过
            elif not control_config.should_apply_syntax_modification(evaluation.overall_score):
                should_skip_refinement = True
                if verbose:
                    threshold = control_config.syntax_threshold
                    print(f"\n   🚪 句法门控：评估分数 {evaluation.overall_score:.2f} ≥ {threshold}")
                    print(f"      且无低置信度模式，无低分维度 → 不修改")
        
        # 如果判定可以跳过，直接返回
        if should_skip_refinement:
            return {
                "patterns": [pattern.__dict__ if hasattr(pattern, '__dict__') else pattern for pattern in patterns],
                "evaluation": evaluation.__dict__ if evaluation and hasattr(evaluation, '__dict__') else None,
                "refinedText": target_text,  # 保持原文不变
                "appliedCorrections": [],
                "ruleUpdates": [],
                "confidence": 1.0,
                "output": target_text,
                "candidates": None,
                "gated": True  # 标记为被门控保护
            }
        
        # 3) Translation Agent: 基于评估结果生成改进翻译
        if verbose:
            print(f"   🤖 句法层翻译...")
            if evaluation:
                control_config = get_global_control_config()
                gating_enabled = control_config and control_config.is_gating_enabled('syntax')
                if gating_enabled:
                    if low_confidence_patterns:
                        # 针对性改进模式
                        print(f"   - 模式: 针对 {len(low_confidence_patterns)} 个低置信度模式进行改进")
                    else:
                        # 门控启用时，能执行到这里说明分数低于阈值
                        threshold = control_config.syntax_threshold
                        print(f"   - 评估分数: {evaluation.overall_score:.2f} < 阈值 {threshold}（需要改进）")
                else:
                    # 门控未启用
                    print(f"   - 评估分数: {evaluation.overall_score:.2f}（门控未启用，执行翻译改进）")
            if patterns:
                total_patterns = len(patterns)
                focus_patterns = len(low_confidence_patterns) if low_confidence_patterns else total_patterns
                print(f"   - 参考 {focus_patterns}/{total_patterns} 个句法模式进行改进")
            if term_table:
                print(f"   - 保护 {len(term_table)} 个术语")
        
        # 检查是否为句法层启用候选选择
        generate_candidates = False
        num_candidates = 3
        if selection_config:
            generate_candidates = selection_config.is_selection_enabled('syntax')
            num_candidates = selection_config.get_num_candidates('syntax')
        
        translation_agent = SyntaxTranslationAgent(
            locale=tgt_lang,
            generate_candidates=generate_candidates,
            num_candidates=num_candidates
        )
        
        # 构建agent输入，传递低置信度模式和低分维度
        agent_input = {
            "source_text": source_text,
            "target_text": target_text,
            "patterns": patterns,
            "evaluation": evaluation,
            "source_lang": src_lang,
            "target_lang": tgt_lang,
            "term_table": term_table or [],
            "low_confidence_patterns": low_confidence_patterns,  # BiExtract的低置信度模式
            "low_score_dimensions": low_score_dimensions  # SyntaxEval的低分维度
        }
        
        # 如果有低置信度模式或低分维度，使用针对性改进模式
        if low_confidence_patterns or low_score_dimensions:
            agent_input["focus_patterns"] = low_confidence_patterns  # 保持兼容性
            agent_input["refinement_mode"] = "targeted"  # 针对性改进模式
        
        translation_result = await translation_agent.execute(agent_input, None)
        
        # 如果生成了多个候选，使用LLM选择器选择最佳
        if generate_candidates and translation_result.candidates and len(translation_result.candidates) > 1:
            if verbose:
                print(f"   🎯 LLM选择器：从{len(translation_result.candidates)}个候选中选择最佳...")
            
            from ..agents.selector import LLMSelectorAgent
            selector = LLMSelectorAgent(locale=tgt_lang)
            
            # 准备上下文（句法规则和评估信息）
            context = None
            if patterns or evaluation:
                context_lines = []
                if patterns:
                    context_lines.append(f"句法模式: {len(patterns)}个")
                if evaluation:
                    context_lines.append(f"评估分数: {getattr(evaluation, 'score', 'N/A')}")
                    if hasattr(evaluation, 'issues') and evaluation.issues:
                        context_lines.append(f"发现问题: {len(evaluation.issues)}个")
                context = "\n".join(context_lines)
            
            selector_result = await selector.execute({
                'source_text': source_text,
                'candidates': translation_result.candidates,
                'context': context,
                'layer_type': 'syntax'
            }, None)
            
            # 更新翻译结果
            translation_result.refined_text = selector_result.best_candidate
            translation_result.confidence = selector_result.confidence
            
            if verbose:
                print(f"   ✓ 选择结果: 候选#{selector_result.best_candidate_index + 1}, 置信度: {selector_result.confidence:.2f}")
                print(f"   理由: {selector_result.reasoning[:100]}...")
        
        if verbose:
            print(f"\n   【详细】句法改进:")
            print(f"   改进前: {target_text}")
            print(f"   改进后: {translation_result.refined_text}")
            print(f"   置信度: {translation_result.confidence:.2f}")

        return {
            "patterns": [pattern.__dict__ if hasattr(pattern, '__dict__') else pattern for pattern in patterns],
            "evaluation": evaluation.__dict__ if evaluation and hasattr(evaluation, '__dict__') else None,
            "refinedText": translation_result.refined_text,
            "appliedCorrections": translation_result.applied_corrections,
            "ruleUpdates": translation_result.rule_updates,
            "confidence": translation_result.confidence if hasattr(translation_result, 'confidence') else 1.0,
            "output": translation_result.refined_text,
            "candidates": translation_result.candidates if hasattr(translation_result, 'candidates') else None  # 候选翻译列表
        }
    
    except Exception as e:
        if verbose:
            print(f"   ⚠️  句法层翻译失败: {e}")
            import traceback
            traceback.print_exc()
        
        return {
            "output": target_text,
            "error": str(e)
        }