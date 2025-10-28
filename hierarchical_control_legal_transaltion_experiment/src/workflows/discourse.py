"""
篇章一致性工作流 - 三步篇章分析-综合过程
1. Query Agent: 使用混合检索识别相关翻译记忆
2. Evaluate Agent: 分析当前翻译与参考的差异（用词、句法、风格）
3. Translation Agent: 基于差异分析结果调整翻译
"""
from typing import Dict, Any, Optional, TYPE_CHECKING
from ..agents.discourse.discourse_query import DiscourseQueryAgent
from ..agents.discourse.discourse_evaluate import DiscourseEvaluateAgent
from ..agents.discourse.discourse_translation import DiscourseTranslationAgent
from ..agents.utils import get_global_control_config

if TYPE_CHECKING:
    from ..agents.utils import TranslationControlConfig


async def run_discourse_workflow(
    source_text: str,
    current_translation: str,
    src_lang: str,
    tgt_lang: str,
    trace: Optional[Dict[str, Any]] = None,
    use_tm: bool = True,
    verbose: bool = False,
    selection_config: Optional['TranslationControlConfig'] = None
) -> Dict[str, Any]:
    """运行篇章工作流：三步篇章分析-综合过程
    
    Args:
        source_text: 源文本
        current_translation: 当前翻译（来自前一轮）
        src_lang: 源语言
        tgt_lang: 目标语言
        trace: 前面轮次的trace信息（可选）
        use_tm: 是否使用翻译记忆（默认True）
        verbose: 是否显示详细信息
    
    Returns:
        包含最终翻译和篇章分析结果的字典
    """
    try:
        # 初始化TM相关变量
        tm_results = []
        evaluation = None
        top_references = []
        
        # 1) Query Agent: 使用混合检索识别相关翻译记忆（可选）
        if use_tm:
            if verbose:
                print(f"   🔍 检索翻译记忆...")
            
            query_agent = DiscourseQueryAgent(locale=tgt_lang)
            tm_results = await query_agent.execute({
                "text": source_text,  # 使用源文本检索
                "source_lang": src_lang,
                "target_lang": tgt_lang,
                "top_k": 5
            }, None)
            
            if verbose:
                if tm_results:
                    print(f"   检索到 {len(tm_results)} 个相关翻译记忆")
                    print(f"\n   【详细】前5个最佳匹配:")
                    for i, tm in enumerate(tm_results[:5], 1):
                        print(f"   {i}. 相似度: {tm.similarity_score:.2f}")
                        print(f"      源: {tm.source_text[:50]}...")
                        print(f"      译: {tm.target_text[:50]}...")
                else:
                    print(f"   未找到相关翻译记忆")
        else:
            if verbose:
                print(f"   ⚠️ 翻译记忆已禁用，跳过TM检索")
        
        # 2) Evaluate Agent: 分析当前翻译与前3个参考的差异
        if use_tm and tm_results:
            # 选择前3个最佳匹配作为参考
            top_references = [
                {
                    'source_text': tm.source_text,
                    'target_text': tm.target_text,
                    'similarity_score': tm.similarity_score,
                    'context': getattr(tm, 'context', ''),
                    'legal_domain': getattr(tm, 'legal_domain', '')
                }
                for tm in tm_results[:3]
            ]
            
            if verbose:
                print(f"   ✓ 评估与参考译文的差异...")
                # 显示使用的TM参考
                print(f"\n   【详细】使用的TM参考 ({len(top_references)} 条):")
                for i, ref in enumerate(top_references[:3], 1):
                    similarity = ref.get('similarity_score', 0.0)  # 使用正确的字段名
                    print(f"   {i}. 相似度: {similarity:.2f}")
                    print(f"      源: {ref.get('source_text', '')[:60]}...")  # 使用正确的字段名
                    print(f"      译: {ref.get('target_text', '')[:60]}...")  # 使用正确的字段名
                if len(top_references) > 3:
                    print(f"   ... 还有 {len(top_references) - 3} 条参考")
            
            evaluate_agent = DiscourseEvaluateAgent(locale=tgt_lang)
            evaluation = await evaluate_agent.execute({
                "source_text": source_text,
                "text": current_translation,
                "references": top_references,
                "target_lang": tgt_lang
            }, None)
            
            if verbose and evaluation:
                print(f"\n   【详细】篇章评估结果:")
                print(f"   - 总体评估分数: {evaluation.overall_score:.2f}")
                
                # 显示各维度评分
                if hasattr(evaluation, 'coherence_score'):
                    print(f"   - 连贯性: {evaluation.coherence_score:.2f}")
                if hasattr(evaluation, 'style_consistency'):
                    print(f"   - 风格一致性: {evaluation.style_consistency:.2f}")
                if hasattr(evaluation, 'terminology_consistency'):
                    print(f"   - 术语一致性: {evaluation.terminology_consistency:.2f}")
                
                # 显示改进建议
                if hasattr(evaluation, 'suggestions') and evaluation.suggestions:
                    print(f"\n   【详细】改进建议 ({len(evaluation.suggestions)} 条):")
                    for i, suggestion in enumerate(evaluation.suggestions[:3], 1):
                        print(f"   {i}. {suggestion}")
                    if len(evaluation.suggestions) > 3:
                        print(f"   ... 还有 {len(evaluation.suggestions) - 3} 条建议")
        
        # 🚪 门控：检查是否需要应用篇章修改（基于评估分数或TM质量）
        control_config = get_global_control_config()
        if control_config and control_config.is_gating_enabled('discourse'):
            # 如果评估分数高，或者没有有效的TM参考，可能不需要修改
            should_modify = True
            
            # 检查评估分数
            if evaluation and hasattr(evaluation, 'overall_score'):
                if not control_config.should_apply_discourse_modification(evaluation.overall_score):
                    should_modify = False
                    if verbose:
                        threshold = control_config.discourse_threshold
                        print(f"   🚪 篇章门控：评估分数 {evaluation.overall_score:.2f} ≥ {threshold}，翻译质量良好，不修改")
            
            # 检查是否有有效的TM参考
            if should_modify and not top_references:
                should_modify = False
                if verbose:
                    print(f"   🚪 篇章门控：无有效TM参考，不修改")
            
            if not should_modify:
                return {
                    "tm_results": [tm.__dict__ if hasattr(tm, '__dict__') else tm for tm in tm_results] if tm_results else [],
                    "evaluation": evaluation.__dict__ if evaluation and hasattr(evaluation, '__dict__') else None,
                    "finalText": current_translation,  # 保持原文不变
                    "appliedPatches": [],
                    "trace": trace or {},
                    "confidence": 1.0,
                    "output": current_translation,
                    "candidates": None,
                    "gated": True  # 标记为被门控保护
                }
        
        # 3) Translation Agent: 基于差异分析结果调整翻译
        if verbose:
            print(f"\n   🤖 篇章层翻译整合...")
            if evaluation:
                score = getattr(evaluation, 'overall_score', 0)
                control_config = get_global_control_config()
                # 使用全局配置或默认阈值
                default_threshold = 0.8
                threshold = control_config.discourse_threshold if control_config else default_threshold
                
                # 根据门控状态显示不同信息
                gating_enabled = control_config and control_config.is_gating_enabled('discourse')
                if gating_enabled:
                    # 门控启用时，能执行到这里说明分数低于阈值
                    print(f"   - 评估分数: {score:.2f} < 阈值 {threshold}（需要改进）")
                else:
                    # 门控未启用，只显示分数
                    print(f"   - 评估分数: {score:.2f}（门控未启用，执行翻译整合）")
            
            if top_references:
                print(f"   - 参考 {len(top_references)} 个TM记忆进行风格整合")
                avg_sim = sum(ref.get('similarity_score', 0) for ref in top_references) / len(top_references)  # 使用正确的字段名
                print(f"   - 平均相似度: {avg_sim:.2f}")
        
        # 检查是否为篇章层启用候选选择
        generate_candidates = False
        num_candidates = 3
        if selection_config:
            generate_candidates = selection_config.is_selection_enabled('discourse')
            num_candidates = selection_config.get_num_candidates('discourse')
        
        translation_agent = DiscourseTranslationAgent(
            locale=tgt_lang,
            generate_candidates=generate_candidates,
            num_candidates=num_candidates
        )
        
        # 准备选中的参考（使用分析过的前3个）
        selected_references = []
        if tm_results:
            # 🚪 门控：基于相似度过滤TM参考
            control_config = get_global_control_config()
            filtered_tms = tm_results[:3]  # 默认使用前3个
            
            if control_config and control_config.is_gating_enabled('discourse'):
                tm_threshold = control_config.tm_similarity_threshold
                before_count = len(tm_results[:3])
                filtered_tms = [
                    tm for tm in tm_results[:3]
                    if tm.similarity_score >= tm_threshold
                ]
                filtered_count = before_count - len(filtered_tms)
                if verbose and filtered_count > 0:
                    print(f"   🚪 TM门控：过滤了 {filtered_count} 个低相似度TM（阈值: {tm_threshold}）")
            
            for tm in filtered_tms:
                selected_references.append({
                    'reference': f"{tm.source_text} → {tm.target_text}",
                    'weight': tm.similarity_score,
                    'source': tm.source_text,
                    'target': tm.target_text
                })
            
            if verbose and selected_references:
                print(f"   使用 {len(selected_references)} 个筛选后的TM参考")
        
        translation_result = await translation_agent.execute({
            "source_text": source_text,
            "current_translation": current_translation,
            "selected_references": selected_references,
            "evaluation": evaluation,
            "syntactic_suggestions": [],  # 可从上一轮获取
            "source_lang": src_lang,
            "target_lang": tgt_lang
        }, None)
        
        if verbose:
            if hasattr(translation_result, 'refined_translation') and translation_result.refined_translation:
                print(f"\n   【详细】篇章层翻译结果:")
                print(f"   - 输入: {current_translation[:80]}...")
                print(f"   - 输出: {translation_result.refined_translation[:80]}...")
                if current_translation == translation_result.refined_translation:
                    print(f"   - 状态: 未修改（保持原翻译）")
                else:
                    print(f"   - 状态: 已优化")
        
        # 如果生成了多个候选，使用LLM选择器选择最佳
        if generate_candidates and translation_result.candidates and len(translation_result.candidates) > 1:
            if verbose:
                print(f"   🎯 LLM选择器：从{len(translation_result.candidates)}个候选中选择最佳...")
            
            from ..agents.selector import LLMSelectorAgent
            selector = LLMSelectorAgent(locale=tgt_lang)
            
            # 准备上下文（TM参考和评估信息）
            context = None
            if selected_references or evaluation:
                context_lines = []
                if selected_references:
                    context_lines.append(f"参考译文: {len(selected_references)}条")
                    for ref in selected_references[:2]:
                        context_lines.append(f"  - {ref.get('target', '')[:50]}...")
                if evaluation:
                    context_lines.append(f"评估: {getattr(evaluation, 'overall_score', 'N/A')}")
                context = "\n".join(context_lines)
            
            selector_result = await selector.execute({
                'source_text': source_text,
                'candidates': translation_result.candidates,
                'context': context,
                'layer_type': 'discourse'
            }, None)
            
            # 更新翻译结果
            translation_result.final_text = selector_result.best_candidate
            translation_result.confidence = selector_result.confidence
            
            if verbose:
                print(f"   ✓ 选择结果: 候选#{selector_result.best_candidate_index + 1}, 置信度: {selector_result.confidence:.2f}")
                print(f"   理由: {selector_result.reasoning[:100]}...")
        
        if verbose:
            print(f"\n   【详细】篇章整合:")
            print(f"   整合前: {current_translation}")
            print(f"   整合后: {translation_result.final_text}")

        return {
            "references": [
                {
                    'source_text': tm.source_text,
                    'target_text': tm.target_text,
                    'similarity_score': tm.similarity_score
                } for tm in tm_results[:3]
            ] if tm_results else [],
            "evaluation": evaluation.__dict__ if evaluation and hasattr(evaluation, '__dict__') else None,
            "finalText": translation_result.final_text,
            "integratedReferences": translation_result.integrated_references,
            "memoryUpdates": translation_result.memory_updates,
            # 兼容旧字段名
            "tm_used": len(tm_results) > 0 if tm_results else False,
            "tm_matches": len(tm_results) if tm_results else 0,
            "tm_applied": len(translation_result.integrated_references) if translation_result.integrated_references else 0,
            "coherence": getattr(evaluation, 'overall_score', 0.0) if evaluation else 0.0,
            "discourse_score": translation_result.confidence if hasattr(translation_result, 'confidence') else 0.0,
            "confidence": getattr(evaluation, 'overall_score', 1.0) if evaluation else 1.0,
            "output": translation_result.final_text,
            "candidates": translation_result.candidates if hasattr(translation_result, 'candidates') else None  # 候选翻译列表
        }
    
    except Exception as e:
        if verbose:
            print(f"   ⚠️  篇章层翻译失败: {e}")
            import traceback
            traceback.print_exc()
        
        return {
            "output": current_translation,
            "error": str(e)
        }