"""
术语工作流 - 三步术语提取-验证-翻译过程
1. Mono-Extract Agent: 识别和提取关键法律术语
2. Search + Evaluate: 从术语库检索候选翻译并评估质量
3. Translation Agent: 基于验证术语表生成初始翻译
"""
from typing import Dict, Any, Optional, TYPE_CHECKING
from ..agents.terminology.mono_extract import MonoExtractAgent
from ..agents.terminology.search import SearchAgent
from ..agents.terminology.evaluate import EvaluateAgent
from ..agents.terminology.translation import TranslationAgent
from ..agents.utils import get_global_control_config

if TYPE_CHECKING:
    from ..agents.utils import TranslationControlConfig


async def run_terminology_workflow(
    text: str,
    src_lang: str,
    tgt_lang: str,
    use_termbase: bool = True,
    db_path: str = 'terms.db',
    verbose: bool = False,
    selection_config: Optional['TranslationControlConfig'] = None
) -> Dict[str, Any]:
    """运行术语工作流：三步术语提取-验证-翻译过程
    
    Args:
        text: 源文本
        src_lang: 源语言
        tgt_lang: 目标语言
        use_termbase: 是否使用术语库
        db_path: 术语库路径
        verbose: 是否显示详细信息
    
    Returns:
        包含翻译结果和术语表的字典
    """
    try:
        # 1) Mono-Extract Agent: 识别和提取关键法律术语
        if verbose:
            print(f"   📝 提取术语...")
        
        mono_agent = MonoExtractAgent(locale=src_lang)
        extracted_terms = await mono_agent.execute({
            "text": text,
            "domain": "legal"
        }, None)
        
        if verbose:
            print(f"   提取到 {len(extracted_terms)} 个术语")
            if extracted_terms:
                print(f"\n   【详细】提取的术语:")
                for i, term in enumerate(extracted_terms[:10], 1):
                    conf = getattr(term, 'confidence', 'N/A')
                    print(f"   {i}. {term.term} (置信度: {conf})")
                if len(extracted_terms) > 10:
                    print(f"   ... 还有 {len(extracted_terms) - 10} 个术语")
        
        term_table = []
        
        # 2) Search + Evaluate: 如果使用术语库，检索并评估术语翻译
        if use_termbase and extracted_terms:
            # 2a) Search Agent: 从术语库检索候选翻译
            if verbose:
                print(f"   🔍 从术语库检索...")
            
            search_agent = SearchAgent(locale=src_lang, db_path=db_path)
            search_results = await search_agent.execute({
                "terms": [term.term for term in extracted_terms],
                "source_lang": src_lang,
                "target_lang": tgt_lang,
                "domain": ""  # 不限制domain，搜索所有法律术语
            }, None)
            
            # 构建候选术语表（包含上下文信息）
            candidate_table = [
                {
                    'source': r.term,
                    'target': r.translation,
                    'confidence': r.confidence,
                    'context': r.context  # 添加原始上下文，用于评估匹配度
                }
                for r in search_results
            ]
            
            if verbose:
                print(f"   检索到 {len(candidate_table)} 个候选翻译")
                if candidate_table:
                    print(f"\n   【详细】检索到的候选翻译:")
                    for i, cand in enumerate(candidate_table[:10], 1):
                        ctx_preview = cand.get('context', '')[:40] + '...' if cand.get('context') and len(cand.get('context', '')) > 40 else cand.get('context', '无')
                        print(f"   {i}. {cand['source']} → {cand['target']}")
                        print(f"      上下文: {ctx_preview}")
                    if len(candidate_table) > 10:
                        print(f"   ... 还有 {len(candidate_table) - 10} 个候选")
            
            # 2b) Evaluate Agent: 评估和验证翻译质量
            if candidate_table:
                if verbose:
                    print(f"   ✓ 评估术语翻译...")
                
                evaluate_agent = EvaluateAgent(locale=tgt_lang)
                evaluations = await evaluate_agent.execute({
                    "translations": candidate_table,
                    "source_text": text,
                    "source_lang": src_lang,
                    "target_lang": tgt_lang
                }, None)
                
                # 只保留验证通过的术语
                if evaluations:
                    # 首先过滤is_valid
                    valid_terms = [
                        {
                            'source': eval_result.term,
                            'target': eval_result.translation,
                            'confidence': eval_result.confidence
                        }
                        for eval_result in evaluations
                        if eval_result.is_valid
                    ]
                    if verbose:
                        print(f"   评估后保留 {len(valid_terms)} 个有效术语")
                        if valid_terms:
                            print(f"\n   【详细】验证后的术语表:")
                            for i, term in enumerate(valid_terms[:10], 1):
                                print(f"   {i}. {term['source']} → {term['target']} (置信度: {term['confidence']:.2f})")
                            if len(valid_terms) > 10:
                                print(f"   ... 还有 {len(valid_terms) - 10} 个术语")
                    
                    
                    
                    # 转回列表
                    term_table = valid_terms
                    
                    # 🚪 门控：基于置信度过滤术语
                    control_config = get_global_control_config()
                    if control_config and control_config.is_gating_enabled('terminology'):
                        threshold = control_config.terminology_threshold
                        before_count = len(term_table)
                        term_table = [
                            term for term in term_table
                            if term['confidence'] >= threshold
                        ]
                        filtered_count = before_count - len(term_table)
                        if verbose and filtered_count > 0:
                            print(f"   🚪 术语门控：过滤了 {filtered_count} 个低置信度术语（阈值: {threshold}）")
                     
                    if verbose:
                        print(f"   过滤后保留 {len(term_table)} 个有效术语")
                        if term_table:
                            print(f"\n   【详细】过滤后的术语表:")
                            for i, term in enumerate(term_table[:10], 1):
                                print(f"   {i}. {term['source']} → {term['target']} (置信度: {term['confidence']:.2f})")
                            if len(term_table) > 10:
                                print(f"   ... 还有 {len(term_table) - 10} 个术语")
                else:
                    term_table = candidate_table  # 如果评估失败，保留候选表
        
        # 3) Translation Agent: 基于验证术语表生成初始翻译
        if verbose:
            print(f"   🤖 LLM翻译...")
        
        # 检查是否为术语层启用候选选择
        generate_candidates = False
        num_candidates = 3
        if selection_config:
            generate_candidates = selection_config.is_selection_enabled('terminology')
            num_candidates = selection_config.get_num_candidates('terminology')
        
        translation_agent = TranslationAgent(
            locale=tgt_lang,
            generate_candidates=generate_candidates,
            num_candidates=num_candidates
        )
        translation_result = await translation_agent.execute({
            "source_text": text,
            "term_table": term_table,
            "source_lang": src_lang,
            "target_lang": tgt_lang
        }, None)
        
        # 如果生成了多个候选，使用LLM选择器选择最佳
        if generate_candidates and translation_result.candidates and len(translation_result.candidates) > 1:
            if verbose:
                print(f"   🎯 LLM选择器：从{len(translation_result.candidates)}个候选中选择最佳...")
            
            from ..agents.selector import LLMSelectorAgent
            selector = LLMSelectorAgent(locale=tgt_lang)
            
            # 准备上下文（术语表信息）
            context = None
            if term_table:
                context_lines = ["术语表:"]
                for term in term_table[:10]:  # 最多显示10个术语
                    context_lines.append(f"  {term.get('source', '')} → {term.get('target', '')}")
                context = "\n".join(context_lines)
            
            selector_result = await selector.execute({
                'source_text': text,
                'candidates': translation_result.candidates,
                'context': context,
                'layer_type': 'terminology'
            }, None)
            
            # 更新翻译结果
            translation_result.translated_text = selector_result.best_candidate
            translation_result.confidence = selector_result.confidence
            
            if verbose:
                print(f"   ✓ 选择结果: 候选#{selector_result.best_candidate_index + 1}, 置信度: {selector_result.confidence:.2f}")
                print(f"   理由: {selector_result.reasoning[:100]}...")
        
        if verbose:
            print(f"\n   【详细】翻译结果:")
            print(f"   源文: {text}")
            print(f"   译文: {translation_result.translated_text}")
            print(f"   置信度: {translation_result.confidence:.2f}\n")

        return {
            "extractedTerms": [term.__dict__ if hasattr(term, '__dict__') else term for term in extracted_terms],
            "termTable": term_table,
            "translatedText": translation_result.translated_text,
            "confidence": translation_result.confidence,
            "output": translation_result.translated_text,
            "source_text": text,  # 保存源文本供后续使用
            "candidates": translation_result.candidates if hasattr(translation_result, 'candidates') else None,  # 候选翻译列表
            "terms_found": len(term_table)  # 用于统计显示
        }
    
    except Exception as e:
        if verbose:
            print(f"   ⚠️  术语层翻译失败: {e}")
            import traceback
            traceback.print_exc()
        
        return {
            "output": text,
            "termTable": [],
            "terms_found": 0,
            "error": str(e)
        }