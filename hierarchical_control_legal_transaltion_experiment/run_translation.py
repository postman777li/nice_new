#!/usr/bin/env python3
"""
简化的本地翻译工具
直接运行翻译，无需API服务器
"""
import asyncio
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

# 导入配置
# 门控阈值配置已移除，直接使用硬编码默认值

from src.workflows.terminology import run_terminology_workflow
from src.workflows.syntax import run_syntactic_workflow
from src.workflows.discourse import run_discourse_workflow
from src.agents.baseline_translation import BaselineTranslationAgent
from src.agents.quality_assessor import QualityAssessorAgent
from src.agents.utils import TranslationControlConfig, set_global_control_config


class SimpleTranslator:
    """简化的翻译器"""
    
    def __init__(self, config: Dict[str, Any], verbose: bool = False):
        self.config = config
        self.verbose = verbose
        # 支持指定运行哪些轮次（用于消融实验）
        self.max_rounds = config.get('max_rounds', 3)  # 1, 2, 或 3
        
        # 创建统一的翻译控制配置（整合候选选择和门控）
        # 支持新参数名和旧参数名（向后兼容）
        selection_layers = config.get('selection_layers') or config.get('comet_layers') or 'none'
        num_candidates = config.get('num_candidates') or config.get('comet_candidates', 3)
        gating_layers = config.get('gating_layers', 'none')
        
        control_config = TranslationControlConfig.from_args(
            selection_layers=selection_layers,
            num_candidates=num_candidates,
            gating_layers=gating_layers,
            term_threshold=config.get('term_gate_threshold', 0.8),
            syntax_threshold=config.get('syntax_gate_threshold', 0.8),
            discourse_threshold=config.get('discourse_gate_threshold', 0.8),
            tm_threshold=config.get('tm_gate_threshold', 0.7)
        )
        
        # 设置为全局配置（供workflows使用）
        set_global_control_config(control_config)
        
        # 保留selection_config属性以保持向后兼容
        self.selection_config = control_config
        
        if verbose:
            print(f"✓ 翻译控制配置: {control_config}")
    
    async def translate(self, source: str, src_lang: str, tgt_lang: str) -> Dict[str, Any]:
        """执行翻译"""
        print(f"\n{'='*60}")
        print(f"翻译任务")
        print(f"{'='*60}")
        print(f"源语言: {src_lang}")
        print(f"目标语言: {tgt_lang}")
        print(f"源文本: {source}")
        print(f"{'='*60}\n")
        
        # 使用配置
        hierarchical = self.config.get('hierarchical', True)
        use_termbase = self.config.get('useTermBase', True)
        use_tm = self.config.get('useTM', True)
        
        result = {
            'source': source,
            'src_lang': src_lang,
            'tgt_lang': tgt_lang,
            'config': self.config,
            'trace': {}
        }
        
        try:
            if hierarchical:
                # 层次化工作流（根据max_rounds控制运行哪些轮次）
                # 轮次1: 术语层翻译
                print("🔍 轮次1: 术语层翻译...")
                r1_result = await self._run_terminology_round(
                    source, src_lang, tgt_lang, use_termbase
                )
                result['trace']['r1'] = r1_result
                result['r1_output'] = r1_result.get('output', source)
                print(f"   结果: {result['r1_output']}\n")
                result['final'] = result['r1_output']  # 默认第一轮结果
                
                # 轮次2: 句法层翻译（如果启用）
                if self.max_rounds >= 2:
                    print("🔍 轮次2: 句法层翻译...")
                    r1_target = result['r1_output']
                    r2_result = await self._run_syntax_round(
                        source, r1_target, src_lang, tgt_lang
                    )
                    result['trace']['r2'] = r2_result
                    result['r2_output'] = r2_result.get('output', result['r1_output'])
                    print(f"   结果: {result['r2_output']}\n")
                    
                    # 调试：检查r2是否改进了翻译
                    if result['r1_output'] == result['r2_output']:
                        print(f"   ⚠️  句法层未改进翻译（输出与输入相同）")
                    else:
                        print(f"   ✓ 句法层改进了翻译")
                    
                    result['final'] = result['r2_output']  # 更新为第二轮结果
                
                # 轮次3: 篇章层整合（如果启用）
                if self.max_rounds >= 3:
                    print("🔍 轮次3: 篇章层整合...")
                    r2_or_r1_output = result['final']  # 保存上一轮的输出
                    r3_result = await self._run_discourse_round(
                        result['final'], src_lang, tgt_lang, result['trace'], use_tm=use_tm
                    )
                    result['trace']['r3'] = r3_result
                    result['final'] = r3_result.get('output', result['final'])
                    print(f"   结果: {result['final']}\n")
                    
                    # 调试：检查r3是否改进了翻译
                    if r2_or_r1_output == result['final']:
                        print(f"   ⚠️  篇章层未改进翻译（输出与输入相同）")
                    else:
                        print(f"   ✓ 篇章层改进了翻译")
            else:
                # 单轮直接翻译（纯LLM基线，无任何控制策略）
                print("🔍 基线翻译（纯LLM，无控制策略）...")
                baseline_agent = BaselineTranslationAgent(locale=tgt_lang)
                baseline_result = await baseline_agent.execute({
                    'source_text': source,
                    'source_lang': src_lang,
                    'target_lang': tgt_lang
                }, None)
                
                result['trace']['baseline'] = {
                    'translated_text': baseline_result.translated_text,
                    'confidence': baseline_result.confidence
                }
                result['final'] = baseline_result.translated_text
                print(f"   结果: {result['final']}\n")
            
            result['success'] = True
            
        except Exception as e:
            print(f"❌ 翻译失败: {e}")
            import traceback
            traceback.print_exc()
            result['success'] = False
            result['error'] = str(e)
            result['final'] = source
        
        return result
    
    async def _run_terminology_round(self, text: str, src_lang: str, tgt_lang: str, use_termbase: bool) -> Dict[str, Any]:
        """术语层翻译（使用 terminology workflow）"""
        # 直接调用 terminology workflow
        result = await run_terminology_workflow(
            text=text,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            use_termbase=use_termbase,
            db_path=self.config.get('term_db', 'terms.db'),
            verbose=self.verbose,
            selection_config=self.selection_config
        )
        
        # 保存术语表供句法层使用
        if 'termTable' in result:
            self._current_term_table = result['termTable']
        
        return result
    
    async def _run_syntax_round(self, source_text: str, target_text: str, src_lang: str, tgt_lang: str) -> Dict[str, Any]:
        """句法层翻译（使用 syntax workflow）
        
        Args:
            source_text: 源文本（原始输入）
            target_text: 目标文本（第一轮翻译结果）
            src_lang: 源语言
            tgt_lang: 目标语言
        """
        # 获取术语表（用于保护）
        term_table = getattr(self, '_current_term_table', [])
        
        # 直接调用 syntax workflow
        result = await run_syntactic_workflow(
            source_text=source_text,
            target_text=target_text,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            term_table=term_table,
            verbose=self.verbose,
            selection_config=self.selection_config
        )
        
        return result
    
    async def _run_discourse_round(self, translated_text: str, src_lang: str, tgt_lang: str, trace: Dict, use_tm: bool=True) -> Dict[str, Any]:
        """篇章层整合（使用 discourse workflow）
        
        Args:
            translated_text: 第二轮翻译结果
            src_lang: 源语言
            tgt_lang: 目标语言
            trace: 之前的trace信息
            use_tm: 是否使用翻译记忆
        """
        # 获取原始源文本
        source_text = trace.get('r1', {}).get('source_text', translated_text)
        
        # 直接调用 discourse workflow
        result = await run_discourse_workflow(
            source_text=source_text,
            current_translation=translated_text,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            trace=trace,
            use_tm=use_tm,
            verbose=self.verbose,
            selection_config=self.selection_config
        )
        
        return result


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='法律文本翻译工具（本地版）')
    parser.add_argument('--source', required=True, help='源文本')
    parser.add_argument('--src-lang', default='zh', help='源语言 (默认: zh)')
    parser.add_argument('--tgt-lang', default='en', help='目标语言 (默认: en)')
    parser.add_argument('--hierarchical', action='store_true', default=True, help='使用层级翻译')
    parser.add_argument('--no-hierarchical', dest='hierarchical', action='store_false', help='不使用层级翻译')
    parser.add_argument('--use-termbase', action='store_true', default=True, help='使用术语库')
    parser.add_argument('--term-db', default='terms.db', help='术语库路径 (默认: terms.db)')
    parser.add_argument('--no-termbase', dest='use_termbase', action='store_false', help='不使用术语库')
    parser.add_argument('--use-tm', action='store_true', default=True, help='使用翻译记忆')
    parser.add_argument('--no-tm', dest='use_tm', action='store_false', help='不使用翻译记忆')
    parser.add_argument('--output', '-o', help='输出文件路径 (JSON格式)')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
    
    # LLM选择器配置
    parser.add_argument('--selection-layers', type=str, default='none',
                        help='启用LLM候选选择的层级 (可选: terminology, syntax, discourse, all, last, none; 默认: none)')
    parser.add_argument('--num-candidates', type=int, default=3,
                        help='每层生成的候选数量 (默认: 3)')
    
    # 门控参数
    parser.add_argument('--gating-layers', type=str, default='all',
                        help='启用门控的层级: none/all/terminology,syntax,discourse (默认: all)')
    parser.add_argument('--term-gate-threshold', type=float, default=0.8,
                        help='术语置信度门控阈值，低于此值的术语被过滤（默认: 0.8）')
    parser.add_argument('--syntax-gate-threshold', type=float, default=0.9,
                        help='句法评估分数门控阈值，高于此值不修改（默认: 0.9）')
    parser.add_argument('--discourse-gate-threshold', type=float, default=0.9,
                        help='篇章评估分数门控阈值，高于此值不修改（默认: 0.9）')
    parser.add_argument('--tm-gate-threshold', type=float, default=0.4,
                        help='TM相似度门控阈值，低于此值的TM被过滤（默认: 0.4）')
    
    # 质量评估参数
    parser.add_argument('--reference', help='参考译文（用于质量评估）')
    parser.add_argument('--evaluate', action='store_true', 
                        help='启用质量评估：对比翻译结果和参考译文，给出改进建议（需要提供--reference）')
    
    args = parser.parse_args()
    
    # 检查必需的环境变量
    import os
    api_key = os.getenv('OPENAI_API_KEY', '').strip()
    if not api_key:
        print("\n" + "=" * 60)
        print("❌ 错误：未设置 OPENAI_API_KEY 环境变量")
        print("=" * 60)
        print("\n请设置 API 密钥后再运行：")
        print("  export OPENAI_API_KEY='your-api-key-here'")
        print("\n或者在 .env 文件中配置：")
        print("  OPENAI_API_KEY=your-api-key-here")
        print("=" * 60 + "\n")
        return 1
    
    # 构建配置
    config = {
        'hierarchical': args.hierarchical,
        'useTermBase': args.use_termbase,
        'useTM': args.use_tm,
        'selection_layers': args.selection_layers,
        'num_candidates': args.num_candidates,
        'gating_layers': args.gating_layers,
        'term_gate_threshold': args.term_gate_threshold,
        'syntax_gate_threshold': args.syntax_gate_threshold,
        'discourse_gate_threshold': args.discourse_gate_threshold,
        'tm_gate_threshold': args.tm_gate_threshold,
        'term_db': args.term_db
    }
    
    # 显示控制机制配置（如果启用）
    if (args.selection_layers and args.selection_layers != 'none') or (args.gating_layers and args.gating_layers != 'none'):
        print(f"\n{'='*60}")
        print(f"翻译控制机制配置")
        print(f"{'='*60}")
        if args.selection_layers and args.selection_layers != 'none':
            print(f"✓ LLM候选选择: {args.selection_layers} 层级, {args.num_candidates} 个候选")
        else:
            print(f"✓ LLM候选选择: 未启用")
        
        if args.gating_layers and args.gating_layers != 'none':
            print(f"✓ 门控机制: {args.gating_layers} 层级")
            print(f"  - 术语阈值: {args.term_gate_threshold}")
            print(f"  - 句法阈值: {args.syntax_gate_threshold}")
            print(f"  - 篇章阈值: {args.discourse_gate_threshold}")
            print(f"  - TM阈值: {args.tm_gate_threshold}")
        else:
            print(f"✓ 门控机制: 未启用")
        print(f"{'='*60}\n")
    
    # 创建翻译器
    translator = SimpleTranslator(config, verbose=args.verbose)
    
    # 执行翻译
    result = await translator.translate(args.source, args.src_lang, args.tgt_lang)
    
    # 输出结果
    print(f"\n{'='*60}")
    print(f"翻译结果")
    print(f"{'='*60}")
    print(f"源文本: {result['source']}")
    print(f"译文:   {result['final']}")
    
    if args.verbose:
        print(f"\n统计信息:")
        if result['trace'].get('r1'):
            r1 = result['trace']['r1']
            print(f"  术语层: {r1.get('terms_found', 0)} 个术语, 置信度: {r1.get('confidence', 0):.2f}")
        if result['trace'].get('r2'):
            r2 = result['trace']['r2']
            print(f"  句法层: {r2.get('patterns', 0)} 个模式, 置信度: {r2.get('confidence', 0):.2f}")
        if result['trace'].get('r3'):
            r3 = result['trace']['r3']
            print(f"  篇章层: TM匹配 {r3.get('tm_matches', 0)} 个, 应用 {r3.get('tm_applied', 0)} 个, 分数: {r3.get('discourse_score', 0):.2f}")
    else:
        if result['trace'].get('r1', {}).get('terms_found', 0) > 0:
            print(f"找到术语: {result['trace']['r1']['terms_found']} 个")
    
    print(f"{'='*60}\n")
    
    # 质量评估（如果提供了参考译文且启用了评估）
    if args.evaluate and args.reference and result['success']:
        print(f"\n{'='*60}")
        print(f"📊 质量评估（对比参考译文）")
        print(f"{'='*60}")
        print(f"源文本: {result['source']}")
        print(f"译文: {result['final']}")
        print(f"参考译文: {args.reference}")
        print(f"\n正在评估...")
        
        try:
            assessor = QualityAssessorAgent(locale=args.tgt_lang)
            assessment = await assessor.execute({
                'source_text': result['source'],
                'translation': result['final'],
                'reference': args.reference,
                'source_lang': args.src_lang,
                'target_lang': args.tgt_lang
            }, None)
            
            # 显示评估结果
            print(f"\n✅ 评估完成！")
            print(f"\n{'─'*60}")
            print(f"📈 评分详情")
            print(f"{'─'*60}")
            print(f"  总体评分: {assessment.overall_score:.2%} {'⭐' * int(assessment.overall_score * 5)}")
            print(f"  - 准确性:   {assessment.accuracy_score:.2%}")
            print(f"  - 流畅性:   {assessment.fluency_score:.2%}")
            print(f"  - 术语:     {assessment.terminology_score:.2%}")
            print(f"  - 风格:     {assessment.style_score:.2%}")
            
            if assessment.strengths:
                print(f"\n{'─'*60}")
                print(f"✨ 翻译优点")
                print(f"{'─'*60}")
                for i, strength in enumerate(assessment.strengths, 1):
                    print(f"  {i}. {strength}")
            
            if assessment.weaknesses:
                print(f"\n{'─'*60}")
                print(f"⚠️  需要改进")
                print(f"{'─'*60}")
                for i, weakness in enumerate(assessment.weaknesses, 1):
                    print(f"  {i}. {weakness}")
            
            if assessment.suggestions:
                print(f"\n{'─'*60}")
                print(f"💡 改进建议")
                print(f"{'─'*60}")
                for i, suggestion in enumerate(assessment.suggestions, 1):
                    print(f"  {i}. {suggestion}")
            
            if args.verbose and assessment.detailed_comparison:
                print(f"\n{'─'*60}")
                print(f"📝 详细对比分析")
                print(f"{'─'*60}")
                print(f"  {assessment.detailed_comparison}")
            
            print(f"\n{'='*60}\n")
            
            # 将评估结果添加到结果中
            result['quality_assessment'] = {
                'overall_score': assessment.overall_score,
                'accuracy_score': assessment.accuracy_score,
                'fluency_score': assessment.fluency_score,
                'terminology_score': assessment.terminology_score,
                'style_score': assessment.style_score,
                'strengths': assessment.strengths,
                'weaknesses': assessment.weaknesses,
                'suggestions': assessment.suggestions,
                'detailed_comparison': assessment.detailed_comparison
            }
            
        except Exception as e:
            print(f"❌ 质量评估失败: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
    elif args.evaluate and not args.reference:
        print(f"\n⚠️  提示: 启用了质量评估 (--evaluate) 但未提供参考译文 (--reference)")
        print(f"    请使用: --reference '参考译文内容' 来启用质量评估\n")
    
    # 保存到文件
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"✅ 结果已保存到: {args.output}\n")
    
    return 0 if result['success'] else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

