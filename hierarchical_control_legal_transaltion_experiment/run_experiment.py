#!/usr/bin/env python3
"""
批量实验运行器
"""
import asyncio
import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import sys
import os

sys.path.insert(0, str(Path(__file__).parent))

# 导入配置
# 门控阈值配置已移除，直接在命令行参数中定义默认值

# 加载 .env 文件（如果存在）
try:
    from dotenv import load_dotenv
    # 尝试从多个位置加载 .env 文件
    env_paths = [
        Path(__file__).parent / '.env',  # 项目根目录
        Path.cwd() / '.env',             # 当前工作目录
        Path.home() / '.env'             # 用户主目录
    ]
    
    env_loaded = False
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path, override=False)  # override=False 表示不覆盖已有环境变量
            print(f"✓ 已加载环境配置: {env_path}")
            env_loaded = True
            break
    
    if not env_loaded:
        # 如果没有找到 .env 文件，尝试默认加载
        load_dotenv(override=False)
except ImportError:
    # 如果没有安装 python-dotenv，给出提示但继续运行
    print("💡 提示: 安装 python-dotenv 可以自动加载 .env 文件: pip install python-dotenv")

from run_translation import SimpleTranslator
from datasets import LegalDataset, TestSample
from metrics import LegalTranslationMetrics
from src.agents.utils import TranslationControlConfig, set_global_control_config, ControlConfigPresets
from src.agents.quality_assessor import QualityAssessorAgent


class ExperimentRunner:
    """简化的实验运行器"""
    
    def __init__(self, output_dir: str = "outputs", max_concurrent: int = 10):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.metrics = LegalTranslationMetrics()
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def run_sample(self, sample: TestSample, config: Dict[str, Any], verbose: bool = False, sample_idx: int = 0, total: int = 0, save_intermediate: bool = False, enable_quality_assessment: bool = False) -> Dict[str, Any]:
        """运行单个样本（带并发控制）"""
        async with self.semaphore:  # 控制并发数
            translator = SimpleTranslator(config, verbose=verbose)
            
            try:
                if verbose:
                    print(f"[{sample_idx}/{total}] 开始翻译: {sample.id}")
                
                result = await translator.translate(
                    source=sample.source,
                    src_lang=sample.src_lang,
                    tgt_lang=sample.tgt_lang
                )
                
                # 检查翻译结果是否为空
                final_text = result['final'].strip() if result['final'] else ''
                if not final_text and result['success']:
                    # 翻译结果为空，标记为失败
                    result['success'] = False
                    result['error'] = 'Empty translation result'
                    if verbose:
                        print(f"[{sample_idx}/{total}] ⚠️  翻译结果为空: {sample.id}")
                
                # 计算指标
                if result['success'] and sample.target and final_text:
                    metrics = self.metrics.calculate_all_metrics(
                        source=sample.source,
                        target=final_text,
                        reference=sample.target,
                        src_lang=sample.src_lang,
                        tgt_lang=sample.tgt_lang,
                        term_table=result['trace'].get('r1', {}).get('termTable', [])
                    )
                else:
                    metrics = {}
                
                # 质量评估（如果启用且有参考译文）
                quality_assessment = None
                if enable_quality_assessment and result['success'] and sample.target and final_text:
                    try:
                        if verbose:
                            print(f"[{sample_idx}/{total}] 📊 进行质量评估...")
                        
                        assessor = QualityAssessorAgent(locale=sample.tgt_lang)
                        assessment = await assessor.execute({
                            'source_text': sample.source,
                            'translation': final_text,
                            'reference': sample.target,
                            'source_lang': sample.src_lang,
                            'target_lang': sample.tgt_lang
                        }, None)
                        
                        quality_assessment = {
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
                        
                        if verbose:
                            print(f"[{sample_idx}/{total}] ✓ 质量评估完成: 总分 {assessment.overall_score:.2%}")
                    
                    except Exception as e:
                        if verbose:
                            print(f"[{sample_idx}/{total}] ⚠️  质量评估失败: {e}")
                        quality_assessment = {'error': str(e)}
                
                # 提取中间层结果（如果开启save_intermediate）
                intermediate_results = {}
                if save_intermediate and result['success']:
                    trace = result.get('trace', {})
                    
                    if verbose and save_intermediate:
                        print(f"[{sample_idx}/{total}] 💾 提取中间结果: trace包含 {list(trace.keys())}")
                    
                    # Round 1: 术语层
                    if 'r1' in trace and trace['r1'].get('output'):
                        intermediate_results['round1_terminology'] = {
                            'prediction': trace['r1']['output'],
                            'terms_used': len(trace['r1'].get('termTable', [])),
                            'confidence': trace['r1'].get('confidence', 0.0)
                        }
                        if verbose:
                            print(f"  ✓ 提取了 round1_terminology")
                    elif 'r1' in trace and verbose:
                        print(f"  ⚠️  r1存在但无output")
                    
                    # Round 2: 句法层
                    if 'r2' in trace and trace['r2'].get('output'):
                        intermediate_results['round2_syntax'] = {
                            'prediction': trace['r2']['output'],
                            'confidence': trace['r2'].get('confidence', 0.0)
                        }
                        if verbose:
                            print(f"  ✓ 提取了 round2_syntax")
                    elif 'r2' in trace and verbose:
                        print(f"  ⚠️  r2存在但无output")
                    
                    # Round 3: 篇章层
                    if 'r3' in trace and trace['r3'].get('output'):
                        intermediate_results['round3_discourse'] = {
                            'prediction': trace['r3']['output'],
                            'tm_used': trace['r3'].get('tm_used', False),
                            'confidence': trace['r3'].get('confidence', 0.0)
                        }
                        if verbose:
                            print(f"  ✓ 提取了 round3_discourse")
                    elif 'r3' in trace and verbose:
                        print(f"  ⚠️  r3存在但无output")
                    
                    if intermediate_results and verbose:
                        print(f"  💾 中间结果包含: {list(intermediate_results.keys())}")
                    elif save_intermediate and not intermediate_results:
                        print(f"  ⚠️  样本 {sample.id}: save_intermediate=True 但未提取到任何中间结果")
                
                if verbose and not save_intermediate:
                    print(f"[{sample_idx}/{total}] ✓ 完成: {sample.id}")
                
                result_dict = {
                    'sample_id': sample.id,
                    'source': sample.source,
                    'target': sample.target,
                    'prediction': final_text or sample.source,  # 如果为空，返回源文本
                    'success': result['success'],
                    'metrics': metrics,
                    'trace': result['trace'],
                    'metadata': sample.metadata,
                    **(({'error': result.get('error')}) if 'error' in result else {})
                }
                
                # 添加质量评估结果
                if quality_assessment:
                    result_dict['quality_assessment'] = quality_assessment
                
                # 添加中间结果
                if intermediate_results:
                    result_dict['intermediate'] = intermediate_results
                elif save_intermediate and result['success']:
                    # 如果设置了save_intermediate但没有intermediate_results，打印警告
                    print(f"  ⚠️  警告: 样本 {sample.id} save_intermediate=True 但intermediate_results为空")
                
                return result_dict
                
            except Exception as e:
                print(f"❌ 样本 {sample.id} 失败: {e}")
                return {
                    'sample_id': sample.id,
                    'source': sample.source,
                    'target': sample.target,
                    'prediction': sample.source,
                    'success': False,
                    'error': str(e),
                    'metrics': {},
                    'metadata': sample.metadata
                }
    
    async def run_ablation(self, samples: List[TestSample], name: str, config: Dict[str, Any], verbose: bool = False, batch_mode: bool = True, save_intermediate: bool = False, enable_quality_assessment: bool = False) -> List[Dict[str, Any]]:
        """运行消融实验（支持批量并发）"""
        print(f"\n{'='*60}")
        print(f"运行消融实验: {name} - {config.get('name', name)}")
        print(f"{'='*60}")
        print(f"样本数: {len(samples)}")
        print(f"层级控制: max_rounds={config.get('max_rounds', 3)}")
        print(f"使用术语库: {config.get('useTermBase', False)}")
        print(f"并发模式: {'批量并发' if batch_mode else '逐个处理'} (最大并发: {self.max_concurrent})")
        if save_intermediate:
            print(f"💾 保存中间层结果: 是")
        if enable_quality_assessment:
            print(f"📊 质量评估: 启用")
        print()
        
        if batch_mode:
            # 批量并发处理
            print(f"🚀 启动批量并发翻译...")
            import time
            start_time = time.time()
            
            # 创建所有任务
            tasks = [
                self.run_sample(sample, config, verbose=verbose, sample_idx=i, total=len(samples), save_intermediate=save_intermediate, enable_quality_assessment=enable_quality_assessment)
                for i, sample in enumerate(samples, 1)
            ]
            
            # 并发执行所有任务
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理异常结果
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    sample = samples[i]
                    print(f"❌ 样本 {sample.id} 异常: {result}")
                    processed_results.append({
                        'sample_id': sample.id,
                        'source': sample.source,
                        'target': sample.target,
                        'prediction': sample.source,
                        'success': False,
                        'error': str(result),
                        'metrics': {},
                        'metadata': sample.metadata
                    })
                else:
                    processed_results.append(result)
            
            elapsed = time.time() - start_time
            print(f"\n✓ 批量翻译完成，耗时: {elapsed:.2f}秒")
            print(f"  平均速度: {len(samples)/elapsed:.2f} 条/秒")
            results = processed_results
        else:
            # 逐个处理（原有逻辑）
            results = []
            for i, sample in enumerate(samples, 1):
                print(f"[{i}/{len(samples)}] 处理: {sample.id}")
                result = await self.run_sample(sample, config, verbose=verbose, sample_idx=i, total=len(samples), save_intermediate=save_intermediate, enable_quality_assessment=enable_quality_assessment)
                results.append(result)
                
                if result['success']:
                    print(f"  ✅ 完成")
                else:
                    print(f"  ❌ 失败: {result.get('error', 'Unknown')}")
        
        # 统计每层修改情况（如果有trace信息）
        layer_modifications = {
            'r1_has_terms': 0,  # 术语层使用了术语
            'r1_term_count': [],  # 术语数量列表
            'r1_to_r2': 0,  # 句法层修改
            'r2_to_r3': 0,  # 篇章层修改
            'r1_to_r3': 0,  # 总体修改
            'r2_gated': 0,  # 句法层被门控
            'r3_gated': 0,  # 篇章层被门控
            'total_with_trace': 0
        }
        
        for r in results:
            if 'trace' in r and r['trace']:
                trace = r['trace']
                layer_modifications['total_with_trace'] += 1
                
                # 检查R1术语层
                if 'r1' in trace:
                    # 检查是否使用了术语
                    term_table = trace['r1'].get('termTable', [])
                    if term_table and len(term_table) > 0:
                        layer_modifications['r1_has_terms'] += 1
                        layer_modifications['r1_term_count'].append(len(term_table))
                
                # 检查R1->R2修改
                if 'r1' in trace and 'r2' in trace:
                    r1_out = trace['r1'].get('output', '')
                    r2_out = trace['r2'].get('output', '')
                    if r1_out and r2_out and r1_out != r2_out:
                        layer_modifications['r1_to_r2'] += 1
                    
                    # 检查R2是否被门控
                    if trace['r2'].get('gated', False):
                        layer_modifications['r2_gated'] += 1
                
                # 检查R2->R3修改
                if 'r2' in trace and 'r3' in trace:
                    r2_out = trace['r2'].get('output', '')
                    r3_out = trace['r3'].get('output', '')
                    if r2_out and r3_out and r2_out != r3_out:
                        layer_modifications['r2_to_r3'] += 1
                    
                    # 检查R3是否被门控
                    if trace['r3'].get('gated', False):
                        layer_modifications['r3_gated'] += 1
                
                # 检查R1->R3总体修改
                if 'r1' in trace and 'r3' in trace:
                    r1_out = trace['r1'].get('output', '')
                    r3_out = trace['r3'].get('output', '')
                    if r1_out and r3_out and r1_out != r3_out:
                        layer_modifications['r1_to_r3'] += 1
        
        # 显示修改统计
        if layer_modifications['total_with_trace'] > 0:
            total = layer_modifications['total_with_trace']
            print(f"\n📊 层级修改统计:")
            
            # 术语层统计
            if layer_modifications['r1_has_terms'] > 0:
                avg_terms = sum(layer_modifications['r1_term_count']) / len(layer_modifications['r1_term_count'])
                print(f"  术语层(R1)使用术语: {layer_modifications['r1_has_terms']}/{total} ({layer_modifications['r1_has_terms']/total*100:.1f}%)")
                print(f"    平均术语数: {avg_terms:.1f} 个")
            else:
                print(f"  术语层(R1)使用术语: 0/{total} (0.0%)")
            
            # 句法层和篇章层统计
            print(f"  句法层(R1→R2)修改: {layer_modifications['r1_to_r2']}/{total} ({layer_modifications['r1_to_r2']/total*100:.1f}%)")
            print(f"  篇章层(R2→R3)修改: {layer_modifications['r2_to_r3']}/{total} ({layer_modifications['r2_to_r3']/total*100:.1f}%)")
            print(f"  总体(R1→R3)修改: {layer_modifications['r1_to_r3']}/{total} ({layer_modifications['r1_to_r3']/total*100:.1f}%)")
            
            if layer_modifications['r2_gated'] > 0 or layer_modifications['r3_gated'] > 0:
                print(f"\n🚪 门控统计:")
                if layer_modifications['r2_gated'] > 0:
                    print(f"  句法层被门控: {layer_modifications['r2_gated']}/{total} ({layer_modifications['r2_gated']/total*100:.1f}%)")
                if layer_modifications['r3_gated'] > 0:
                    print(f"  篇章层被门控: {layer_modifications['r3_gated']}/{total} ({layer_modifications['r3_gated']/total*100:.1f}%)")
        
        # 计算平均指标
        valid_results = [r for r in results if r['success'] and r['metrics']]
        if valid_results:
            avg_metrics = {}
            metric_names = ['termbase_accuracy', 'deontic_preservation', 'conditional_logic_preservation', 'comet_score']
            for metric in metric_names:
                values = [r['metrics'].get(metric, 0) for r in valid_results if metric in r['metrics']]
                if values:
                    avg_metrics[metric] = sum(values) / len(values)
            
            print(f"\n{name} 平均指标:")
            for metric, value in avg_metrics.items():
                print(f"  {metric}: {value:.3f}")
            
            print(f"成功率: {len(valid_results)}/{len(results)} ({len(valid_results)/len(results)*100:.1f}%)")
        
        return results
    
    def _clean_for_json(self, obj, seen=None):
        """清理对象中的循环引用和不可序列化的内容"""
        if seen is None:
            seen = set()
        
        # 获取对象ID
        obj_id = id(obj)
        if obj_id in seen:
            return None  # 循环引用，返回None
        
        # 基本类型直接返回
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        
        # 列表
        if isinstance(obj, list):
            seen.add(obj_id)
            result = [self._clean_for_json(item, seen) for item in obj]
            seen.remove(obj_id)
            return result
        
        # 字典
        if isinstance(obj, dict):
            seen.add(obj_id)
            result = {}
            for key, value in obj.items():
                # 跳过一些已知的问题字段
                if key in ['_llm_client', '_db', '_tm_db', 'config']:
                    continue
                try:
                    result[key] = self._clean_for_json(value, seen)
                except:
                    result[key] = str(value)  # 序列化失败就转成字符串
            seen.remove(obj_id)
            return result
        
        # 对象（有__dict__属性）
        if hasattr(obj, '__dict__'):
            seen.add(obj_id)
            result = {}
            for key, value in obj.__dict__.items():
                if key.startswith('_'):  # 跳过私有属性
                    continue
                try:
                    result[key] = self._clean_for_json(value, seen)
                except:
                    result[key] = str(value)
            seen.remove(obj_id)
            return result
        
        # 其他类型转为字符串
        return str(obj)
    
    def save_results(self, all_results: Dict[str, List[Dict[str, Any]]]):
        """保存结果"""
        timestamp = int(time.time())
        output_file = self.output_dir / f"experiment_results_{timestamp}.json"
        
        # 清理循环引用
        cleaned_results = self._clean_for_json(all_results)
        
        # 保存完整结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 结果已保存到: {output_file}")
        
        # 如果有从full中提取的中间层结果，单独保存它们
        saved_intermediate = []
        for ablation in ['terminology', 'terminology_syntax']:
            if ablation in all_results and len(all_results[ablation]) > 0:
                # 检查是否是从full提取的（没有完整的trace字段）
                sample = all_results[ablation][0]
                if 'trace' not in sample or not sample.get('trace'):
                    # 单独保存
                    intermediate_file = self.output_dir / f"experiment_results_{timestamp}_{ablation}.json"
                    with open(intermediate_file, 'w', encoding='utf-8') as f:
                        json.dump({ablation: all_results[ablation]}, f, ensure_ascii=False, indent=2)
                    saved_intermediate.append(intermediate_file)
                    print(f"  ✅ {ablation}层结果已单独保存到: {intermediate_file}")
        
        if saved_intermediate:
            print(f"  💾 共保存了 {len(saved_intermediate)} 个中间层结果文件")
        
        # 如果有质量评估结果，单独保存
        for ablation, results in all_results.items():
            if results and len(results) > 0:
                # 检查是否有质量评估数据
                has_quality_assessment = any('quality_assessment' in r and r.get('quality_assessment') for r in results)
                if has_quality_assessment:
                    # 提取质量评估数据
                    quality_data = []
                    for r in results:
                        if 'quality_assessment' in r and r.get('quality_assessment'):
                            quality_data.append({
                                'sample_id': r.get('sample_id'),
                                'source': r.get('source'),
                                'target': r.get('target'),
                                'prediction': r.get('prediction'),
                                'quality_assessment': r['quality_assessment']
                            })
                    
                    if quality_data:
                        qa_file = self.output_dir / f"experiment_results_{timestamp}_{ablation}_quality_assessment.json"
                        with open(qa_file, 'w', encoding='utf-8') as f:
                            json.dump({
                                'ablation': ablation,
                                'total_samples': len(quality_data),
                                'samples': quality_data
                            }, f, ensure_ascii=False, indent=2)
                        print(f"  📊 {ablation}层质量评估结果已单独保存到: {qa_file}")
        
        return output_file


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='运行翻译实验')
    parser.add_argument('--samples', type=int, default=0, help='样本数量（0=全部，默认: 0使用完整测试集）')
    parser.add_argument('--ablations', nargs='+', 
                       choices=['baseline', 'terminology', 'terminology_syntax', 'full'],
                       default=['baseline', 'full'],
                       help='要运行的消融实验：baseline(纯LLM), terminology(术语), terminology_syntax(术语+句法), full(全部三层)。默认: baseline+full')
    parser.add_argument('--output-dir', default='outputs', help='输出目录')
    parser.add_argument('--verbose', action='store_true', help='显示详细输出')
    parser.add_argument('--test-set', default='dataset/processed/test_set_zh_en.json', help='测试集路径')
    parser.add_argument('--max-concurrent', type=int, default=10, help='最大并发数（默认: 10）')
    parser.add_argument('--no-batch', action='store_true', help='禁用批量并发模式，逐个处理')
    parser.add_argument('--preprocess', action='store_true', help='预处理术语：从数据集提取、去重、翻译术语并导入数据库')
    parser.add_argument('--preprocess-only', action='store_true', help='仅运行术语预处理，不执行翻译实验')
    parser.add_argument('--term-db', default='terms.db', help='术语库路径（默认: terms.db）')
    parser.add_argument('--save-intermediate', action='store_true', help='保存中间层翻译结果（术语层、句法层、篇章层），适用于full实验')
    
    # LLM候选选择参数
    parser.add_argument('--selection-layers', type=str, default='none', 
                       help='启用LLM候选选择的层级: none/last/all/discourse/terminology,syntax,discourse (默认: none)')
    parser.add_argument('--num-candidates', type=int, default=3,
                       help='生成的候选数量（默认: 3）')
    
    # 门控参数（输入级别过滤）
    parser.add_argument('--gating-layers', type=str, default='all',
                       help='启用门控的层级: none/all/terminology,syntax,discourse (默认: none)')
    parser.add_argument('--term-gate-threshold', type=float, default=0.8,
                       help='术语置信度门控阈值，低于此值的术语被过滤（默认: 0.8）')
    parser.add_argument('--syntax-gate-threshold', type=float, default=0.85,
                       help='句法评估分数门控阈值，高于此值不修改（默认: 0.85）')
    parser.add_argument('--discourse-gate-threshold', type=float, default=0.9,
                       help='篇章评估分数门控阈值，高于此值不修改（默认: 0.9）')
    parser.add_argument('--tm-gate-threshold', type=float, default=0.4,
                       help='TM相似度门控阈值，低于此值的TM被过滤（默认: 0.4）')
    
    # 质量评估参数
    parser.add_argument('--enable-quality-assessment', action='store_true', 
                       help='启用质量评估：对每个翻译结果进行详细的质量评估（需要参考译文）')
    
    args = parser.parse_args()
    
    # 检查必需的环境变量
    api_key = os.getenv('OPENAI_API_KEY', '').strip()
    if not api_key:
        print("\n" + "=" * 60)
        print("❌ 错误：未设置 OPENAI_API_KEY 环境变量")
        print("=" * 60)
        print("\n请通过以下方式之一设置 API 密钥：")
        print("\n方式1：创建 .env 文件（推荐）")
        print("  在项目根目录创建 .env 文件，内容如下：")
        print("  ─────────────────────────────────────")
        print("  OPENAI_API_KEY=your-api-key-here")
        print("  OPENAI_BASE_URL=https://api.openai.com/v1  # 可选")
        print("  OPENAI_API_MODEL=gpt-4o-mini                # 可选")
        print("  ─────────────────────────────────────")
        print("\n方式2：命令行设置（临时）")
        print("  export OPENAI_API_KEY='your-api-key-here'")
        print("\n方式3：系统环境变量（永久）")
        print("  # 添加到 ~/.bashrc 或 ~/.zshrc")
        print("  echo 'export OPENAI_API_KEY=your-api-key-here' >> ~/.bashrc")
        print("  source ~/.bashrc")
        print("\n支持的环境变量：")
        print("  OPENAI_API_KEY        - OpenAI API密钥（必需）")
        print("  OPENAI_BASE_URL       - 自定义API端点（可选，如火山引擎）")
        print("  OPENAI_API_MODEL      - 默认模型（可选，默认: gpt-4o-mini）")
        print("  LLM_TIMEOUT           - 请求超时时间（可选，秒，默认: 300）")
        print("  LLM_MAX_CONCURRENT    - 最大并发数（可选，默认: 10）")
        print("  HF_ENDPOINT           - Hugging Face镜像（可选，如: https://hf-mirror.com）")
        print("=" * 60 + "\n")
        return 1
    
    print("=" * 60)
    print("法律翻译批量实验")
    print("=" * 60)
    print(f"✓ API密钥: {api_key[:8]}...{api_key[-4:]}")
    if os.getenv('OPENAI_BASE_URL'):
        print(f"✓ API端点: {os.getenv('OPENAI_BASE_URL')}")
    print(f"✓ 默认模型: {os.getenv('OPENAI_API_MODEL', 'gpt-4o-mini')}")
    
    # 创建并设置全局翻译控制配置
    control_config = TranslationControlConfig.from_args(
        selection_layers=args.selection_layers,
        num_candidates=args.num_candidates,
        gating_layers=args.gating_layers,
        term_threshold=args.term_gate_threshold,
        syntax_threshold=args.syntax_gate_threshold,
        discourse_threshold=args.discourse_gate_threshold,
        tm_threshold=args.tm_gate_threshold
    )
    set_global_control_config(control_config)
    
    # 显示LLM候选选择配置
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
    
    # 显示质量评估配置
    if args.enable_quality_assessment:
        print(f"✓ 质量评估: 启用（将对所有翻译结果进行详细质量评估）")
    else:
        print(f"✓ 质量评估: 未启用")
    
    # 加载真实测试集
    print(f"\n加载测试数据集...")
    test_set_path = Path(args.test_set) if Path(args.test_set).is_absolute() else Path(__file__).parent / args.test_set
    
    if test_set_path.exists():
        import json
        with open(test_set_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        # 从entries字段获取数据
        entries = test_data.get('entries', test_data if isinstance(test_data, list) else [])
        metadata = test_data.get('metadata', {})
        
        print(f"  数据集信息: {metadata.get('pair', 'unknown')} - {metadata.get('total_entries', len(entries))} 条")
        print(f"  领域: {', '.join(metadata.get('domains', []))}")
        
        # 转换为TestSample对象
        all_samples = []
        for item in entries:
            sample = TestSample(
                id=str(item.get('id', len(all_samples) + 1)),
                source=item['source'],
                target=item.get('target', ''),
                src_lang='zh',  # 从文件名推断
                tgt_lang='en',
                document_id=item.get('law', 'unknown'),
                article_id=str(item.get('id', '')),
                metadata={
                    'domain': item.get('domain', ''),
                    'year': item.get('year', ''),
                    'law': item.get('law', '')
                }
            )
            all_samples.append(sample)
        
        # 根据参数选择样本数量
        if args.samples > 0 and args.samples < len(all_samples):
            samples = all_samples[:args.samples]
            print(f"  ✓ 从测试集加载了 {len(samples)}/{len(all_samples)} 个样本")
        else:
            samples = all_samples
            print(f"  ✓ 加载了完整测试集: {len(samples)} 个样本")
    else:
        # 如果测试集不存在，使用示例数据
        print(f"  ⚠️  未找到测试集: {test_set_path}")
        print(f"  使用示例数据...")
        samples = [
            TestSample(
                id=f"sample_{i}",
                source=f"这是测试样本{i}的源文本。劳动者享有平等就业的权利。",
                target=f"This is the source text of test sample {i}. Workers have the right to equal employment.",
                src_lang="zh",
                tgt_lang="en",
                document_id="test",
                article_id=str(i),
                metadata={}
            )
            for i in range(1, args.samples + 1)
        ]
        print(f"  创建了 {len(samples)} 个示例样本")
    
    # 术语预处理（可选）
    if args.preprocess or args.preprocess_only:
        from src.agents.terminology.preprocess import TerminologyPreprocessor
        
        print(f"\n{'='*60}")
        print("术语批量预处理")
        print(f"{'='*60}")
        
        # 确定术语库路径
        term_db_path = Path(args.term_db) if Path(args.term_db).is_absolute() else Path(__file__).parent / args.term_db
        
        # 创建预处理器
        preprocessor = TerminologyPreprocessor(
            src_lang='zh',
            tgt_lang='en',
            domain='law',
            db_path=str(term_db_path),
            max_concurrent=args.max_concurrent,
            batch_size=20
        )
        
        # 执行预处理
        output_file = Path(args.output_dir) / f"preprocessed_terms_{int(time.time())}.json"
        stats = await preprocessor.preprocess_dataset(
            samples=samples,
            output_file=output_file,
            verbose=True
        )
        
        # 显示统计信息
        print(f"\n{'='*60}")
        print("术语预处理统计")
        print(f"{'='*60}")
        print(f"总样本数: {stats['total_samples']}")
        print(f"提取术语数: {stats['total_extracted']}")
        print(f"去重后术语数: {stats['deduplicated']}")
        print(f"从数据库获取: {stats['from_database']}")
        print(f"新翻译术语: {stats['from_llm']}")
        print(f"导入数据库: {stats['imported_to_db']}")
        print(f"结果文件: {output_file}")
        print(f"{'='*60}\n")
        
        # 如果仅预处理，则退出
        if args.preprocess_only:
            print("✅ 术语预处理完成！")
            return 0
    
    # 消融实验配置（渐进式四种实验）
    ablation_configs = {
        'baseline': {
            'name': '基线（纯LLM）',
            'hierarchical': False,
            'useTermBase': False,
            'useTM': False,
            'max_rounds': 1
        },
        'terminology': {
            'name': '术语控制',
            'hierarchical': True,
            'useTermBase': True,
            'useTM': False,
            'max_rounds': 1  # 只运行术语层
        },
        'terminology_syntax': {
            'name': '术语+句法控制',
            'hierarchical': True,
            'useTermBase': True,
            'useTM': False,
            'max_rounds': 2  # 运行术语层和句法层
        },
        'full': {
            'name': '完整系统（术语+句法+篇章）',
            'hierarchical': True,
            'useTermBase': True,
            'useTM': True,  # 默认启用TM
            'max_rounds': 3  # 运行所有三层
        }
    }
    
    # 运行实验
    runner = ExperimentRunner(args.output_dir, max_concurrent=args.max_concurrent)
    all_results = {}
    
    batch_mode = not args.no_batch
    
    for ablation_name in args.ablations:
        if ablation_name in ablation_configs:
            config = ablation_configs[ablation_name]
            
            # 添加LLM选择器配置
            config['selection_layers'] = args.selection_layers
            config['num_candidates'] = args.num_candidates
            
            # 添加门控配置
            config['gating_layers'] = args.gating_layers
            config['term_gate_threshold'] = args.term_gate_threshold
            config['syntax_gate_threshold'] = args.syntax_gate_threshold
            config['discourse_gate_threshold'] = args.discourse_gate_threshold
            config['tm_gate_threshold'] = args.tm_gate_threshold
            
            # 对于full实验且开启save_intermediate时，保存中间结果
            save_intermediate = args.save_intermediate and ablation_name == 'full'
            results = await runner.run_ablation(
                samples, 
                ablation_name, 
                config, 
                verbose=args.verbose, 
                batch_mode=batch_mode,
                save_intermediate=save_intermediate,
                enable_quality_assessment=args.enable_quality_assessment
            )
            all_results[ablation_name] = results
            
            # 如果是full实验且保存了中间结果，自动生成terminology和terminology_syntax的结果
            if save_intermediate and ablation_name == 'full':
                print(f"\n{'='*60}")
                print("从full实验中提取中间层结果...")
                print(f"{'='*60}")
                
                # 调试：检查有多少结果包含intermediate字段
                samples_with_intermediate = sum(1 for sample in results if 'intermediate' in sample)
                print(f"📊 包含intermediate字段的样本: {samples_with_intermediate}/{len(results)}")
                
                # 提取术语层结果
                terminology_results = []
                for sample in results:
                    if 'intermediate' in sample and 'round1_terminology' in sample['intermediate']:
                        terminology_results.append({
                            'sample_id': sample['sample_id'],
                            'source': sample['source'],
                            'target': sample['target'],
                            'prediction': sample['intermediate']['round1_terminology']['prediction'],
                            'success': True,
                            'metrics': {},
                            'metadata': sample.get('metadata', {}),
                            'terms_used': sample['intermediate']['round1_terminology'].get('terms_used', 0),
                            'confidence': sample['intermediate']['round1_terminology'].get('confidence', 0.0)
                        })
                    elif 'intermediate' in sample:
                        # 调试：打印intermediate的keys
                        if not terminology_results and len(terminology_results) < 3:  # 只打印前3个
                            print(f"  ⚠️  样本 {sample['sample_id']} 有intermediate但缺少round1_terminology")
                            print(f"      intermediate keys: {list(sample['intermediate'].keys())}")
                
                if terminology_results:
                    all_results['terminology'] = terminology_results
                    print(f"✓ 提取了 {len(terminology_results)} 个术语层结果")
                else:
                    print(f"⚠️  未能提取术语层结果")
                
                # 提取术语+句法层结果
                syntax_results = []
                for sample in results:
                    if 'intermediate' in sample and 'round2_syntax' in sample['intermediate']:
                        syntax_results.append({
                            'sample_id': sample['sample_id'],
                            'source': sample['source'],
                            'target': sample['target'],
                            'prediction': sample['intermediate']['round2_syntax']['prediction'],
                            'success': True,
                            'metrics': {},
                            'metadata': sample.get('metadata', {}),
                            'confidence': sample['intermediate']['round2_syntax'].get('confidence', 0.0)
                        })
                    elif 'intermediate' in sample:
                        # 调试：打印intermediate的keys
                        if not syntax_results and len(syntax_results) < 3:  # 只打印前3个
                            print(f"  ⚠️  样本 {sample['sample_id']} 有intermediate但缺少round2_syntax")
                            print(f"      intermediate keys: {list(sample['intermediate'].keys())}")
                
                if syntax_results:
                    all_results['terminology_syntax'] = syntax_results
                    print(f"✓ 提取了 {len(syntax_results)} 个术语+句法层结果")
                else:
                    print(f"⚠️  未能提取术语+句法层结果")
                
                if terminology_results or syntax_results:
                    print(f"✓ 从1次full实验自动生成了 {1 + bool(terminology_results) + bool(syntax_results)} 个消融实验结果！")
                else:
                    print(f"❌ 未能从full实验中提取中间层结果，可能trace数据不完整")
    
    # 保存结果
    output_file = runner.save_results(all_results)
    
    print(f"\n{'='*60}")
    print("实验完成！")
    print(f"{'='*60}")
    print(f"结果文件: {output_file}")
    print()
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

