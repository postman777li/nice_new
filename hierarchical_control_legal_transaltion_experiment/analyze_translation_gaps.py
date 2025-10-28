#!/usr/bin/env python3
"""
翻译差异深度分析工具
分析多层次翻译系统与人类译文的差异，找出改进方向
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict, Counter
import re

sys.path.insert(0, str(Path(__file__).parent))


class TranslationGapAnalyzer:
    """翻译差异分析器"""
    
    def __init__(self, result_file: Path):
        self.result_file = result_file
        self.results = self._load_results()
        
    def _load_results(self) -> Dict:
        """加载结果文件"""
        with open(self.result_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def analyze_all(self, ablation: str = None, top_n: int = 20) -> Dict[str, Any]:
        """全面分析"""
        print("=" * 80)
        print("翻译差异深度分析报告")
        print("=" * 80)
        print(f"\n结果文件: {self.result_file}")
        
        # 选择要分析的消融实验
        if ablation is None:
            # 优先选择full，否则选第一个
            if 'full' in self.results:
                ablation = 'full'
            else:
                ablation = list(self.results.keys())[0]
        
        if ablation not in self.results:
            print(f"\n❌ 错误: 找不到消融实验 '{ablation}'")
            print(f"可用的实验: {', '.join(self.results.keys())}")
            return {}
        
        samples = self.results[ablation]
        print(f"\n分析消融实验: {ablation}")
        print(f"样本数: {len(samples)}")
        
        # 1. 术语使用差异分析
        print(f"\n{'='*80}")
        print("1. 术语使用差异分析")
        print(f"{'='*80}")
        self._analyze_terminology_gaps(samples, top_n)
        
        # 2. 句法结构差异分析
        print(f"\n{'='*80}")
        print("2. 句法结构差异分析")
        print(f"{'='*80}")
        self._analyze_syntactic_gaps(samples, top_n)
        
        # 3. 长度与复杂度差异
        print(f"\n{'='*80}")
        print("3. 长度与复杂度差异")
        print(f"{'='*80}")
        self._analyze_length_complexity(samples)
        
        # 4. 低分样本案例分析
        print(f"\n{'='*80}")
        print("4. 低分样本案例分析 (COMET < 0.3)")
        print(f"{'='*80}")
        self._analyze_low_score_cases(samples, top_n=10)
        
        # 5. 高分样本参考
        print(f"\n{'='*80}")
        print("5. 高分样本参考 (COMET > 0.7)")
        print(f"{'='*80}")
        self._analyze_high_score_cases(samples, top_n=5)
        
        # 6. 具体改进建议
        print(f"\n{'='*80}")
        print("6. 具体改进建议")
        print(f"{'='*80}")
        self._generate_improvement_suggestions(samples)
        
        return {}
    
    def _analyze_terminology_gaps(self, samples: List[Dict], top_n: int):
        """分析术语使用差异"""
        # 提取人类译文和模型译文中的专业术语（大写开头的词组）
        human_terms = Counter()
        model_terms = Counter()
        term_mismatches = []
        
        for sample in samples:
            target = sample.get('target', '')
            prediction = sample.get('prediction', '')
            
            if not target or not prediction:
                continue
            
            # 提取专业术语（连续的大写词）
            human_t = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', target)
            model_t = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', prediction)
            
            human_terms.update(human_t)
            model_terms.update(model_t)
            
            # 找出术语不一致的情况
            h_set = set(human_t)
            m_set = set(model_t)
            if h_set != m_set:
                diff = h_set.symmetric_difference(m_set)
                if diff and len(target) < 200:  # 只记录短句子
                    term_mismatches.append({
                        'sample_id': sample.get('sample_id'),
                        'human_only': h_set - m_set,
                        'model_only': m_set - h_set,
                        'source': sample.get('source', '')[:80],
                        'comet': sample.get('metrics', {}).get('comet_score', 0)
                    })
        
        print(f"\n人类译文高频术语 (Top {top_n}):")
        for term, count in human_terms.most_common(top_n):
            model_count = model_terms.get(term, 0)
            diff = count - model_count
            status = "✓" if diff == 0 else f"△({diff:+d})"
            print(f"  {status} {term:30s}: 人类{count:3d}次, 模型{model_count:3d}次")
        
        print(f"\n模型独有高频术语 (Top {top_n}):")
        model_only = [(t, c) for t, c in model_terms.items() if t not in human_terms]
        for term, count in sorted(model_only, key=lambda x: x[1], reverse=True)[:top_n]:
            print(f"  - {term:30s}: {count:3d}次")
        
        print(f"\n术语不一致案例 (前5个):")
        for i, case in enumerate(term_mismatches[:5], 1):
            print(f"\n  案例 {i} (sample_id={case['sample_id']}, COMET={case['comet']:.2f}):")
            print(f"    源文: {case['source']}...")
            if case['human_only']:
                print(f"    人类独有: {', '.join(case['human_only'])}")
            if case['model_only']:
                print(f"    模型独有: {', '.join(case['model_only'])}")
    
    def _analyze_syntactic_gaps(self, samples: List[Dict], top_n: int):
        """分析句法结构差异"""
        # 分析情态动词、被动语态、连接词等
        modals_human = Counter()
        modals_model = Counter()
        
        passives_human = 0
        passives_model = 0
        
        connectives_human = Counter()
        connectives_model = Counter()
        
        modal_verbs = ['shall', 'should', 'must', 'may', 'can', 'will', 'would', 'could', 'might']
        connectives = ['and', 'or', 'but', 'however', 'therefore', 'thus', 'unless', 'if', 'when', 'where']
        
        for sample in samples:
            target = sample.get('target', '').lower()
            prediction = sample.get('prediction', '').lower()
            
            if not target or not prediction:
                continue
            
            # 情态动词统计
            for modal in modal_verbs:
                modals_human[modal] += target.count(f' {modal} ')
                modals_model[modal] += prediction.count(f' {modal} ')
            
            # 被动语态（简单识别：be + past participle）
            passives_human += len(re.findall(r'\b(?:is|are|was|were|be|been|being)\s+\w+ed\b', target))
            passives_model += len(re.findall(r'\b(?:is|are|was|were|be|been|being)\s+\w+ed\b', prediction))
            
            # 连接词统计
            for conn in connectives:
                connectives_human[conn] += target.count(f' {conn} ')
                connectives_model[conn] += prediction.count(f' {conn} ')
        
        print(f"\n情态动词使用对比:")
        for modal in sorted(modal_verbs, key=lambda x: modals_human[x], reverse=True):
            h_count = modals_human[modal]
            m_count = modals_model[modal]
            if h_count > 0 or m_count > 0:
                diff = m_count - h_count
                status = "✓" if abs(diff) <= h_count * 0.1 else "△"
                print(f"  {status} {modal:10s}: 人类{h_count:4d}次, 模型{m_count:4d}次 ({diff:+4d})")
        
        print(f"\n被动语态使用:")
        print(f"  人类: {passives_human:4d}次")
        print(f"  模型: {passives_model:4d}次 ({passives_model-passives_human:+4d})")
        
        print(f"\n连接词使用对比 (Top 10):")
        for conn in sorted(connectives, key=lambda x: connectives_human[x], reverse=True)[:10]:
            h_count = connectives_human[conn]
            m_count = connectives_model[conn]
            if h_count > 0 or m_count > 0:
                diff = m_count - h_count
                print(f"  {conn:10s}: 人类{h_count:4d}次, 模型{m_count:4d}次 ({diff:+4d})")
    
    def _analyze_length_complexity(self, samples: List[Dict]):
        """分析长度与复杂度差异"""
        human_lengths = []
        model_lengths = []
        human_words = []
        model_words = []
        
        for sample in samples:
            target = sample.get('target', '')
            prediction = sample.get('prediction', '')
            
            if not target or not prediction:
                continue
            
            human_lengths.append(len(target))
            model_lengths.append(len(prediction))
            
            human_words.append(len(target.split()))
            model_words.append(len(prediction.split()))
        
        if not human_lengths:
            print("  无有效样本")
            return
        
        print(f"\n字符长度统计:")
        print(f"  人类译文: 平均 {sum(human_lengths)/len(human_lengths):.1f}, "
              f"中位数 {sorted(human_lengths)[len(human_lengths)//2]:.1f}")
        print(f"  模型译文: 平均 {sum(model_lengths)/len(model_lengths):.1f}, "
              f"中位数 {sorted(model_lengths)[len(model_lengths)//2]:.1f}")
        print(f"  差异: {(sum(model_lengths)-sum(human_lengths))/len(human_lengths):.1f} "
              f"({(sum(model_lengths)/sum(human_lengths)-1)*100:+.1f}%)")
        
        print(f"\n词数统计:")
        print(f"  人类译文: 平均 {sum(human_words)/len(human_words):.1f} 词")
        print(f"  模型译文: 平均 {sum(model_words)/len(model_words):.1f} 词")
        print(f"  差异: {(sum(model_words)-sum(human_words))/len(human_words):.1f} 词 "
              f"({(sum(model_words)/sum(human_words)-1)*100:+.1f}%)")
    
    def _analyze_low_score_cases(self, samples: List[Dict], top_n: int):
        """分析低分案例"""
        # 按COMET分数排序
        scored_samples = [
            s for s in samples 
            if s.get('metrics', {}).get('comet_score') is not None
            and s.get('prediction', '').strip()
        ]
        scored_samples.sort(key=lambda x: x.get('metrics', {}).get('comet_score', 0))
        
        print(f"\n显示最低分的 {top_n} 个样本:\n")
        
        for i, sample in enumerate(scored_samples[:top_n], 1):
            comet = sample.get('metrics', {}).get('comet_score', 0)
            print(f"案例 {i} (sample_id={sample.get('sample_id')}, COMET={comet:.3f})")
            print(f"  源文: {sample.get('source', '')}")
            print(f"  人类: {sample.get('target', '')}")
            print(f"  模型: {sample.get('prediction', '')}")
            
            # 简单差异分析
            target_words = set(sample.get('target', '').lower().split())
            pred_words = set(sample.get('prediction', '').lower().split())
            
            missing = target_words - pred_words
            extra = pred_words - target_words
            
            if missing:
                print(f"  缺失词: {', '.join(list(missing)[:10])}")
            if extra:
                print(f"  多余词: {', '.join(list(extra)[:10])}")
            print()
    
    def _analyze_high_score_cases(self, samples: List[Dict], top_n: int):
        """分析高分案例（作为参考）"""
        scored_samples = [
            s for s in samples 
            if s.get('metrics', {}).get('comet_score') is not None
            and s.get('prediction', '').strip()
        ]
        scored_samples.sort(key=lambda x: x.get('metrics', {}).get('comet_score', 0), reverse=True)
        
        print(f"\n显示最高分的 {top_n} 个样本（参考学习）:\n")
        
        for i, sample in enumerate(scored_samples[:top_n], 1):
            comet = sample.get('metrics', {}).get('comet_score', 0)
            print(f"案例 {i} (sample_id={sample.get('sample_id')}, COMET={comet:.3f})")
            print(f"  源文: {sample.get('source', '')}")
            print(f"  人类: {sample.get('target', '')}")
            print(f"  模型: {sample.get('prediction', '')}")
            print()
    
    def _generate_improvement_suggestions(self, samples: List[Dict]):
        """生成改进建议"""
        suggestions = []
        
        # 统计各层效果
        term_accuracy = []
        syntax_scores = []
        discourse_scores = []
        
        for sample in samples:
            trace = sample.get('trace', {})
            metrics = sample.get('metrics', {})
            
            # 术语准确性
            if 'r1' in trace and trace['r1'].get('terms_found', 0) > 0:
                term_acc = metrics.get('termbase_accuracy', 0)
                term_accuracy.append(term_acc)
            
            # 句法保真度
            if 'r2' in trace:
                deontic = metrics.get('deontic_preservation', 0)
                syntax_scores.append(deontic)
            
            # 篇章一致性
            if 'r3' in trace:
                discourse_score = trace['r3'].get('coherence', 0)
                if discourse_score > 0:
                    discourse_scores.append(discourse_score)
        
        print("\n基于数据的改进建议:\n")
        
        # 术语层
        if term_accuracy:
            avg_term = sum(term_accuracy) / len(term_accuracy)
            print(f"1. 术语层 (平均准确度: {avg_term:.3f})")
            if avg_term < 0.3:
                suggestions.append("   建议: 术语库覆盖率较低，需要扩充术语库或改进术语提取算法")
            else:
                suggestions.append("   状态: 术语层效果较好")
        
        # 句法层
        if syntax_scores:
            avg_syntax = sum(syntax_scores) / len(syntax_scores)
            print(f"\n2. 句法层 (平均情态保真: {avg_syntax:.3f})")
            if avg_syntax < 0.8:
                suggestions.append("   建议: 情态动词保真度不足，需要加强句法模式识别与应用")
            else:
                suggestions.append("   状态: 句法层效果良好")
        
        # 篇章层
        if discourse_scores:
            avg_discourse = sum(discourse_scores) / len(discourse_scores)
            print(f"\n3. 篇章层 (平均一致性: {avg_discourse:.3f})")
            if avg_discourse < 0.5:
                suggestions.append("   建议: 翻译记忆库需要扩充，或调整相似度匹配阈值")
            else:
                suggestions.append("   状态: 篇章层效果可接受")
        
        # 通用建议
        print(f"\n4. 通用改进方向:")
        suggestions.extend([
            "   - 增加更多法律领域的训练数据到术语库和TM",
            "   - 考虑引入后编辑模块，针对低分样本进行二次优化",
            "   - 对比分析高分和低分样本，提取成功模式",
            "   - 建立人工反馈循环，持续优化系统"
        ])
        
        for s in suggestions:
            print(s)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='翻译差异深度分析')
    parser.add_argument('result_file', type=str, help='实验结果文件路径（JSON）')
    parser.add_argument('--ablation', type=str, help='要分析的消融实验名称（默认: full）')
    parser.add_argument('--top-n', type=int, default=20, help='显示Top N（默认: 20）')
    
    args = parser.parse_args()
    
    result_file = Path(args.result_file)
    if not result_file.exists():
        print(f"❌ 错误：文件不存在: {result_file}")
        return 1
    
    analyzer = TranslationGapAnalyzer(result_file)
    analyzer.analyze_all(ablation=args.ablation, top_n=args.top_n)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

