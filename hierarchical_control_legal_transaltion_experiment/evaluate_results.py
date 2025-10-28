#!/usr/bin/env python3
"""
实验结果评估脚本
使用现代机器翻译评估指标对实验结果进行评分
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.metrics import MetricSuite


# 指标显示优先顺序
METRIC_DISPLAY_ORDER = ['bleu', 'chrf', 'bertscore', 'bertscore_f1', 'comet', 'comet_score', 'gemba']


class ResultEvaluator:
    """实验结果评估器"""
    
    def __init__(self, metrics: List[str] = None, use_hf_mirror: bool = True, group_by: str = None, batch_size: int = 64):
        """
        Args:
            metrics: 要使用的指标列表
            use_hf_mirror: 是否使用HF镜像加速
        """
        # 默认使用快速指标
        if metrics is None:
            metrics = ['bleu', 'chrf', 'bertscore', 'comet']
        
        print(f"初始化评估指标: {', '.join(metrics)}")
        self.metric_suite = MetricSuite(
            metrics=metrics,
            lang='zh',
            use_gpu=True,  # 自动使用GPU加速（如果可用）
            use_hf_mirror=use_hf_mirror
        )
        self.metrics = metrics
        # 分组字段（来自样本 metadata 中的键）如: 'law' | 'domain' | 'year'
        self.group_by = group_by
        self.batch_size = batch_size
    
    def evaluate_sample(self, sample: Dict[str, Any]) -> Dict[str, float]:
        """
        评估单个样本
        
        Args:
            sample: 样本数据，包含 source, target, prediction
            
        Returns:
            指标分数字典
        """
        source = sample.get('source', '')
        target = sample.get('target', '')  # 参考翻译
        prediction = sample.get('prediction', '')  # 模型翻译
        
        if not target or not prediction:
            return {}
        
        try:
            scores = self.metric_suite.compute(
                source=source,
                prediction=prediction,
                reference=target
            )
            return scores
        except Exception as e:
            print(f"  ⚠️  样本 {sample.get('sample_id', 'unknown')} 评估失败: {e}")
            return {}
    
    def evaluate_results(self, results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        评估完整的实验结果
        
        Args:
            results: 实验结果字典 {ablation_name: [samples]}
            
        Returns:
            评估报告
        """
        report = {}
        
        for ablation_name, samples in results.items():
            print(f"\n{'='*60}")
            print(f"评估: {ablation_name}")
            print(f"{'='*60}")
            print(f"样本数: {len(samples)}")
            
            # 过滤成功的样本
            valid_samples = [s for s in samples if s.get('success', False) and s.get('target', '')]
            print(f"有效样本: {len(valid_samples)}/{len(samples)}")
            
            if not valid_samples:
                print("  ⚠️  没有有效样本，跳过")
                continue
            
            # 评估每个样本（优先批量路径）
            print(f"\n开始评估...")
            all_scores: List[Dict[str, float]] = []
            group_to_scores: Dict[str, List[Dict[str, float]]] = {}
            group_to_counts: Dict[str, int] = {}

            # 准备批量输入
            src_list = [s.get('source', '') for s in valid_samples]
            pred_list = [s.get('prediction', '') for s in valid_samples]
            ref_list = [s.get('target', '') for s in valid_samples]

            # 使用批量评估
            batch_scores = self.metric_suite.compute_batch(src_list, pred_list, ref_list, batch_size=self.batch_size)
            for i, scores in enumerate(batch_scores):
                if scores:
                    all_scores.append(scores)
                    if self.group_by:
                        md = valid_samples[i].get('metadata', {}) or {}
                        group_key = str(md.get(self.group_by, 'unknown'))
                        group_to_scores.setdefault(group_key, []).append(scores)
                        group_to_counts[group_key] = group_to_counts.get(group_key, 0) + 1
            
            # 计算平均分
            if all_scores:
                avg_scores = {}
                for metric in self.metrics:
                    # 处理不同的指标键名
                    possible_keys = [metric, f'{metric}_f1', f'{metric}_score']
                    for key in possible_keys:
                        values = [s.get(key, 0) for s in all_scores if key in s]
                        if values:
                            avg_scores[key] = sum(values) / len(values)
                            break
                
                # 分组平均分（如指定了 --group-by）
                grouped_avg = {}
                if self.group_by and group_to_scores:
                    for group_key, score_list in group_to_scores.items():
                        # 汇总该组内出现过的所有指标键
                        keys_in_group = set()
                        for s in score_list:
                            keys_in_group.update(s.keys())
                        group_avg = {}
                        for key in sorted(keys_in_group):
                            values = [s[key] for s in score_list if key in s]
                            if values:
                                group_avg[key] = sum(values) / len(values)
                        grouped_avg[group_key] = group_avg

                # 保存结果
                report[ablation_name] = {
                    'total_samples': len(samples),
                    'valid_samples': len(valid_samples),
                    'evaluated_samples': len(all_scores),
                    'success_rate': len(valid_samples) / len(samples) if samples else 0,
                    'avg_scores': avg_scores,
                    'all_scores': all_scores,
                    **({'grouped_avg': grouped_avg} if grouped_avg else {}),
                    **({'group_counts': group_to_counts} if self.group_by else {})
                }
                
                # 打印结果（按指定顺序）
                print(f"\n{ablation_name} 平均分数:")
                # 按顺序打印存在的指标
                printed_metrics = set()
                for metric in METRIC_DISPLAY_ORDER:
                    if metric in avg_scores:
                        print(f"  {metric:20s}: {avg_scores[metric]*100:7.2f}")
                        printed_metrics.add(metric)
                # 打印其他未在优先列表中的指标
                for metric in sorted(avg_scores.keys()):
                    if metric not in printed_metrics:
                        print(f"  {metric:20s}: {avg_scores[metric]*100:7.2f}")
                # 打印分组结果
                if self.group_by and grouped_avg:
                    print(f"\n按 {self.group_by} 分组平均分:")
                    for group_key, gavg in grouped_avg.items():
                        count = group_to_counts.get(group_key, 0)
                        print(f"  [{group_key}] (n={count})")
                        # 按顺序打印存在的指标
                        printed_metrics = set()
                        for metric in METRIC_DISPLAY_ORDER:
                            if metric in gavg:
                                print(f"    {metric:18s}: {gavg[metric]*100:7.2f}")
                                printed_metrics.add(metric)
                        # 打印其他未在优先列表中的指标
                        for metric in sorted(gavg.keys()):
                            if metric not in printed_metrics:
                                print(f"    {metric:18s}: {gavg[metric]*100:7.2f}")
            else:
                print(f"  ⚠️  所有样本评估失败")
        
        return report
    
    def save_report(self, report: Dict[str, Any], output_path: Path):
        """保存评估报告"""
        # 准备可序列化的报告
        serializable_report = {}
        for ablation, data in report.items():
            serializable_report[ablation] = {
                'total_samples': data['total_samples'],
                'valid_samples': data['valid_samples'],
                'evaluated_samples': data['evaluated_samples'],
                'success_rate': data['success_rate'],
                'avg_scores': data['avg_scores']
                # 不保存 all_scores 以减小文件大小
            }
            # 可选保存分组平均分与各组样本数
            if 'grouped_avg' in data:
                serializable_report[ablation]['grouped_avg'] = data['grouped_avg']
            if 'group_counts' in data:
                serializable_report[ablation]['group_counts'] = data['group_counts']
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_report, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 评估报告已保存到: {output_path}")
    
    def print_summary(self, report: Dict[str, Any]):
        """打印评估摘要"""
        print(f"\n{'='*80}")
        print(" "*30 + "评估摘要")
        print(f"{'='*80}\n")
        
        # 表头
        ablations = list(report.keys())
        if not ablations:
            print("没有有效的评估结果")
            return
        
        # 获取所有指标名称（按指定顺序）
        all_metrics_set = set()
        for data in report.values():
            all_metrics_set.update(data.get('avg_scores', {}).keys())
        
        # 按优先顺序排列指标
        all_metrics = []
        for metric in METRIC_DISPLAY_ORDER:
            if metric in all_metrics_set:
                all_metrics.append(metric)
                all_metrics_set.remove(metric)
        
        # 添加其他未在优先列表中的指标（按字母顺序）
        all_metrics.extend(sorted(all_metrics_set))
        
        # 打印表格
        print(f"{'消融实验':<20} {'样本数':<10} {'成功率':<10}", end='')
        for metric in all_metrics:
            print(f"{metric:<15}", end='')
        print()
        print("-" * (40 + 15 * len(all_metrics)))
        
        for ablation in ablations:
            data = report[ablation]
            print(f"{ablation:<20} ", end='')
            print(f"{data['valid_samples']}/{data['total_samples']:<6} ", end='')
            print(f"{data['success_rate']*100:6.1f}%   ", end='')
            
            for metric in all_metrics:
                score = data['avg_scores'].get(metric, 0)
                print(f"{score*100:<15.2f}", end='')
            print()
        
        print("\n" + "="*80)
        # 若包含分组结果，追加打印简要分组摘要
        any_grouped = any('grouped_avg' in data for data in report.values())
        if any_grouped:
            print("\n按组摘要（各消融实验）")
            for ablation in ablations:
                data = report[ablation]
                if 'grouped_avg' not in data:
                    continue
                print(f"\n[{ablation}] 按组平均分（仅展示存在的指标）")
                group_counts = data.get('group_counts', {})
                for group_key, gavg in data['grouped_avg'].items():
                    count = group_counts.get(group_key, 0)
                    print(f"  {group_key} (n={count})", end='')
                    # 以固定顺序输出：按总体指标顺序，否则按键名
                    keys_order = [k for k in all_metrics if k in gavg] or sorted(gavg.keys())
                    for key in keys_order:
                        print(f"  {key}:{gavg.get(key, 0)*100:.2f}", end='')
                    print()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='评估实验结果')
    parser.add_argument('result_file', type=str, help='实验结果文件路径（JSON）')
    parser.add_argument('--metrics', nargs='+', 
                       choices=['bleu', 'chrf', 'bertscore', 'comet', 'gemba'],
                       default=['bleu', 'chrf', 'bertscore', 'comet'],
                       help='要使用的评估指标（默认: bleu chrf）')
    parser.add_argument('--output', type=str, help='评估报告输出路径（可选）')
    parser.add_argument('--no-hf-mirror', action='store_true', help='禁用HF镜像加速')
    parser.add_argument('--group-by', choices=['law', 'domain', 'year'], help='按样本 metadata 字段分组统计平均分')
    parser.add_argument('--batch-size', type=int, default=64, help='评估批大小（默认64）')
    
    args = parser.parse_args()
    
    print("="*60)
    print("实验结果评估工具")
    print("="*60)
    
    # 加载实验结果
    result_file = Path(args.result_file)
    if not result_file.exists():
        print(f"❌ 错误：文件不存在: {result_file}")
        return 1
    
    print(f"\n加载实验结果: {result_file}")
    with open(result_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    print(f"  找到 {len(results)} 个消融实验")
    for ablation, samples in results.items():
        print(f"    - {ablation}: {len(samples)} 个样本")
    
    # 创建评估器
    evaluator = ResultEvaluator(
        metrics=args.metrics,
        use_hf_mirror=not args.no_hf_mirror,
        group_by=args.group_by,
        batch_size=args.batch_size
    )
    
    # 评估结果
    report = evaluator.evaluate_results(results)
    
    # 打印摘要
    evaluator.print_summary(report)
    
    # 保存报告
    if args.output:
        output_path = Path(args.output)
    else:
        # 默认保存在同目录下
        output_path = result_file.parent / f"{result_file.stem}_evaluation.json"
    
    evaluator.save_report(report, output_path)
    
    print("\n✅ 评估完成!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

