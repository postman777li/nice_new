"""
实验结果分析
"""
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any
from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report


class ExperimentAnalyzer:
    """实验结果分析器"""
    
    def __init__(self, results_file: str):
        self.results_file = results_file
        self.results = self._load_results()
        self.df = self._create_dataframe()
    
    def _load_results(self) -> Dict[str, List[Dict[str, Any]]]:
        """加载实验结果"""
        with open(self.results_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _create_dataframe(self) -> pd.DataFrame:
        """创建分析用的DataFrame"""
        rows = []
        
        for ablation_name, ablation_results in self.results.items():
            for result in ablation_results:
                if 'error' in result:
                    continue
                
                row = {
                    'ablation': ablation_name,
                    'sample_id': result['sample_id'],
                    'src_lang': result['src_lang'],
                    'tgt_lang': result['tgt_lang'],
                    'document_id': result['metadata'].get('document_id', ''),
                    'length': result['metadata'].get('length', 0)
                }
                
                # 添加指标
                if 'metrics' in result:
                    for metric, value in result['metrics'].items():
                        row[metric] = value
                
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def calculate_summary_statistics(self) -> Dict[str, Any]:
        """计算汇总统计"""
        summary = {}
        
        for ablation in self.df['ablation'].unique():
            ablation_df = self.df[self.df['ablation'] == ablation]
            
            ablation_stats = {
                'count': len(ablation_df),
                'metrics': {}
            }
            
            # 计算各指标的平均值和标准差
            metric_columns = ['termbase_accuracy', 'deontic_preservation', 
                            'conditional_logic_preservation', 'comet_score']
            
            for metric in metric_columns:
                if metric in ablation_df.columns:
                    values = ablation_df[metric].dropna()
                    if len(values) > 0:
                        ablation_stats['metrics'][metric] = {
                            'mean': values.mean(),
                            'std': values.std(),
                            'min': values.min(),
                            'max': values.max(),
                            'median': values.median()
                        }
            
            summary[ablation] = ablation_stats
        
        return summary
    
    def perform_statistical_tests(self) -> Dict[str, Any]:
        """执行统计检验"""
        tests = {}
        
        # 获取完整配置的结果作为基准
        if 'full' in self.results:
            baseline = self.df[self.df['ablation'] == 'full']
            
            for ablation in self.df['ablation'].unique():
                if ablation == 'full':
                    continue
                
                ablation_df = self.df[self.df['ablation'] == ablation]
                
                ablation_tests = {}
                
                # 对每个指标进行配对t检验
                metric_columns = ['termbase_accuracy', 'deontic_preservation', 
                                'conditional_logic_preservation', 'comet_score']
                
                for metric in metric_columns:
                    if metric in baseline.columns and metric in ablation_df.columns:
                        baseline_values = baseline[metric].dropna()
                        ablation_values = ablation_df[metric].dropna()
                        
                        if len(baseline_values) > 0 and len(ablation_values) > 0:
                            # 配对t检验
                            t_stat, p_value = stats.ttest_rel(baseline_values, ablation_values)
                            
                            # 效应量 (Cohen's d)
                            pooled_std = np.sqrt((baseline_values.var() + ablation_values.var()) / 2)
                            cohens_d = (baseline_values.mean() - ablation_values.mean()) / pooled_std
                            
                            ablation_tests[metric] = {
                                't_statistic': t_stat,
                                'p_value': p_value,
                                'cohens_d': cohens_d,
                                'significant': p_value < 0.05
                            }
                
                tests[f'full_vs_{ablation}'] = ablation_tests
        
        return tests
    
    def create_visualizations(self, output_dir: str = "results"):
        """创建可视化图表"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 设置图表样式
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. 指标对比箱线图
        self._plot_metrics_comparison(output_path)
        
        # 2. 消融实验效果图
        self._plot_ablation_effects(output_path)
        
        # 3. 语言对分析
        self._plot_language_pair_analysis(output_path)
        
        # 4. 文档类型分析
        self._plot_document_analysis(output_path)
    
    def _plot_metrics_comparison(self, output_path: Path):
        """绘制指标对比图"""
        metric_columns = ['termbase_accuracy', 'deontic_preservation', 
                         'conditional_logic_preservation', 'comet_score']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metric_columns):
            if metric in self.df.columns:
                sns.boxplot(data=self.df, x='ablation', y=metric, ax=axes[i])
                axes[i].set_title(f'{metric.replace("_", " ").title()}')
                axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_ablation_effects(self, output_path: Path):
        """绘制消融实验效果图"""
        metric_columns = ['termbase_accuracy', 'deontic_preservation', 
                         'conditional_logic_preservation', 'comet_score']
        
        # 计算相对于完整配置的改进/下降
        if 'full' in self.df['ablation'].values:
            baseline_means = self.df[self.df['ablation'] == 'full'][metric_columns].mean()
            
            ablation_effects = {}
            for ablation in self.df['ablation'].unique():
                if ablation != 'full':
                    ablation_means = self.df[self.df['ablation'] == ablation][metric_columns].mean()
                    effects = (ablation_means - baseline_means) / baseline_means * 100
                    ablation_effects[ablation] = effects
            
            # 绘制热力图
            effects_df = pd.DataFrame(ablation_effects).T
            plt.figure(figsize=(10, 6))
            sns.heatmap(effects_df, annot=True, fmt='.1f', cmap='RdYlBu_r', center=0)
            plt.title('Ablation Effects (% Change from Full Configuration)')
            plt.xlabel('Metrics')
            plt.ylabel('Ablation Configurations')
            plt.tight_layout()
            plt.savefig(output_path / 'ablation_effects.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_language_pair_analysis(self, output_path: Path):
        """绘制语言对分析图"""
        if 'src_lang' in self.df.columns and 'tgt_lang' in self.df.columns:
            self.df['language_pair'] = self.df['src_lang'] + '-' + self.df['tgt_lang']
            
            metric_columns = ['termbase_accuracy', 'deontic_preservation', 
                             'conditional_logic_preservation', 'comet_score']
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.flatten()
            
            for i, metric in enumerate(metric_columns):
                if metric in self.df.columns:
                    sns.barplot(data=self.df, x='language_pair', y=metric, hue='ablation', ax=axes[i])
                    axes[i].set_title(f'{metric.replace("_", " ").title()} by Language Pair')
                    axes[i].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(output_path / 'language_pair_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_document_analysis(self, output_path: Path):
        """绘制文档类型分析图"""
        if 'document_id' in self.df.columns:
            metric_columns = ['termbase_accuracy', 'deontic_preservation', 
                             'conditional_logic_preservation', 'comet_score']
            
            # 按文档类型分组分析
            doc_analysis = self.df.groupby(['document_id', 'ablation'])[metric_columns].mean().reset_index()
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.flatten()
            
            for i, metric in enumerate(metric_columns):
                if metric in doc_analysis.columns:
                    sns.barplot(data=doc_analysis, x='document_id', y=metric, hue='ablation', ax=axes[i])
                    axes[i].set_title(f'{metric.replace("_", " ").title()} by Document')
                    axes[i].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(output_path / 'document_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def generate_report(self, output_file: str = "analysis_report.md"):
        """生成分析报告"""
        summary = self.calculate_summary_statistics()
        tests = self.perform_statistical_tests()
        
        report = []
        report.append("# 法律翻译实验分析报告\n")
        
        # 1. 实验概述
        report.append("## 实验概述")
        report.append(f"- 总样本数: {len(self.df)}")
        report.append(f"- 消融配置数: {len(self.df['ablation'].unique())}")
        report.append(f"- 语言对: {', '.join(self.df['src_lang'].unique())} → {', '.join(self.df['tgt_lang'].unique())}")
        report.append("")
        
        # 2. 汇总统计
        report.append("## 汇总统计")
        for ablation, stats in summary.items():
            report.append(f"### {ablation}")
            report.append(f"- 样本数: {stats['count']}")
            report.append("- 指标表现:")
            for metric, values in stats['metrics'].items():
                report.append(f"  - {metric}: {values['mean']:.3f} ± {values['std']:.3f}")
            report.append("")
        
        # 3. 统计检验结果
        report.append("## 统计检验结果")
        for comparison, test_results in tests.items():
            report.append(f"### {comparison}")
            for metric, test in test_results.items():
                significance = "显著" if test['significant'] else "不显著"
                report.append(f"- {metric}: t={test['t_statistic']:.3f}, p={test['p_value']:.3f}, d={test['cohens_d']:.3f} ({significance})")
            report.append("")
        
        # 4. 结论
        report.append("## 结论")
        report.append("基于实验结果的分析结论...")
        
        # 保存报告
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"Analysis report saved to {output_file}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze experiment results')
    parser.add_argument('results_file', help='Path to results JSON file')
    parser.add_argument('--output-dir', default='results', help='Output directory for visualizations')
    parser.add_argument('--report', default='analysis_report.md', help='Report filename')
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = ExperimentAnalyzer(args.results_file)
    
    # 生成可视化
    analyzer.create_visualizations(args.output_dir)
    
    # 生成报告
    analyzer.generate_report(args.report)
    
    print("Analysis completed!")


if __name__ == "__main__":
    main()
