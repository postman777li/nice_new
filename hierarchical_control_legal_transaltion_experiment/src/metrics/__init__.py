"""
现代机器翻译评估指标模块

基于最新的机器翻译评估研究实现（2023-2024）
"""
from .bleu import BLEUMetric
from .chrf import ChrFMetric
from .bertscore import BERTScoreMetric
from .comet import COMETMetric
from .gemba_mqm import GEMBAMetric, GEMBAMQMMetric
from .metric_suite import MetricSuite

__all__ = [
    'BLEUMetric',
    'ChrFMetric',
    'BERTScoreMetric',
    'COMETMetric',
    'GEMBAMetric',
    'GEMBAMQMMetric',  # 兼容性别名
    'MetricSuite',
]

