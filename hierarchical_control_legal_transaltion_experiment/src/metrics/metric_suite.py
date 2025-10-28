"""
统一的指标计算套件
"""
from typing import List, Dict, Any, Optional
import logging
import asyncio
from math import ceil

from .bleu import BLEUMetric
from .chrf import ChrFMetric
from .bertscore import BERTScoreMetric
from .comet import COMETMetric
from .gemba_mqm import GEMBAMetric

logger = logging.getLogger(__name__)


class MetricSuite:
    """统一的翻译评估指标套件"""
    
    def __init__(self,
                 metrics: List[str] = None,
                 lang: str = "zh",
                 use_gpu: bool = False,
                 gemba_method: str = "GEMBA-DA",
                 use_hf_mirror: bool = True):
        """
        Args:
            metrics: 要使用的指标列表，None表示使用所有指标
                可选: ['bleu', 'chrf', 'bertscore', 'comet', 'gemba']
            lang: 语言代码（如 'zh', 'en'）
            use_gpu: 是否使用GPU（for BERTScore, COMET）
            gemba_method: GEMBA方法 ('GEMBA-MQM' 或 'GEMBA-DA')
            use_hf_mirror: 是否使用HF镜像加速（默认True）
        """
        self.lang = lang
        self.use_gpu = use_gpu
        self.gemba_method = gemba_method
        self.use_hf_mirror = use_hf_mirror
        
        # 默认使用所有指标
        if metrics is None:
            metrics = ['bleu', 'chrf', 'bertscore', 'comet']
        
        self.metrics = {}
        self._initialize_metrics(metrics)
    
    def _initialize_metrics(self, metric_names: List[str]):
        """初始化指标"""
        for name in metric_names:
            try:
                if name == 'bleu':
                    self.metrics['bleu'] = BLEUMetric(tokenize=self.lang)
                    logger.info("✓ BLEU initialized")
                
                elif name == 'chrf':
                    self.metrics['chrf'] = ChrFMetric()
                    logger.info("✓ chrF++ initialized")
                
                elif name == 'bertscore':
                    self.metrics['bertscore'] = BERTScoreMetric(
                        model_type="xlm-roberta-large",
                        lang=self.lang,
                        device='cuda' if self.use_gpu else 'cpu',
                        use_hf_mirror=self.use_hf_mirror
                    )
                    logger.info("✓ BERTScore initialized")
                
                elif name == 'comet':
                    self.metrics['comet'] = COMETMetric(
                        model_name="Unbabel/wmt22-comet-da",
                        gpus=1 if self.use_gpu else 0,
                        use_hf_mirror=self.use_hf_mirror
                    )
                    logger.info("✓ COMET initialized")
                
                elif name == 'gemba' or name == 'gemba_mqm' or name == 'gemba_da':
                    self.metrics['gemba'] = GEMBAMetric(
                        method=self.gemba_method,
                        model="gpt-4"
                    )
                    logger.info(f"✓ {self.gemba_method} initialized")
                
                else:
                    logger.warning(f"未知指标: {name}")
            
            except Exception as e:
                logger.warning(f"初始化 {name} 失败: {e}")
    
    def compute(self,
                source: str,
                prediction: str,
                reference: str) -> Dict[str, float]:
        """
        计算所有指标
        
        Args:
            source: 源文本
            prediction: 预测翻译
            reference: 参考翻译
            
        Returns:
            包含所有指标分数的字典
        """
        results = {}
        
        # BLEU
        if 'bleu' in self.metrics:
            try:
                results['bleu'] = self.metrics['bleu'].sentence_score(prediction, reference)
            except Exception as e:
                logger.error(f"BLEU计算失败: {e}")
                results['bleu'] = 0.0
        
        # chrF++
        if 'chrf' in self.metrics:
            try:
                results['chrf'] = self.metrics['chrf'].sentence_score(prediction, reference)
            except Exception as e:
                logger.error(f"chrF++计算失败: {e}")
                results['chrf'] = 0.0
        
        # BERTScore
        if 'bertscore' in self.metrics:
            try:
                bert_scores = self.metrics['bertscore'].compute([prediction], [reference])
                results['bertscore_f1'] = bert_scores['f1']
                results['bertscore_p'] = bert_scores['precision']
                results['bertscore_r'] = bert_scores['recall']
            except Exception as e:
                logger.error(f"BERTScore计算失败: {e}")
                results['bertscore_f1'] = 0.0
        
        # COMET
        if 'comet' in self.metrics:
            try:
                comet_result = self.metrics['comet'].compute(
                    [source], [prediction], [reference]
                )
                results['comet'] = comet_result['mean']
            except Exception as e:
                logger.error(f"COMET计算失败: {e}")
                results['comet'] = 0.0
        
        # GEMBA (可能是MQM或DA)
        if 'gemba' in self.metrics:
            try:
                # 根据语言代码转换为语言名称
                lang_map = {'zh': 'Chinese', 'en': 'English', 'de': 'German', 'fr': 'French'}
                source_lang = lang_map.get(self.lang, 'Chinese')
                target_lang = 'English' if self.lang == 'zh' else 'Chinese'
                
                gemba_result = self.metrics['gemba'].compute(
                    [source], [prediction], source_lang, target_lang
                )
                results['gemba'] = gemba_result['mean']
            except Exception as e:
                logger.error(f"GEMBA计算失败: {e}")
                results['gemba'] = 0.0
        
        return results
    
    async def compute_async(self,
                           source: str,
                           prediction: str,
                           reference: str) -> Dict[str, float]:
        """
        异步计算所有指标
        
        Args:
            source: 源文本
            prediction: 预测翻译
            reference: 参考翻译
            
        Returns:
            包含所有指标分数的字典
        """
        # 对于异步指标使用异步方法
        results = self.compute(source, prediction, reference)
        
        # GEMBA异步版本
        if 'gemba' in self.metrics:
            try:
                # 根据语言代码转换为语言名称
                lang_map = {'zh': 'Chinese', 'en': 'English', 'de': 'German', 'fr': 'French'}
                source_lang = lang_map.get(self.lang, 'Chinese')
                target_lang = 'English' if self.lang == 'zh' else 'Chinese'
                
                gemba_result = await self.metrics['gemba'].compute_async(
                    [source], [prediction], source_lang, target_lang
                )
                results['gemba'] = gemba_result['mean']
            except Exception as e:
                logger.error(f"GEMBA计算失败: {e}")
                results['gemba'] = 0.0
        
        return results

    def compute_batch(self,
                      sources: List[str],
                      predictions: List[str],
                      references: List[str],
                      batch_size: int = 64) -> List[Dict[str, float]]:
        """
        批量计算所有指标，按输入顺序返回每句的分数字典。
        对于支持批量的指标（BERTScore, COMET）合并调用；BLEU/chrF逐句计算。
        """
        num = len(predictions)
        assert len(sources) == num and len(references) == num

        # 预分配结果列表
        results_list: List[Dict[str, float]] = [dict() for _ in range(num)]

        # 先处理逐句指标（BLEU, chrF）
        for i in range(num):
            s = sources[i]
            p = predictions[i]
            r = references[i]

            if 'bleu' in self.metrics:
                try:
                    results_list[i]['bleu'] = self.metrics['bleu'].sentence_score(p, r)
                except Exception as e:
                    logger.error(f"BLEU计算失败: {e}")
                    results_list[i]['bleu'] = 0.0

            if 'chrf' in self.metrics:
                try:
                    results_list[i]['chrf'] = self.metrics['chrf'].sentence_score(p, r)
                except Exception as e:
                    logger.error(f"chrF++计算失败: {e}")
                    results_list[i]['chrf'] = 0.0

        # BERTScore 批量
        if 'bertscore' in self.metrics:
            try:
                for start in range(0, num, batch_size):
                    end = min(start + batch_size, num)
                    batch = self.metrics['bertscore'].compute_batch(
                        predictions[start:end], references[start:end]
                    )
                    f1_list = batch.get('f1_list', [])
                    p_list = batch.get('precision_list', [])
                    r_list = batch.get('recall_list', [])
                    for j, idx in enumerate(range(start, end)):
                        if j < len(f1_list):
                            results_list[idx]['bertscore_f1'] = float(f1_list[j])
                        if j < len(p_list):
                            results_list[idx]['bertscore_p'] = float(p_list[j])
                        if j < len(r_list):
                            results_list[idx]['bertscore_r'] = float(r_list[j])
            except Exception as e:
                logger.error(f"BERTScore批量计算失败: {e}")

        # COMET 批量
        if 'comet' in self.metrics:
            try:
                for start in range(0, num, batch_size):
                    end = min(start + batch_size, num)
                    comet_out = self.metrics['comet'].compute(
                        sources[start:end], predictions[start:end], references[start:end]
                    )
                    scores = comet_out.get('scores', [])
                    for j, idx in enumerate(range(start, end)):
                        if j < len(scores):
                            results_list[idx]['comet'] = float(scores[j])
            except Exception as e:
                logger.error(f"COMET批量计算失败: {e}")

        # GEMBA 目前仍逐句或异步，不在此批量
        return results_list


# 使用示例
if __name__ == "__main__":
    # 创建指标套件（快速指标）
    suite = MetricSuite(metrics=['bleu', 'chrf'])
    
    source = "合同双方应当遵守本协议的所有条款。"
    prediction = "The parties shall comply with all terms of this agreement."
    reference = "Contracting parties must comply with all provisions of this agreement."
    
    scores = suite.compute(source, prediction, reference)
    
    print("Translation Evaluation Scores:")
    for metric, score in scores.items():
        print(f"  {metric}: {score:.2f}")

