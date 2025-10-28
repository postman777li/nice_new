"""
BERTScore 指标
基于预训练模型的语义相似度评估
"""
from typing import List, Union, Dict
import logging
import os
import torch 
torch.set_float32_matmul_precision('medium' )
logger = logging.getLogger(__name__)


class BERTScoreMetric:
    """BERTScore评估指标"""
    
    def __init__(self, 
                 model_type: str = "xlm-roberta-large",
                 num_layers: int = None,
                 lang: str = "zh",
                 rescale_with_baseline: bool = None,
                 use_fast_tokenizer: bool = True,
                 device: str = None,
                 use_hf_mirror: bool = True):
        """
        Args:
            model_type: 预训练模型名称
                - 'xlm-roberta-large' (推荐，多语言)
                - 'bert-base-chinese' (中文)
                - 'bert-base-uncased' (英文)
            num_layers: 使用的层数（None=自动选择）
            lang: 语言代码
            rescale_with_baseline: 是否使用基线重标定（None=自动检测）
            use_fast_tokenizer: 是否使用快速分词器
            device: 计算设备（None=自动检测，'cuda'=GPU，'cpu'=CPU）
            use_hf_mirror: 是否使用HF镜像加速（默认True）
        """
        self.model_type = model_type
        self.num_layers = num_layers
        self.lang = lang
        self.use_fast_tokenizer = use_fast_tokenizer
        self.use_hf_mirror = use_hf_mirror
        self._scorer = None
        
        # 智能设置 rescale_with_baseline
        # xlm-roberta-large + zh 没有预计算基准，禁用以避免警告
        if rescale_with_baseline is None:
            # 已知没有基准的组合
            no_baseline_combinations = [
                ('xlm-roberta-large', 'zh'),
                ('xlm-roberta-large', 'ja'),
                ('xlm-roberta-base', 'zh'),
                ('xlm-roberta-base', 'ja'),
            ]
            if (model_type, lang) in no_baseline_combinations:
                self.rescale_with_baseline = False
                logger.info(f"ℹ️  {model_type} + {lang} 没有预计算基准，已禁用 rescale_with_baseline")
            else:
                self.rescale_with_baseline = True
        else:
            self.rescale_with_baseline = rescale_with_baseline
        
        # 自动检测设备
        if device is None:
            import torch
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if self.device == 'cuda':
                logger.info(f"✓ 检测到GPU，BERTScore将使用GPU加速")
        else:
            self.device = device
        
        # 设置HF镜像
        if self.use_hf_mirror and 'HF_ENDPOINT' not in os.environ:
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
            logger.info("✓ 已启用 Hugging Face 镜像加速: https://hf-mirror.com")
    
    def _get_scorer(self):
        """延迟加载BERTScorer（缓存模型，避免重复加载）"""
        if self._scorer is None:
            try:
                from bert_score import BERTScorer
                logger.info(f"加载 BERTScore 模型: {self.model_type}")
                self._scorer = BERTScorer(
                    model_type=self.model_type,
                    num_layers=self.num_layers,
                    lang=self.lang,
                    device=self.device,
                    rescale_with_baseline=self.rescale_with_baseline,
                    batch_size=32  # 批处理大小
                )
                logger.info(f"✓ BERTScore 模型已加载到 {self.device}")
            except ImportError:
                logger.error("请安装 bert-score: pip install bert-score")
                raise
            except Exception as e:
                logger.error(f"BERTScore 模型加载失败: {e}")
                raise
        return self._scorer
    
    def compute(self, 
                predictions: Union[str, List[str]], 
                references: Union[str, List[str]]) -> Dict[str, float]:
        """
        计算BERTScore
        
        Args:
            predictions: 预测翻译
            references: 参考翻译
            
        Returns:
            包含 precision, recall, f1 的字典
        """
        scorer = self._get_scorer()
        
        # 格式转换
        if isinstance(predictions, str):
            predictions = [predictions]
        if isinstance(references, str):
            references = [references]
        
        try:
            P, R, F1 = scorer.score(predictions, references)
            
            return {
                'precision': P.mean().item(),
                'recall': R.mean().item(),
                'f1': F1.mean().item()
            }
        except Exception as e:
            logger.error(f"BERTScore计算失败: {e}")
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    def compute_batch(self,
                      predictions: List[str],
                      references: List[str]) -> Dict[str, Union[float, List[float]]]:
        """
        批量计算BERTScore，返回逐句分数与整体平均
        """
        scorer = self._get_scorer()
        try:
            P, R, F1 = scorer.score(predictions, references)
            p_list = P.tolist()
            r_list = R.tolist()
            f1_list = F1.tolist()
            return {
                'precision_list': p_list,
                'recall_list': r_list,
                'f1_list': f1_list,
                'precision': float(sum(p_list) / len(p_list)) if p_list else 0.0,
                'recall': float(sum(r_list) / len(r_list)) if r_list else 0.0,
                'f1': float(sum(f1_list) / len(f1_list)) if f1_list else 0.0,
            }
        except Exception as e:
            logger.error(f"BERTScore批量计算失败: {e}")
            return {
                'precision_list': [0.0] * len(predictions),
                'recall_list': [0.0] * len(predictions),
                'f1_list': [0.0] * len(predictions),
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
            }
    
    def sentence_score(self, prediction: str, reference: str) -> float:
        """
        计算单句BERTScore F1
        
        Args:
            prediction: 预测翻译
            reference: 参考翻译
            
        Returns:
            F1分数
        """
        result = self.compute([prediction], [reference])
        return result['f1']


# 使用示例
if __name__ == "__main__":
    # 注意：首次使用会下载模型
    metric = BERTScoreMetric(model_type="xlm-roberta-large")
    
    prediction = "The parties shall comply with the terms"
    reference = "Contracting parties must comply with all terms"
    
    scores = metric.compute([prediction], [reference])
    print(f"BERTScore - P: {scores['precision']:.4f}, R: {scores['recall']:.4f}, F1: {scores['f1']:.4f}")

