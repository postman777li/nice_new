"""
BLEU (Bilingual Evaluation Understudy) 指标
传统的n-gram重叠评估指标
"""
from typing import List, Union
import logging

logger = logging.getLogger(__name__)


class BLEUMetric:
    """BLEU评估指标"""
    
    def __init__(self, lowercase: bool = False, tokenize: str = 'zh'):
        """
        Args:
            lowercase: 是否转换为小写
            tokenize: 分词方式 ('zh', 'intl', '13a', 'char')
        """
        self.lowercase = lowercase
        self.tokenize = tokenize
        self._scorer = None
    
    def _get_scorer(self):
        """延迟加载sacrebleu"""
        if self._scorer is None:
            try:
                from sacrebleu.metrics import BLEU
                self._scorer = BLEU(
                    lowercase=self.lowercase,
                    tokenize=self.tokenize,
                    effective_order=True  # 启用有效阶数（推荐用于句子级BLEU）
                )
            except ImportError:
                logger.error("请安装 sacrebleu: pip install sacrebleu")
                raise
        return self._scorer
    
    def compute(self, 
                predictions: Union[str, List[str]], 
                references: Union[str, List[str], List[List[str]]]) -> float:
        """
        计算BLEU分数
        
        Args:
            predictions: 预测翻译（单个字符串或列表）
            references: 参考翻译（单个、列表或列表的列表）
            
        Returns:
            BLEU分数 (0-100)
        """
        scorer = self._get_scorer()
        
        # 确保格式正确
        if isinstance(predictions, str):
            predictions = [predictions]
        
        if isinstance(references, str):
            references = [[references]]
        elif isinstance(references, list) and len(references) > 0:
            if isinstance(references[0], str):
                references = [[ref] for ref in references]
        
        try:
            result = scorer.corpus_score(predictions, references)
            return result.score
        except Exception as e:
            logger.error(f"BLEU计算失败: {e}")
            return 0.0
    
    def sentence_score(self, prediction: str, reference: str) -> float:
        """
        计算单句BLEU分数
        
        Args:
            prediction: 预测翻译
            reference: 参考翻译
            
        Returns:
            BLEU分数 (0-100)
        """
        scorer = self._get_scorer()
        
        try:
            result = scorer.sentence_score(prediction, [reference])
            return result.score
        except Exception as e:
            logger.error(f"BLEU单句计算失败: {e}")
            return 0.0


# 使用示例
if __name__ == "__main__":
    metric = BLEUMetric(tokenize='zh')
    
    prediction = "合同双方应当遵守协议条款"
    reference = "合同当事人应当遵守本协议的所有条款"
    
    score = metric.sentence_score(prediction, reference)
    print(f"BLEU: {score:.2f}")

