"""
chrF / chrF++ 指标
基于字符级n-gram的F-score，特别适合中文等语言
"""
from typing import List, Union
import logging

logger = logging.getLogger(__name__)


class ChrFMetric:
    """chrF++评估指标"""
    
    def __init__(self, 
                 word_order: int = 2,
                 char_order: int = 6,
                 beta: int = 2,
                 lowercase: bool = False,
                 whitespace: bool = False):
        """
        Args:
            word_order: 词级n-gram最大值 (chrF++使用2)
            char_order: 字符级n-gram最大值
            beta: F-score的beta参数
            lowercase: 是否转小写
            whitespace: 是否包含空格
        """
        self.word_order = word_order
        self.char_order = char_order
        self.beta = beta
        self.lowercase = lowercase
        self.whitespace = whitespace
        self._scorer = None
    
    def _get_scorer(self):
        """延迟加载sacrebleu"""
        if self._scorer is None:
            try:
                from sacrebleu.metrics import CHRF
                self._scorer = CHRF(
                    word_order=self.word_order,
                    char_order=self.char_order,
                    beta=self.beta,
                    lowercase=self.lowercase,
                    whitespace=self.whitespace
                )
            except ImportError:
                logger.error("请安装 sacrebleu: pip install sacrebleu")
                raise
        return self._scorer
    
    def compute(self, 
                predictions: Union[str, List[str]], 
                references: Union[str, List[str], List[List[str]]]) -> float:
        """
        计算chrF++分数
        
        Args:
            predictions: 预测翻译
            references: 参考翻译
            
        Returns:
            chrF++分数 (0-100)
        """
        scorer = self._get_scorer()
        
        # 格式转换
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
            logger.error(f"chrF++计算失败: {e}")
            return 0.0
    
    def sentence_score(self, prediction: str, reference: str) -> float:
        """
        计算单句chrF++分数
        
        Args:
            prediction: 预测翻译
            reference: 参考翻译
            
        Returns:
            chrF++分数 (0-100)
        """
        scorer = self._get_scorer()
        
        try:
            result = scorer.sentence_score(prediction, [reference])
            return result.score
        except Exception as e:
            logger.error(f"chrF++单句计算失败: {e}")
            return 0.0


# 使用示例
if __name__ == "__main__":
    metric = ChrFMetric()
    
    prediction = "The parties shall comply with the terms of the agreement"
    reference = "Contracting parties must comply with all terms of this agreement"
    
    score = metric.sentence_score(prediction, reference)
    print(f"chrF++: {score:.2f}")

