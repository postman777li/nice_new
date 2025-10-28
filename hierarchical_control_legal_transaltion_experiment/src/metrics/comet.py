"""
COMET (Crosslingual Optimized Metric for Evaluation of Translation)
基于神经网络的翻译质量评估指标
"""
from typing import List, Union, Dict
import logging
import os
import torch

logger = logging.getLogger(__name__)


class COMETMetric:
    """COMET评估指标"""
    
    def __init__(self, 
                 model_name: str = "Unbabel/wmt22-comet-da",
                 batch_size: int = 8,
                 gpus: int = None,
                 use_hf_mirror: bool = True):
        """
        Args:
            model_name: COMET模型名称
                - 'Unbabel/wmt22-comet-da' (推荐，需要参考翻译)
                - 'Unbabel/wmt22-cometkiwi-da' (QE模型，不需要参考)
                - 'Unbabel/XCOMET-XXL' (最新最大模型)
            batch_size: 批处理大小
            gpus: 使用的GPU数量（None=自动检测，0=CPU，1=单GPU）
            use_hf_mirror: 是否使用HF镜像加速（默认True）
        """
        self.model_name = model_name
        self.batch_size = batch_size
        
        # 自动检测GPU
        if gpus is None:
            self.gpus = 1 if torch.cuda.is_available() else 0
            if self.gpus > 0:
                logger.info(f"✓ 检测到GPU，COMET将使用GPU加速")
        else:
            self.gpus = gpus
            
        self.use_hf_mirror = use_hf_mirror
        self._model = None
        self._load_attempted = False
        self._load_error = None
        
        # 设置HF镜像
        if self.use_hf_mirror and 'HF_ENDPOINT' not in os.environ:
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
            logger.info("✓ 已启用 Hugging Face 镜像加速: https://hf-mirror.com")
    
    def _load_model(self):
        """延迟加载COMET模型"""
        if self._model is None and not self._load_attempted:
            try:
                from comet import download_model, load_from_checkpoint
                
                logger.info(f"加载COMET模型: {self.model_name}")
                # 依次尝试若干兼容名称
                candidate_names = [
                    self.model_name,
                    self.model_name.split('/')[-1] if '/' in self.model_name else self.model_name,
                    'wmt22-comet-da',
                    'wmt20-comet-da'
                ]
                last_err = None
                for name in candidate_names:
                    try:
                        logger.info(f"尝试加载COMET模型别名: {name}")
                        model_path = download_model(name)
                        self._model = load_from_checkpoint(model_path)
                        logger.info("COMET模型加载完成")
                        break
                    except Exception as e:
                        last_err = e
                        logger.warning(f"COMET模型别名 '{name}' 加载失败: {e}")
                        self._model = None
                if self._model is None and last_err is not None:
                    raise last_err
            except ImportError:
                logger.error("请安装 unbabel-comet: pip install unbabel-comet")
                self._load_error = "unbabel-comet not installed"
                self._load_attempted = True
                return None
            except Exception as e:
                logger.error(f"COMET模型加载失败: {e}")
                self._load_error = str(e)
                self._load_attempted = True
                return None
            self._load_attempted = True
        return self._model
    
    def compute(self, 
                sources: Union[str, List[str]],
                predictions: Union[str, List[str]], 
                references: Union[str, List[str]] = None) -> Dict[str, float]:
        """
        计算COMET分数
        
        Args:
            sources: 源文本
            predictions: 预测翻译
            references: 参考翻译（QE模型不需要）
            
        Returns:
            包含分数和系统级分数的字典
        """
        model = self._load_model()
        if model is None:
            # 加载失败时，避免反复尝试，直接返回零分
            return {
                'scores': [0.0] * (1 if isinstance(sources, str) else len(sources)),
                'system_score': 0.0,
                'mean': 0.0
            }
        
        # 格式转换
        if isinstance(sources, str):
            sources = [sources]
        if isinstance(predictions, str):
            predictions = [predictions]
        if references is not None and isinstance(references, str):
            references = [references]
        
        # 准备数据
        data = []
        for i, (src, mt) in enumerate(zip(sources, predictions)):
            sample = {
                "src": src,
                "mt": mt
            }
            if references is not None:
                sample["ref"] = references[i]
            data.append(sample)
        
        try:
            # 预测分数
            model_output = model.predict(
                data, 
                batch_size=self.batch_size,
                gpus=self.gpus
            )
            
            # 提取分数
            if isinstance(model_output, dict):
                scores = model_output.get('scores', [])
                system_score = model_output.get('system_score', 0.0)
            else:
                scores = model_output
                system_score = sum(scores) / len(scores) if scores else 0.0
            
            return {
                'scores': scores,
                'system_score': system_score,
                'mean': system_score
            }
        except Exception as e:
            logger.error(f"COMET计算失败: {e}")
            return {
                'scores': [0.0] * len(sources),
                'system_score': 0.0,
                'mean': 0.0
            }
    
    def sentence_score(self, 
                       source: str,
                       prediction: str, 
                       reference: str = None) -> float:
        """
        计算单句COMET分数
        
        Args:
            source: 源文本
            prediction: 预测翻译
            reference: 参考翻译
            
        Returns:
            COMET分数
        """
        result = self.compute([source], [prediction], [reference] if reference else None)
        return result['mean']


# 使用示例
if __name__ == "__main__":
    # 注意：首次使用会下载模型（较大）
    metric = COMETMetric(model_name="Unbabel/wmt22-comet-da")
    
    source = "合同双方应当遵守本协议的所有条款。"
    prediction = "The parties shall comply with all terms of this agreement."
    reference = "Contracting parties must comply with all provisions of this agreement."
    
    score = metric.sentence_score(source, prediction, reference)
    print(f"COMET: {score:.4f}")

