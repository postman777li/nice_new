"""
GEMBA-MQM 和 GEMBA-DA 评估指标
基于官方实现: https://github.com/MicrosoftTranslator/GEMBA

参考文献:
- GEMBA-MQM: Kocmi & Federmann (2023) "GEMBA-MQM: Detecting Translation Quality Error Spans with GPT-4"
- GEMBA-DA: Kocmi & Federmann (2023) "Large Language Models Are State-of-the-Art Evaluators of Translation Quality"
"""
from typing import List, Union, Dict, Optional
import logging
import asyncio
import re

logger = logging.getLogger(__name__)


class GEMBAMetric:
    """GEMBA评估指标 - 支持 GEMBA-MQM 和 GEMBA-DA 两种方法"""
    
    def __init__(self, 
                 method: str = "GEMBA-MQM",
                 model: str = "gpt-4",
                 temperature: float = 0.1):
        """
        Args:
            method: 评估方法 ('GEMBA-MQM' 或 'GEMBA-DA')
                - GEMBA-MQM: 基于MQM框架的详细错误检测和评分
                - GEMBA-DA: 直接评估（Direct Assessment），输出0-100分数
            model: LLM模型名称（推荐 gpt-4）
            temperature: 温度参数（官方推荐0.1）
        """
        if method not in ["GEMBA-MQM", "GEMBA-DA"]:
            raise ValueError(f"不支持的方法: {method}，请使用 'GEMBA-MQM' 或 'GEMBA-DA'")
        
        self.method = method
        self.model = model
        self.temperature = temperature
    
    def _get_gemba_mqm_prompt(self, 
                               source: str, 
                               translation: str, 
                               source_lang: str = "Chinese",
                               target_lang: str = "English") -> str:
        """
        构建GEMBA-MQM评估prompt（参考官方实现）
        
        官方prompt会要求模型：
        1. 识别翻译错误
        2. 按MQM标准分类错误（Accuracy, Fluency, Terminology, Style, Locale）
        3. 标注错误严重程度（Major, Minor, Critical）
        4. 计算最终分数
        """
        prompt = f"""You are an evaluator for machine translation. Your task is to identify translation errors in the hypothesis translation and classify them using the MQM error taxonomy.

The source text is in {source_lang} and the target text should be in {target_lang}.

Source: {source}
Translation: {translation}

Instructions:
1. Carefully compare the translation with the source text
2. Identify ALL translation errors
3. For each error, provide:
   - Error span in the translation
   - Error category (Accuracy, Fluency, Terminology, Style, Locale, or Other)
   - Error severity (Minor, Major, or Critical)
   - Brief explanation

4. Calculate the final score using MQM formula:
   - Start with 100 points
   - Deduct 1 point per Minor error
   - Deduct 5 points per Major error
   - Deduct 10 points per Critical error
   - Minimum score is 0

Return your analysis in JSON format:
{{
    "errors": [
        {{
            "error_span": "word or phrase with error",
            "category": "Accuracy|Fluency|Terminology|Style|Locale|Other",
            "severity": "Minor|Major|Critical",
            "explanation": "brief explanation of the error"
        }}
    ],
    "score": 85,
    "error_count": {{
        "minor": 2,
        "major": 1,
        "critical": 0
    }},
    "overall_quality": "brief assessment"
}}

If there are no errors, return an empty errors list and score of 100."""
        
        return prompt
    
    def _get_gemba_da_prompt(self, 
                             source: str, 
                             translation: str,
                             source_lang: str = "Chinese",
                             target_lang: str = "English") -> str:
        """
        构建GEMBA-DA评估prompt（参考官方实现）
        
        GEMBA-DA使用Direct Assessment方法，直接输出0-100的分数
        """
        prompt = f"""You are a professional translation quality evaluator. Your task is to evaluate the quality of a machine translation from {source_lang} to {target_lang}.

Source ({source_lang}): {source}
Translation ({target_lang}): {translation}

Please evaluate the translation quality considering:
1. Accuracy: How accurately does the translation convey the meaning of the source?
2. Fluency: How natural and fluent is the translation in the target language?
3. Adequacy: Is all information from the source present in the translation?

Provide a single score from 0 to 100, where:
- 0-25: Poor quality (major errors, meaning lost)
- 26-50: Fair quality (several errors, meaning partially preserved)
- 51-75: Good quality (minor errors, meaning mostly preserved)
- 76-100: Excellent quality (minimal or no errors, meaning fully preserved)

Return your evaluation in JSON format:
{{
    "score": 85,
    "explanation": "brief explanation of the score"
}}

Be objective and base your score only on the translation quality."""
        
        return prompt
    
    async def _call_llm(self, prompt: str) -> Dict:
        """调用LLM进行评估"""
        try:
            from src.lib.llm_client import LLMClient
            
            client = LLMClient()
            messages = [{"role": "user", "content": prompt}]
            
            result = await client.call_json(
                messages=messages,
                model=self.model,
                temperature=self.temperature
            )
            
            return result
        except Exception as e:
            logger.error(f"{self.method} LLM调用失败: {e}")
            if self.method == "GEMBA-MQM":
                return {
                    "errors": [],
                    "score": 0,
                    "error_count": {"minor": 0, "major": 0, "critical": 0},
                    "overall_quality": f"Evaluation failed: {e}"
                }
            else:  # GEMBA-DA
                return {
                    "score": 0,
                    "explanation": f"Evaluation failed: {e}"
                }
    
    async def compute_async(self,
                           sources: Union[str, List[str]],
                           predictions: Union[str, List[str]],
                           source_lang: str = "Chinese",
                           target_lang: str = "English") -> Dict:
        """
        异步计算GEMBA分数
        
        Args:
            sources: 源文本
            predictions: 预测翻译
            source_lang: 源语言名称（如 "Chinese", "English"）
            target_lang: 目标语言名称（如 "English", "Chinese"）
            
        Returns:
            评估结果字典
        """
        # 格式转换
        if isinstance(sources, str):
            sources = [sources]
        if isinstance(predictions, str):
            predictions = [predictions]
        
        results = []
        for src, pred in zip(sources, predictions):
            # 根据方法选择prompt
            if self.method == "GEMBA-MQM":
                prompt = self._get_gemba_mqm_prompt(src, pred, source_lang, target_lang)
            else:  # GEMBA-DA
                prompt = self._get_gemba_da_prompt(src, pred, source_lang, target_lang)
            
            result = await self._call_llm(prompt)
            results.append(result)
        
        # 计算平均分
        if results:
            avg_score = sum(r.get('score', 0) for r in results) / len(results)
        else:
            avg_score = 0.0
        
        return {
            'method': self.method,
            'results': results,
            'system_score': avg_score,
            'mean': avg_score
        }
    
    def compute(self,
                sources: Union[str, List[str]],
                predictions: Union[str, List[str]],
                source_lang: str = "Chinese",
                target_lang: str = "English") -> Dict:
        """
        同步计算GEMBA分数
        
        Args:
            sources: 源文本
            predictions: 预测翻译
            source_lang: 源语言名称
            target_lang: 目标语言名称
            
        Returns:
            评估结果字典
        """
        return asyncio.run(self.compute_async(sources, predictions, source_lang, target_lang))
    
    async def sentence_score_async(self,
                                   source: str,
                                   prediction: str,
                                   source_lang: str = "Chinese",
                                   target_lang: str = "English") -> float:
        """
        异步计算单句GEMBA分数
        
        Args:
            source: 源文本
            prediction: 预测翻译
            source_lang: 源语言名称
            target_lang: 目标语言名称
            
        Returns:
            分数 (0-100)
        """
        result = await self.compute_async([source], [prediction], source_lang, target_lang)
        return result['mean']
    
    def sentence_score(self,
                      source: str,
                      prediction: str,
                      source_lang: str = "Chinese",
                      target_lang: str = "English") -> float:
        """
        同步计算单句GEMBA分数
        
        Args:
            source: 源文本
            prediction: 预测翻译
            source_lang: 源语言名称
            target_lang: 目标语言名称
            
        Returns:
            分数 (0-100)
        """
        return asyncio.run(self.sentence_score_async(source, prediction, source_lang, target_lang))


# 为了兼容性，保留旧的类名
GEMBAMQMMetric = GEMBAMetric


# 使用示例
if __name__ == "__main__":
    print("="*60)
    print("GEMBA 评估指标测试")
    print("="*60)
    
    source = "合同双方应当遵守本协议的所有条款。"
    prediction = "The parties shall comply with all terms of this agreement."
    
    # 测试 GEMBA-MQM
    print("\n1. GEMBA-MQM (详细错误分析)")
    print("-"*60)
    mqm_metric = GEMBAMetric(method="GEMBA-MQM", model="gpt-4")
    mqm_result = mqm_metric.compute([source], [prediction], "Chinese", "English")
    print(f"分数: {mqm_result['mean']:.2f}")
    print(f"详细结果: {mqm_result['results'][0]}")
    
    # 测试 GEMBA-DA
    print("\n2. GEMBA-DA (直接评估)")
    print("-"*60)
    da_metric = GEMBAMetric(method="GEMBA-DA", model="gpt-4")
    da_result = da_metric.compute([source], [prediction], "Chinese", "English")
    print(f"分数: {da_result['mean']:.2f}")
    print(f"详细结果: {da_result['results'][0]}")
    
    print("\n" + "="*60)

