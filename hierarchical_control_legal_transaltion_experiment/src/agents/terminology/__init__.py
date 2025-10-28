"""
术语处理Agent包
包含术语工作流的核心Agent：
- MonoExtractAgent: 单语术语提取
- SearchAgent: 术语库检索
- EvaluateAgent: 术语评估
- TranslationAgent: 术语翻译
- DeduplicateAgent: 术语去重与合并
- BatchTranslateAgent: 批量术语翻译
- TerminologyPreprocessor: 术语预处理协调器
"""
from .mono_extract import MonoExtractAgent, MonoExtractItem
from .search import SearchAgent, SearchResult
from .evaluate import EvaluateAgent, EvaluateResult
from .translation import TranslationAgent, TranslationResult
from .deduplicate import DeduplicateAgent, DeduplicatedTerm
from .batch_translate import BatchTranslateAgent, BatchTranslationResult
from .preprocess import TerminologyPreprocessor

__all__ = [
    'MonoExtractAgent', 'MonoExtractItem',
    'SearchAgent', 'SearchResult', 
    'EvaluateAgent', 'EvaluateResult',
    'TranslationAgent', 'TranslationResult',
    'DeduplicateAgent', 'DeduplicatedTerm',
    'BatchTranslateAgent', 'BatchTranslationResult',
    'TerminologyPreprocessor'
]
