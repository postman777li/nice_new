"""
Agents模块 - Python版本
"""
from .base import BaseAgent, AgentConfig, AgentRunContext
from .baseline_translation import BaselineTranslationAgent, BaselineTranslationResult
from .preprocess import DocumentTermExtractAgent, DocumentTermTranslateAgent
from .terminology import (
    MonoExtractAgent, MonoExtractItem,
    SearchAgent, SearchResult,
    EvaluateAgent, EvaluateResult,
    TranslationAgent, TranslationResult
)
from .syntax import (
    BiExtractAgent, BiExtractItem,
    SyntaxEvaluateAgent, SyntaxEvaluateResult,
    SyntaxTranslationAgent, SyntaxTranslationResult
)
from .discourse import (
    DiscourseQueryAgent, DiscourseQueryResult,
    DiscourseEvaluateAgent, DiscourseEvaluateResult,
    DiscourseTranslationAgent, DiscourseTranslationResult
)

__all__ = [
    'BaseAgent', 'AgentConfig', 'AgentRunContext',
    'BaselineTranslationAgent', 'BaselineTranslationResult',
    'DocumentTermExtractAgent', 'DocumentTermTranslateAgent',
    'MonoExtractAgent', 'MonoExtractItem',
    'SearchAgent', 'SearchResult',
    'EvaluateAgent', 'EvaluateResult',
    'TranslationAgent', 'TranslationResult',
    'BiExtractAgent', 'BiExtractItem',
    'SyntaxEvaluateAgent', 'SyntaxEvaluateResult',
    'SyntaxTranslationAgent', 'SyntaxTranslationResult',
    'DiscourseQueryAgent', 'DiscourseQueryResult',
    'DiscourseEvaluateAgent', 'DiscourseEvaluateResult',
    'DiscourseTranslationAgent', 'DiscourseTranslationResult'
]
