"""
篇章处理Agent包
包含篇章工作流的三个核心Agent：
- DiscourseQueryAgent: 篇章检索查询
- DiscourseEvaluateAgent: 篇章一致性分析（分析当前翻译与参考的差异）
- DiscourseTranslationAgent: 篇章综合翻译
"""
from .discourse_query import DiscourseQueryAgent, DiscourseQueryResult
from .discourse_evaluate import DiscourseEvaluateAgent, DiscourseEvaluateResult
from .discourse_translation import DiscourseTranslationAgent, DiscourseTranslationResult

__all__ = [
    'DiscourseQueryAgent', 'DiscourseQueryResult',
    'DiscourseEvaluateAgent', 'DiscourseEvaluateResult',
    'DiscourseTranslationAgent', 'DiscourseTranslationResult'
]
