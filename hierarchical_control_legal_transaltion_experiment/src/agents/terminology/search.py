"""
术语检索Agent - 从术语库中检索候选翻译
"""
from typing import List, Dict, Any, Optional
import asyncio
from dataclasses import dataclass
import logging

from ..base import BaseAgent, AgentConfig, AgentRunContext

logger = logging.getLogger(__name__)

# 引入术语库查询，兼容相对/绝对导入
try:
    from ...lib.term_db import TermDatabase
except Exception:  # pragma: no cover
    from lib.term_db import TermDatabase


@dataclass
class SearchResult:
    term: str
    translation: str
    confidence: float
    source: str
    context: str


class SearchAgent(BaseAgent):
    def __init__(self, locale: str = 'zh', db_path: str = 'backend/terms.db'):
        super().__init__(AgentConfig(
            name='terminology:search',
            role='terminology_searcher',
            domain='terminology',
            specialty='术语库检索',
            quality='review',
            locale=locale
        ))
        self.db_path = db_path
        self.db = None
        # 限制单Agent内的并发查询，避免线程池过载
        self._db_semaphore = asyncio.Semaphore(10)

    async def execute(self, input_data: Dict[str, Any], ctx: Optional[AgentRunContext] = None) -> List[SearchResult]:
        """从术语库中检索候选翻译（不再调用 LLM）"""
        terms: List[str] = input_data.get('terms', [])
        source_lang: str = input_data.get('source_lang', 'zh')
        target_lang: str = input_data.get('target_lang', 'en')
        domain: str = input_data.get('domain', 'legal')
        exact_match: bool = input_data.get('exact_match', True)  # 默认使用精确匹配

        if not terms:
            return []

        # 初始化数据库连接
        if self.db is None:
            self.db = TermDatabase(self.db_path)

        results: List[SearchResult] = []
        seen: set = set()

        try:
            for t in terms:
                # 将阻塞的SQLite查询放入线程池，防止阻塞事件循环
                async with self._db_semaphore:
                    hits = await asyncio.to_thread(
                        self.db.search_terms,
                        t,
                        "",
                        source_lang,
                        target_lang,
                        domain,
                        10,
                        exact_match
                    )
                for hit in hits:
                    key = (hit.source_term, hit.target_term, hit.source_lang, hit.target_lang)
                    if key in seen:
                        continue
                    seen.add(key)
                    origin = 'termbase'
                    try:
                        if getattr(hit, 'metadata', None) and isinstance(hit.metadata, dict):
                            origin = hit.metadata.get('source', origin) or origin
                    except Exception:
                        pass
                    results.append(SearchResult(
                        term=hit.source_term,
                        translation=hit.target_term,
                        confidence=hit.confidence,
                        source=origin,
                        context=hit.source_context or ''
                    ))
        except Exception as e:
            logger.error(f"SearchAgent termbase query failed: {e}")
            return []

        return results
