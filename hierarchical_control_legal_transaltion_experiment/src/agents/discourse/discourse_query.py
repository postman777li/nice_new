"""
篇章查询Agent - 使用混合检索识别相关翻译记忆
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
import sys
from pathlib import Path

# 添加项目路径以导入 tm_db
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from ..base import BaseAgent, AgentConfig, AgentRunContext

try:
    from src.lib.tm_db import get_default_tm_db
    TM_DB_AVAILABLE = True
except ImportError:
    TM_DB_AVAILABLE = False
    logging.warning("TM database not available")

logger = logging.getLogger(__name__)


@dataclass
class DiscourseQueryResult:
    source_text: str
    target_text: str
    similarity_score: float
    context: str
    legal_domain: str


class DiscourseQueryAgent(BaseAgent):
    def __init__(self, locale: str = 'zh', tm_db_path: str = "tm_bm25_index.json"):
        super().__init__(AgentConfig(
            name='discourse:query',
            role='discourse_querier',
            domain='discourse',
            specialty='篇章检索查询',
            quality='review',
            locale=locale
        ))
        self.tm_db = None
        self.tm_db_path = tm_db_path

    async def execute(self, input_data: Dict[str, Any], ctx: Optional[AgentRunContext] = None) -> List[DiscourseQueryResult]:
        """使用混合检索识别相关翻译记忆"""
        text = input_data.get('text', '')
        source_lang = input_data.get('source_lang', 'zh')
        target_lang = input_data.get('target_lang', 'en')
        top_k = input_data.get('top_k', 5)
        
        if not text:
            return []
        
        # 尝试使用真实的 TM 数据库
        if TM_DB_AVAILABLE:
            try:
                if self.tm_db is None:
                    # 使用全局单例以避免并发初始化冲突
                    from src.lib.tm_db import get_default_tm_db
                    self.tm_db = get_default_tm_db()
                
                # 生成查询向量
                query_vector = None
                try:
                    from src.lib.embeddings import get_embedding
                    query_vector = get_embedding(text)
                    logger.info("Generated query embedding vector")
                except Exception as e:
                    logger.warning(f"Failed to generate query vector, using BM25 only: {e}")
                
                # 混合检索（BM25 + 向量）
                tm_results = self.tm_db.hybrid_search(
                    query=text,
                    query_vector=query_vector,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    top_k=top_k
                )
                
                if tm_results:
                    logger.info(f"Found {len(tm_results)} TM matches using hybrid search")
                    return [
                        DiscourseQueryResult(
                            source_text=tm.source_text,
                            target_text=tm.target_text,
                            similarity_score=tm.similarity_score,
                            context=tm.context,
                            legal_domain=tm.legal_domain
                        )
                        for tm in tm_results
                    ]
            except Exception as e:
                logger.warning(f"TM database search failed, falling back to LLM: {e}")
        
        # 回退到 LLM 模拟检索
        logger.info("Using LLM-based TM retrieval (fallback)")
        messages = [
            {
                "role": "system",
                "content": f"""你是一个专业的法律篇章检索专家。请为给定的法律文本模拟检索最相关的翻译记忆。

检索要求：
1. 基于语义相似性生成相关的翻译示例
2. 识别top-{top_k}最佳翻译示例
3. 重点关注篇章层面的相似性
4. 考虑法律语境和修辞结构

请为每个检索结果提供：
- 源文本（与查询文本相似的法律文本）
- 目标文本（对应的翻译）
- 相似度评分
- 上下文
- 法律领域

返回JSON格式：
{{
    "results": [
        {{
            "source_text": "源文本",
            "target_text": "目标文本",
            "similarity_score": 0.9,
            "context": "使用上下文",
            "legal_domain": "法律领域"
        }}
    ]
}}"""
            },
            {
                "role": "user",
                "content": f"""请为以下{source_lang}到{target_lang}的法律文本检索最相关的翻译记忆：

查询文本：{text}"""
            }
        ]
        
        try:
            result = await self.call_llm_json(messages)
            results_data = result.get('results', [])
            
            return [
                DiscourseQueryResult(
                    source_text=item.get('source_text', ''),
                    target_text=item.get('target_text', ''),
                    similarity_score=item.get('similarity_score', 0.0),
                    context=item.get('context', ''),
                    legal_domain=item.get('legal_domain', '')
                )
                for item in results_data
            ]
        except Exception as e:
            logger.error(f"Discourse query failed: {e}")
            return []