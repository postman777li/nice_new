"""
翻译记忆（Translation Memory）管理库
结合 BM25 + Milvus 向量检索实现混合检索
"""
import logging
import json
import time
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import os  # <--- 添加这行
from dotenv import load_dotenv # <--- 添加这行


try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logging.warning("rank-bm25 not installed. BM25 search will be disabled.")

from .vector_db import vector_db, VectorSearchResult

load_dotenv() # <--- 添加这行，确保环境变量被加载
logger = logging.getLogger(__name__)


@dataclass
class TMEntry:
    """翻译记忆条目"""
    id: str
    source_text: str
    target_text: str
    source_lang: str
    target_lang: str
    domain: str
    similarity_score: float = 0.0
    context: str = ""
    legal_domain: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if not self.legal_domain and self.domain:
            self.legal_domain = self.domain


class TranslationMemoryDB:
    """翻译记忆数据库管理器（BM25 + Milvus 混合检索）"""
    
    def __init__(self, collection_name: str = "translation_memory", 
                 bm25_index_path: str = "tm_bm25_index.json",
                 vector_dimension: int = 0):
        self.collection_name = collection_name
        self.bm25_index_path = Path(bm25_index_path)
        self.bm25_index = None
        self.corpus_data = []  # BM25 语料库
        
        # # 根据嵌入模型自动判断维度
        # if vector_dimension == 0:
        #     import os
        #     embed_model = os.getenv("OPENAI_EMBED_MODEL", "")
        #     if "doubao" in embed_model.lower():
        #         vector_dimension = 2560
        #     else:
        #         vector_dimension = 768
        #
        # self.vector_dimension = vector_dimension
        # 根据嵌入模型自动判断维度  改
        if vector_dimension == 0:
            import os
            try:
                # 优先从 .env 读取 EMBEDDING_DIM，这和你 .env 文件一致
                vector_dimension = int(os.getenv("EMBEDDING_DIM"))
                logger.info(f"成功从 .env 读取维度 (EMBEDDING_DIM): {vector_dimension}")
            except (TypeError, ValueError, AttributeError):
                # 如果 .env 里没读到，才执行旧的（有问题的）逻辑
                logger.warning("未在 .env 中找到 EMBEDDING_DIM, 尝试旧的自动检测...")
                embed_model = os.getenv("OPENAI_EMBED_MODEL", "")
                if "doubao" in embed_model.lower():
                    vector_dimension = 2560
                else:
                    vector_dimension = 768  # 旧的错误默认值
                    logger.error(f"自动检测默认为 768 维。请确保你的 .env 文件中有 EMBEDDING_DIM=1536")

        self.vector_dimension = vector_dimension
        
        # 确保 Milvus collection 存在
        if not vector_db.connected:
            logger.warning("Milvus not connected. TM will work in degraded mode.")
        elif collection_name not in vector_db.collections:
            vector_db.create_collection(collection_name, dimension=vector_dimension, description="翻译记忆向量集合")
        
        # 加载 BM25 索引
        self._load_bm25_index()
    
    def _load_bm25_index(self):
        """加载 BM25 索引"""
        if not BM25_AVAILABLE:
            logger.warning("BM25 not available, skipping BM25 index loading")
            return
        
        if self.bm25_index_path.exists():
            try:
                with open(self.bm25_index_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.corpus_data = data.get('corpus', [])
                    
                if self.corpus_data:
                    # 重建 BM25 索引
                    tokenized_corpus = [self._tokenize(item['source_text']) for item in self.corpus_data]
                    self.bm25_index = BM25Okapi(tokenized_corpus)
                    logger.info(f"Loaded BM25 index with {len(self.corpus_data)} entries")
            except Exception as e:
                logger.error(f"Failed to load BM25 index: {e}")
                self.corpus_data = []
        else:
            logger.info("No existing BM25 index found")
    
    def _save_bm25_index(self):
        """保存 BM25 索引"""
        if not BM25_AVAILABLE:
            return
        
        try:
            self.bm25_index_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.bm25_index_path, 'w', encoding='utf-8') as f:
                json.dump({'corpus': self.corpus_data}, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved BM25 index with {len(self.corpus_data)} entries")
        except Exception as e:
            logger.error(f"Failed to save BM25 index: {e}")
    
    def _tokenize(self, text: str) -> List[str]:
        """简单分词（可以根据语言优化）"""
        # 中文：按字符分词
        # 英文：按空格分词
        if any('\u4e00' <= char <= '\u9fff' for char in text):
            return list(text)
        else:
            return text.lower().split()
    
    def _generate_id(self, source_text: str, target_text: str, source_lang: str, target_lang: str) -> str:
        """生成唯一ID"""
        content = f"{source_lang}:{target_lang}:{source_text}:{target_text}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def add_entry(self, source_text: str, target_text: str, 
                  source_lang: str, target_lang: str, 
                  domain: str = "legal", 
                  source_vector: Optional[List[float]] = None,
                  context: str = "", 
                  metadata: Optional[Dict[str, Any]] = None) -> bool:
        """添加翻译记忆条目"""
        try:
            entry_id = self._generate_id(source_text, target_text, source_lang, target_lang)
            
            # 添加到 Milvus (如果有向量)
            if vector_db.connected and source_vector:
                data = [{
                    "id": entry_id,
                    "vector": source_vector,
                    "text": f"{source_text}|||{target_text}",  # 用分隔符连接
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "domain": domain,
                    "created_at": int(time.time())
                }]
                vector_db.insert_vectors(self.collection_name, data)
            
            # 添加到 BM25 索引
            if BM25_AVAILABLE:
                corpus_entry = {
                    'id': entry_id,
                    'source_text': source_text,
                    'target_text': target_text,
                    'source_lang': source_lang,
                    'target_lang': target_lang,
                    'domain': domain,
                    'context': context,
                    'metadata': metadata or {}
                }
                self.corpus_data.append(corpus_entry)
                
                # 重建 BM25 索引
                tokenized_corpus = [self._tokenize(item['source_text']) for item in self.corpus_data]
                self.bm25_index = BM25Okapi(tokenized_corpus)
                
                # 定期保存（每100条）
                if len(self.corpus_data) % 100 == 0:
                    self._save_bm25_index()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add TM entry: {e}")
            return False
    
    def batch_add_entries(self, entries: List[Dict[str, Any]], batch_size: int = 100) -> int:
        """批量添加翻译记忆条目（优化版：批量插入 Milvus）"""
        success_count = 0
        
        # 分离有向量和无向量的条目
        entries_with_vectors = []
        entries_without_vectors = []
        
        for entry in entries:
            if entry.get('source_vector'):
                entries_with_vectors.append(entry)
            else:
                entries_without_vectors.append(entry)
        
        # 批量插入 Milvus（有向量的）
        if entries_with_vectors and vector_db.connected:
            logger.info(f"Batch inserting {len(entries_with_vectors)} entries with vectors to Milvus...")
            
            for i in range(0, len(entries_with_vectors), batch_size):
                batch = entries_with_vectors[i:i+batch_size]
                milvus_data = []
                
                for entry in batch:
                    entry_id = self._generate_id(
                        entry['source_text'], 
                        entry['target_text'],
                        entry.get('source_lang', 'zh'),
                        entry.get('target_lang', 'en')
                    )
                    
                    milvus_data.append({
                        "id": entry_id,
                        "vector": entry['source_vector'],
                        "text": f"{entry['source_text']}|||{entry['target_text']}",
                        "source_lang": entry.get('source_lang', 'zh'),
                        "target_lang": entry.get('target_lang', 'en'),
                        "domain": entry.get('domain', 'legal'),
                        "created_at": int(time.time())
                    })
                
                if milvus_data:
                    if vector_db.insert_vectors(self.collection_name, milvus_data):
                        success_count += len(milvus_data)
                        logger.info(f"Inserted batch {i//batch_size + 1}: {len(milvus_data)} vectors")
        
        # 添加到 BM25 索引（所有条目）
        if BM25_AVAILABLE:
            for entry in entries:
                corpus_entry = {
                    'id': self._generate_id(
                        entry['source_text'],
                        entry['target_text'],
                        entry.get('source_lang', 'zh'),
                        entry.get('target_lang', 'en')
                    ),
                    'source_text': entry['source_text'],
                    'target_text': entry['target_text'],
                    'source_lang': entry.get('source_lang', 'zh'),
                    'target_lang': entry.get('target_lang', 'en'),
                    'domain': entry.get('domain', 'legal'),
                    'context': entry.get('context', ''),
                    'metadata': entry.get('metadata', {})
                }
                self.corpus_data.append(corpus_entry)
            
            # 重建 BM25 索引
            tokenized_corpus = [self._tokenize(item['source_text']) for item in self.corpus_data]
            self.bm25_index = BM25Okapi(tokenized_corpus)
            
            # 保存 BM25 索引
            self._save_bm25_index()
            logger.info(f"Updated BM25 index with {len(self.corpus_data)} entries")
        
        # 对于没有向量的条目，也算作成功（因为至少有 BM25）
        success_count += len(entries_without_vectors)
        
        logger.info(f"Batch added {success_count}/{len(entries)} TM entries")
        return success_count
    
    def search_bm25(self, query: str, source_lang: str = "", target_lang: str = "", 
                    top_k: int = 10) -> List[TMEntry]:
        """使用 BM25 搜索"""
        if not BM25_AVAILABLE or not self.bm25_index or not self.corpus_data:
            return []
        
        try:
            # 分词查询
            tokenized_query = self._tokenize(query)
            
            # BM25 打分
            scores = self.bm25_index.get_scores(tokenized_query)
            
            # 获取 top_k 结果
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k * 2]
            
            results = []
            for idx in top_indices:
                if idx >= len(self.corpus_data):
                    continue
                
                item = self.corpus_data[idx]
                
                # 语言过滤
                if source_lang and item['source_lang'] != source_lang:
                    continue
                if target_lang and item['target_lang'] != target_lang:
                    continue
                
                results.append(TMEntry(
                    id=item['id'],
                    source_text=item['source_text'],
                    target_text=item['target_text'],
                    source_lang=item['source_lang'],
                    target_lang=item['target_lang'],
                    domain=item['domain'],
                    similarity_score=float(scores[idx]) / 100.0,  # 归一化BM25分数到0-1
                    context=item.get('context', ''),
                    legal_domain=item.get('domain', ''),
                    metadata=item.get('metadata', {})
                ))
                
                if len(results) >= top_k:
                    break
            
            return results
            
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []
    
    def search_vector(self, query_vector: List[float], source_lang: str = "", 
                     target_lang: str = "", top_k: int = 10) -> List[TMEntry]:
        """使用向量搜索"""
        if not vector_db.connected:
            return []
        
        try:
            # 构建过滤条件
            filters = []
            if source_lang:
                filters.append(f'source_lang == "{source_lang}"')
            if target_lang:
                filters.append(f'target_lang == "{target_lang}"')
            
            filter_expr = " and ".join(filters) if filters else None
            
            # 向量搜索
            vector_results = vector_db.search_vectors(
                self.collection_name, 
                query_vector, 
                top_k=top_k, 
                filters=filter_expr
            )
            
            # 转换为 TMEntry
            results = []
            for vr in vector_results:
                # 解析文本（用 ||| 分隔的 source|||target）
                text = vr.metadata.get('text', '')
                parts = text.split('|||')
                if len(parts) == 2:
                    source_text, target_text = parts
                else:
                    source_text = target_text = text
                
                results.append(TMEntry(
                    id=vr.id,
                    source_text=source_text,
                    target_text=target_text,
                    source_lang=vr.metadata.get('source_lang', ''),
                    target_lang=vr.metadata.get('target_lang', ''),
                    domain=vr.metadata.get('domain', ''),
                    similarity_score=vr.score,
                    context="",
                    legal_domain=vr.metadata.get('domain', ''),
                    metadata={}
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    def hybrid_search(self, query: str, query_vector: Optional[List[float]] = None,
                     source_lang: str = "", target_lang: str = "", 
                     top_k: int = 10, bm25_weight: float = 0.5, 
                     vector_weight: float = 0.5) -> List[TMEntry]:
        """混合检索（BM25 + 向量）"""
        results_map = {}
        
        # BM25 搜索
        if BM25_AVAILABLE and self.bm25_index:
            bm25_results = self.search_bm25(query, source_lang, target_lang, top_k * 2)
            for result in bm25_results:
                if result.id not in results_map:
                    results_map[result.id] = result
                    results_map[result.id].similarity_score = result.similarity_score * bm25_weight
                else:
                    results_map[result.id].similarity_score += result.similarity_score * bm25_weight
        
        # 向量搜索
        if query_vector and vector_db.connected:
            vector_results = self.search_vector(query_vector, source_lang, target_lang, top_k * 2)
            for result in vector_results:
                if result.id not in results_map:
                    results_map[result.id] = result
                    results_map[result.id].similarity_score = result.similarity_score * vector_weight
                else:
                    results_map[result.id].similarity_score += result.similarity_score * vector_weight
        
        # 排序并返回 top_k
        sorted_results = sorted(results_map.values(), key=lambda x: x.similarity_score, reverse=True)
        return sorted_results[:top_k]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取 TM 统计信息"""
        stats = {
            'bm25_entries': len(self.corpus_data) if BM25_AVAILABLE else 0,
            'bm25_available': BM25_AVAILABLE,
            'milvus_available': vector_db.connected
        }
        
        if vector_db.connected:
            milvus_stats = vector_db.get_collection_stats(self.collection_name)
            stats['milvus_entries'] = milvus_stats.get('num_entities', 0)
        
        return stats


# 全局 TM 实例
_default_tm_db = None


def get_default_tm_db() -> TranslationMemoryDB:
    """获取默认的 TM 数据库实例"""
    global _default_tm_db
    if _default_tm_db is None:
        _default_tm_db = TranslationMemoryDB()
    return _default_tm_db

