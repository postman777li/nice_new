"""
SQLite 术语表数据库模块
用于存储和管理翻译术语表
"""
import sqlite3
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json
import time
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Term:
    """术语条目"""
    id: Optional[int] = None
    source_term: str = ""
    target_term: str = ""
    source_lang: str = ""
    target_lang: str = ""
    domain: str = ""
    confidence: float = 1.0
    quality_score: float = 1.0
    combined_score: float = 1.0
    category: str = ""
    law: str = ""
    year: int = 0
    entry_id: str = ""
    source_context: str = ""
    target_context: str = ""
    occurrence_count: int = 1
    original_source_term: str = ""
    original_target_term: str = ""
    metadata: Dict[str, Any] = None
    created_at: int = 0
    updated_at: int = 0
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.created_at == 0:
            self.created_at = int(time.time())
        if self.updated_at == 0:
            self.updated_at = self.created_at
        # 如果没有提供原始术语，使用归一化后的术语
        if not self.original_source_term:
            self.original_source_term = self.source_term
        if not self.original_target_term:
            self.original_target_term = self.target_term


class TermDatabase:
    """术语数据库管理器"""
    
    def __init__(self, db_path: str = "terms.db"):
        self.db_path = db_path
        self._ensure_db_dir()
        self._init_database()
        # 启用WAL和busy_timeout，减少并发锁冲突
        try:
            with sqlite3.connect(self.db_path, timeout=30) as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA journal_mode=WAL;")
                cursor.execute("PRAGMA synchronous=NORMAL;")
                cursor.execute("PRAGMA busy_timeout=5000;")
                conn.commit()
        except Exception as e:
            logger.warning(f"设置SQLite并发参数失败: {e}")
    
    def _ensure_db_dir(self):
        """确保数据库目录存在"""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
    
    def _init_database(self):
        """初始化数据库表"""
        try:
            with sqlite3.connect(self.db_path, timeout=30) as conn:
                cursor = conn.cursor()
                
                # 检查是否需要迁移旧表
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='terms'")
                if cursor.fetchone():
                    # 检查是否有新字段
                    cursor.execute("PRAGMA table_info(terms)")
                    columns = [row[1] for row in cursor.fetchall()]
                    if 'quality_score' not in columns:
                        logger.info("检测到旧表结构，正在迁移...")
                        # 删除旧表
                        cursor.execute("DROP TABLE IF EXISTS terms")
                        cursor.execute("DROP TABLE IF EXISTS term_collection_items")
                        cursor.execute("DROP TABLE IF EXISTS term_collections")
                        conn.commit()
                        logger.info("已删除旧表，将创建新表结构")
                
                # 创建术语表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS terms (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        source_term TEXT NOT NULL,
                        target_term TEXT NOT NULL,
                        source_lang TEXT NOT NULL,
                        target_lang TEXT NOT NULL,
                        domain TEXT DEFAULT '',
                        confidence REAL DEFAULT 1.0,
                        quality_score REAL DEFAULT 1.0,
                        combined_score REAL DEFAULT 1.0,
                        category TEXT DEFAULT '',
                        law TEXT DEFAULT '',
                        year INTEGER DEFAULT 0,
                        entry_id TEXT DEFAULT '',
                        source_context TEXT DEFAULT '',
                        target_context TEXT DEFAULT '',
                        occurrence_count INTEGER DEFAULT 1,
                        original_source_term TEXT DEFAULT '',
                        original_target_term TEXT DEFAULT '',
                        metadata TEXT DEFAULT '{}',
                        created_at INTEGER NOT NULL,
                        updated_at INTEGER NOT NULL
                    )
                ''')
                
                # 创建索引
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_terms_source 
                    ON terms(source_term, source_lang)
                ''')
                
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_terms_target 
                    ON terms(target_term, target_lang)
                ''')
                
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_terms_lang_pair 
                    ON terms(source_lang, target_lang)
                ''')
                
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_terms_domain 
                    ON terms(domain)
                ''')
                
                # 创建术语集合表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS term_collections (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT UNIQUE NOT NULL,
                        description TEXT DEFAULT '',
                        source_lang TEXT NOT NULL,
                        target_lang TEXT NOT NULL,
                        domain TEXT DEFAULT '',
                        created_at INTEGER NOT NULL,
                        updated_at INTEGER NOT NULL
                    )
                ''')
                
                # 创建术语-集合关联表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS term_collection_items (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        collection_id INTEGER NOT NULL,
                        term_id INTEGER NOT NULL,
                        FOREIGN KEY (collection_id) REFERENCES term_collections (id),
                        FOREIGN KEY (term_id) REFERENCES terms (id),
                        UNIQUE(collection_id, term_id)
                    )
                ''')
                
                conn.commit()
                logger.info(f"Initialized term database at {self.db_path}")
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
    
    def add_term(self, term: Term) -> Optional[int]:
        """添加术语"""
        try:
            with sqlite3.connect(self.db_path, timeout=30) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO terms (
                        source_term, target_term, source_lang, target_lang,
                        domain, confidence, quality_score, combined_score, category,
                        law, year, entry_id, source_context, target_context,
                        occurrence_count, original_source_term, original_target_term,
                        metadata, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    term.source_term, term.target_term, term.source_lang, term.target_lang,
                    term.domain, term.confidence, term.quality_score, term.combined_score, term.category,
                    term.law, term.year, term.entry_id, term.source_context, term.target_context,
                    term.occurrence_count, term.original_source_term, term.original_target_term,
                    json.dumps(term.metadata), term.created_at, term.updated_at
                ))
                
                term_id = cursor.lastrowid
                conn.commit()
                
                logger.info(f"Added term: {term.source_term} -> {term.target_term}")
                return term_id
                
        except Exception as e:
            logger.error(f"Failed to add term: {e}")
            return None
    
    def get_term(self, term_id: int) -> Optional[Term]:
        """获取术语"""
        try:
            with sqlite3.connect(self.db_path, timeout=30) as conn:
                cursor = conn.cursor()
                
                cursor.execute('SELECT * FROM terms WHERE id = ?', (term_id,))
                row = cursor.fetchone()
                
                if row:
                    return self._row_to_term(row)
                return None
                
        except Exception as e:
            logger.error(f"Failed to get term {term_id}: {e}")
            return None
    
    def search_terms(self, source_term: str = "", target_term: str = "",
                    source_lang: str = "", target_lang: str = "",
                    domain: str = "", limit: int = 100, exact_match: bool = False) -> List[Term]:
        """搜索术语
        
        Args:
            source_term: 源术语
            target_term: 目标术语
            source_lang: 源语言
            target_lang: 目标语言
            domain: 领域
            limit: 返回数量限制
            exact_match: 是否精确匹配（True=精确，False=模糊）
        """
        try:
            with sqlite3.connect(self.db_path, timeout=30) as conn:
                cursor = conn.cursor()
                
                conditions = []
                params = []
                
                if source_term:
                    if exact_match:
                        conditions.append("source_term = ?")
                        params.append(source_term)
                    else:
                        conditions.append("source_term LIKE ?")
                        params.append(f"%{source_term}%")
                
                if target_term:
                    if exact_match:
                        conditions.append("target_term = ?")
                        params.append(target_term)
                    else:
                        conditions.append("target_term LIKE ?")
                        params.append(f"%{target_term}%")
                
                if source_lang:
                    conditions.append("source_lang = ?")
                    params.append(source_lang)
                
                if target_lang:
                    conditions.append("target_lang = ?")
                    params.append(target_lang)
                
                if domain:
                    conditions.append("domain = ?")
                    params.append(domain)
                
                where_clause = " AND ".join(conditions) if conditions else "1=1"
                query = f"SELECT * FROM terms WHERE {where_clause} ORDER BY confidence DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                return [self._row_to_term(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to search terms: {e}")
            return []
    
    def update_term(self, term: Term) -> bool:
        """更新术语"""
        if not term.id:
            return False
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                term.updated_at = int(time.time())
                
                cursor.execute('''
                    UPDATE terms SET
                        source_term = ?, target_term = ?, source_lang = ?, target_lang = ?,
                        domain = ?, confidence = ?, quality_score = ?, combined_score = ?, category = ?,
                        law = ?, year = ?, entry_id = ?, source_context = ?, target_context = ?,
                        occurrence_count = ?, original_source_term = ?, original_target_term = ?,
                        metadata = ?, updated_at = ?
                    WHERE id = ?
                ''', (
                    term.source_term, term.target_term, term.source_lang, term.target_lang,
                    term.domain, term.confidence, term.quality_score, term.combined_score, term.category,
                    term.law, term.year, term.entry_id, term.source_context, term.target_context,
                    term.occurrence_count, term.original_source_term, term.original_target_term,
                    json.dumps(term.metadata), term.updated_at, term.id
                ))
                
                conn.commit()
                logger.info(f"Updated term {term.id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to update term {term.id}: {e}")
            return False
    
    def delete_term(self, term_id: int) -> bool:
        """删除术语"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('DELETE FROM terms WHERE id = ?', (term_id,))
                conn.commit()
                
                logger.info(f"Deleted term {term_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete term {term_id}: {e}")
            return False
    
    def batch_add_terms(self, terms: List[Term]) -> int:
        """批量添加术语"""
        added_count = 0
        conn = None
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for term in terms:
                try:
                    cursor.execute('''
                        INSERT INTO terms (
                            source_term, target_term, source_lang, target_lang,
                            domain, confidence, quality_score, combined_score, category,
                            law, year, entry_id, source_context, target_context,
                            occurrence_count, original_source_term, original_target_term,
                            metadata, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        term.source_term, term.target_term, term.source_lang, term.target_lang,
                        term.domain, term.confidence, term.quality_score, term.combined_score, term.category,
                        term.law, term.year, term.entry_id, term.source_context, term.target_context,
                        term.occurrence_count, term.original_source_term, term.original_target_term,
                        json.dumps(term.metadata), term.created_at, term.updated_at
                    ))
                    added_count += 1
                except Exception as e:
                    logger.warning(f"Failed to add term {term.source_term}: {e}")
            
            conn.commit()
            logger.info(f"Batch added {added_count} terms")
            
        except Exception as e:
            logger.error(f"Failed to batch add terms: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()
        
        return added_count
    
    def get_term_stats(self) -> Dict[str, Any]:
        """获取术语统计信息"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 总术语数
            cursor.execute('SELECT COUNT(*) FROM terms')
            total_terms = cursor.fetchone()[0]
            
            # 按语言对统计
            cursor.execute('''
                SELECT source_lang, target_lang, COUNT(*) 
                FROM terms 
                GROUP BY source_lang, target_lang
            ''')
            lang_pairs = cursor.fetchall()
            
            # 按领域统计
            cursor.execute('''
                SELECT domain, COUNT(*) 
                FROM terms 
                WHERE domain != '' 
                GROUP BY domain
                ORDER BY COUNT(*) DESC
            ''')
            domains = cursor.fetchall()
            
            conn.close()
            
            return {
                "total_terms": total_terms,
                "language_pairs": [{"source": lp[0], "target": lp[1], "count": lp[2]} for lp in lang_pairs],
                "domains": [{"domain": d[0], "count": d[1]} for d in domains]
            }
                
        except Exception as e:
            logger.error(f"Failed to get term stats: {e}")
            return {}
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取术语统计信息（别名方法，返回更详细的格式）"""
        stats = self.get_term_stats()
        
        if not stats:
            return {
                'total_terms': 0,
                'language_pairs': {},
                'domains': {},
                'laws': {}
            }
        
        # 转换语言对格式
        language_pairs = {}
        for lp in stats.get('language_pairs', []):
            key = f"{lp['source']}-{lp['target']}"
            language_pairs[key] = lp['count']
        
        # 转换领域格式
        domains = {}
        for d in stats.get('domains', []):
            domains[d['domain']] = d['count']
        
        return {
            'total_terms': stats.get('total_terms', 0),
            'language_pairs': language_pairs,
            'domains': domains,
            'laws': {}  # 可以后续扩展
        }
    
    def _row_to_term(self, row) -> Term:
        """将数据库行转换为Term对象"""
        return Term(
            id=row[0],
            source_term=row[1],
            target_term=row[2],
            source_lang=row[3],
            target_lang=row[4],
            domain=row[5],
            confidence=row[6],
            quality_score=row[7],
            combined_score=row[8],
            category=row[9],
            law=row[10],
            year=row[11],
            entry_id=row[12],
            source_context=row[13],
            target_context=row[14],
            occurrence_count=row[15],
            original_source_term=row[16],
            original_target_term=row[17],
            metadata=json.loads(row[18]) if row[18] else {},
            created_at=row[19],
            updated_at=row[20]
        )


# 全局术语数据库实例（用于默认操作）
_default_term_db = None


def get_default_term_db() -> TermDatabase:
    """获取默认的术语数据库实例"""
    global _default_term_db
    if _default_term_db is None:
        _default_term_db = TermDatabase()
    return _default_term_db


def add_legal_term(source: str, target: str, source_lang: str, target_lang: str,
                  domain: str = "legal", confidence: float = 1.0, context: str = "") -> Optional[int]:
    """添加法律术语"""
    term = Term(
        source_term=source,
        target_term=target,
        source_lang=source_lang,
        target_lang=target_lang,
        domain=domain,
        confidence=confidence,
        context=context
    )
    return get_default_term_db().add_term(term)


def search_legal_terms(source_term: str, source_lang: str, target_lang: str,
                      domain: str = "legal", limit: int = 10) -> List[Term]:
    """搜索法律术语"""
    return get_default_term_db().search_terms(
        source_term=source_term,
        source_lang=source_lang,
        target_lang=target_lang,
        domain=domain,
        limit=limit
    )


def get_term_translation(source_term: str, source_lang: str, target_lang: str,
                        domain: str = "legal") -> Optional[str]:
    """获取术语翻译"""
    terms = get_default_term_db().search_terms(
        source_term=source_term,
        source_lang=source_lang,
        target_lang=target_lang,
        domain=domain,
        limit=1
    )
    
    if terms:
        return terms[0].target_term
    return None


def import_terms_from_dict(terms_dict: List[Dict[str, Any]], source_lang: str = "zh", target_lang: str = "en", db_path: str = "terms.db") -> int:
    """从字典列表导入术语
    
    Args:
        terms_dict: 术语字典列表
        source_lang: 源语言代码（默认'zh'）
        target_lang: 目标语言代码（默认'en'）
        db_path: 数据库文件路径（默认'terms.db'）
    
    Returns:
        成功导入的术语数量
    """
    terms = []
    for item in terms_dict:
        # 确保语言字段不为空
        src_lang = item.get("source_lang", "") or source_lang or "zh"
        tgt_lang = item.get("target_lang", "") or target_lang or "en"
        
        term = Term(
            source_term=item.get("source_term", item.get("source", "")),
            target_term=item.get("target_term", item.get("target", "")),
            source_lang=src_lang,
            target_lang=tgt_lang,
            domain=item.get("domain", "legal"),
            confidence=item.get("confidence", 1.0),
            quality_score=item.get("quality_score", 1.0),
            combined_score=item.get("combined_score", 1.0),
            category=item.get("category", ""),
            law=item.get("law", ""),
            year=item.get("year", 0),
            entry_id=str(item.get("entry_id", "")),
            source_context=item.get("source_context", ""),
            target_context=item.get("target_context", ""),
            occurrence_count=item.get("occurrence_count", 1),
            original_source_term=item.get("original_source_term", ""),
            original_target_term=item.get("original_target_term", ""),
            metadata=item.get("metadata", {})
        )
        terms.append(term)
    
    # 使用指定路径的数据库实例
    db = TermDatabase(db_path)
    return db.batch_add_terms(terms)
