"""
术语预处理协调器 - 组织完整的术语预处理流程
"""
from typing import List, Dict, Any, Optional
from pathlib import Path
import asyncio
import logging
import json

from .mono_extract import MonoExtractAgent, MonoExtractItem
from .deduplicate import DeduplicateAgent, DeduplicatedTerm
from .batch_translate import BatchTranslateAgent, BatchTranslationResult
from .search import SearchAgent

# 引入术语库和数据集类型
try:
    from ...lib.term_db import TermDatabase
except Exception:
    from lib.term_db import TermDatabase

# TestSample在顶层datasets模块
try:
    from datasets import TestSample
except Exception:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    from datasets import TestSample

logger = logging.getLogger(__name__)


class TerminologyPreprocessor:
    """术语预处理协调器"""
    
    def __init__(self,
                 src_lang: str = 'zh',
                 tgt_lang: str = 'en',
                 domain: str = 'law',
                 db_path: str = 'backend/terms.db',
                 max_concurrent: int = 10,
                 batch_size: int = 20):
        """
        初始化术语预处理器
        
        Args:
            src_lang: 源语言
            tgt_lang: 目标语言
            domain: 领域
            db_path: 术语库路径
            max_concurrent: 最大并发数（用于提取术语）
            batch_size: 批量翻译的批次大小
        """
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.domain = domain
        self.db_path = db_path
        self.batch_size = batch_size
        
        # 初始化各个Agent
        self.mono_extract = MonoExtractAgent(locale=src_lang)
        self.dedup = DeduplicateAgent(locale=src_lang)
        self.batch_translate = BatchTranslateAgent(locale=src_lang, db_path=db_path)
        self.search = SearchAgent(locale=src_lang, db_path=db_path)
        self.db = TermDatabase(db_path)
        
        # 并发控制
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def preprocess_dataset(self, 
                                samples: List[TestSample],
                                output_file: Optional[Path] = None,
                                verbose: bool = True) -> Dict[str, Any]:
        """
        预处理数据集：提取、去重、翻译、导入术语
        
        Args:
            samples: 测试样本列表
            output_file: 可选的输出文件路径（保存中间结果）
            verbose: 是否显示详细信息
        
        Returns:
            统计信息字典
        """
        if verbose:
            print("\n" + "="*60)
            print("术语批量预处理")
            print("="*60)
            print(f"数据集样本数: {len(samples)}")
            print(f"源语言: {self.src_lang} -> 目标语言: {self.tgt_lang}")
            print(f"领域: {self.domain}")
            print(f"术语库: {self.db_path}")
        
        # 步骤1: 批量提取术语
        if verbose:
            print("\n" + "-"*60)
            print("步骤1: 批量提取术语")
            print("-"*60)
        
        extracted_terms, contexts = await self._extract_terms_from_samples(samples, verbose)
        
        if verbose:
            total_terms = sum(len(terms) for terms in extracted_terms)
            print(f"✓ 从 {len(samples)} 个样本中提取了 {total_terms} 个术语")
        
        # 步骤2: 去重合并
        if verbose:
            print("\n" + "-"*60)
            print("步骤2: 去重合并术语")
            print("-"*60)
        
        deduplicated = await self.dedup.execute({
            'extracted_terms': extracted_terms,
            'contexts': contexts,
            'max_contexts': 2
        })
        
        if verbose:
            print(f"✓ 去重后得到 {len(deduplicated)} 个不重复术语")
            if deduplicated:
                print(f"  Top 10 高频术语:")
                for i, term in enumerate(deduplicated[:10], 1):
                    print(f"    {i}. {term.term} (出现{term.count}次, 分数{term.score:.2f})")
        
        # 步骤3: 查询 + 批量翻译
        if verbose:
            print("\n" + "-"*60)
            print("步骤3: 查询数据库 + 批量翻译")
            print("-"*60)
        
        translations = await self.batch_translate.execute({
            'terms': deduplicated,
            'src_lang': self.src_lang,
            'tgt_lang': self.tgt_lang,
            'domain': self.domain,
            'batch_size': self.batch_size
        })
        
        if verbose:
            from_db = sum(1 for t in translations if t.source == 'database')
            from_llm = sum(1 for t in translations if t.source == 'llm')
            print(f"✓ 翻译完成: 数据库{from_db}个 + LLM{from_llm}个 = 共{len(translations)}个")
        
        # 步骤4: 导入到术语库
        if verbose:
            print("\n" + "-"*60)
            print("步骤4: 导入到术语库")
            print("-"*60)
        
        imported = self._import_to_database(translations, verbose)
        
        if verbose:
            print(f"✓ 成功导入 {imported} 个新术语到数据库")
        
        # 构建统计结果
        stats = {
            'total_samples': len(samples),
            'total_extracted': sum(len(terms) for terms in extracted_terms),
            'deduplicated': len(deduplicated),
            'from_database': sum(1 for t in translations if t.source == 'database'),
            'from_llm': sum(1 for t in translations if t.source == 'llm'),
            'imported_to_db': imported,
            'top_terms': [
                {
                    'term': t.term,
                    'translation': next((tr.translation for tr in translations if tr.term == t.term), ''),
                    'count': t.count,
                    'score': t.score
                }
                for t in deduplicated[:20]
            ]
        }
        
        # 保存到文件（可选）
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            if verbose:
                print(f"\n结果已保存到: {output_file}")
        
        if verbose:
            print("\n" + "="*60)
            print("✅ 术语预处理完成！")
            print("="*60)
        
        return stats
    
    async def _extract_terms_from_samples(self, 
                                         samples: List[TestSample],
                                         verbose: bool = True) -> tuple:
        """从样本中批量提取术语（并发）"""
        async def extract_one(sample: TestSample, idx: int) -> tuple:
            async with self.semaphore:
                try:
                    terms = await self.mono_extract.execute({
                        'text': sample.source
                    })
                    if verbose and (idx + 1) % 10 == 0:
                        print(f"  进度: {idx + 1}/{len(samples)}")
                    return terms, sample.source
                except Exception as e:
                    logger.error(f"提取术语失败 (sample {sample.id}): {e}")
                    return [], sample.source
        
        # 并发提取
        tasks = [extract_one(sample, i) for i, sample in enumerate(samples)]
        results = await asyncio.gather(*tasks)
        
        # 分离术语和上下文
        extracted_terms = [r[0] for r in results]
        contexts = [r[1] for r in results]
        
        return extracted_terms, contexts
    
    def _import_to_database(self, 
                           translations: List[BatchTranslationResult],
                           verbose: bool = True) -> int:
        """导入翻译结果到术语库"""
        imported = 0
        skipped = 0
        
        for trans in translations:
            # 只导入LLM新翻译的术语（来自数据库的已经存在）
            if trans.source != 'llm' or not trans.translation:
                skipped += 1
                continue
            
            try:
                self.db.add_term(
                    source_term=trans.term,
                    target_term=trans.translation,
                    source_lang=self.src_lang,
                    target_lang=self.tgt_lang,
                    domain=self.domain,
                    source='batch_preprocessing'
                )
                imported += 1
            except Exception as e:
                logger.error(f"导入术语失败 ({trans.term}): {e}")
        
        if verbose and skipped > 0:
            print(f"  跳过 {skipped} 个已存在或无效的术语")
        
        return imported

