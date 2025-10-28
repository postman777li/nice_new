#!/usr/bin/env python3
"""
导入翻译记忆到数据库
支持从双语平行语料导入到 TM（Milvus + BM25）
"""
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import os
from dotenv import load_dotenv

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lib.tm_db import TranslationMemoryDB, get_default_tm_db

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_dataset(file_path: Path) -> List[Dict[str, Any]]:
    """加载数据集"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 支持两种格式：直接列表 或 包含 data 字段的对象
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return data.get('data', data.get('entries', []))
        else:
            logger.error(f"Unsupported data format in {file_path}")
            return []
    except Exception as e:
        logger.error(f"Failed to load dataset from {file_path}: {e}")
        return []


def get_embedding(text: str) -> List[float]:
    """获取文本的嵌入向量（使用 OpenAI 兼容 API）"""
    try:
        from src.lib.embeddings import get_embedding as _get_embedding
        return _get_embedding(text)
    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}")
        # 返回空向量表示失败
        return []


def import_from_parallel_corpus(file_path: Path, tm_db: TranslationMemoryDB, 
                                source_lang: str, target_lang: str, 
                                domain: str = "legal", 
                                max_entries: int = -1,
                                use_embeddings: bool = False,
                                batch_size: int = 5) -> int:
    """从平行语料导入 TM"""
    logger.info(f"Loading dataset from {file_path}...")
    entries = load_dataset(file_path)
    
    if not entries:
        logger.error("No entries found in dataset")
        return 0
    
    if max_entries > 0:
        entries = entries[:max_entries]
    
    logger.info(f"Found {len(entries)} entries, preparing to import...")
    
    # 准备 TM 条目
    tm_entries = []
    source_texts = []
    
    for entry in tqdm(entries, desc="Processing entries"):
        # 支持多种键名格式
        source_text = entry.get(source_lang, entry.get('source', entry.get('zh', '')))
        target_text = entry.get(target_lang, entry.get('target', entry.get('en', '')))
        
        if not source_text or not target_text:
            continue
        
        tm_entry = {
            'source_text': source_text,
            'target_text': target_text,
            'source_lang': source_lang,
            'target_lang': target_lang,
            'domain': domain,
            'context': entry.get('context', ''),
            'metadata': {
                'law': entry.get('law', ''),
                'year': entry.get('year', ''),
                'entry_id': entry.get('id', entry.get('entry_id', ''))
            }
        }
        
        tm_entries.append(tm_entry)
        if use_embeddings:
            source_texts.append(source_text)
    
    # 批量生成嵌入向量（如果启用）
    if use_embeddings and source_texts:
        logger.info(f"Generating embeddings for {len(source_texts)} texts in batches of {batch_size}...")
        try:
            from src.lib.embeddings import get_embeddings_batch
            
            all_embeddings = []
            failed_batches = 0
            
            for i in tqdm(range(0, len(source_texts), batch_size), desc="Generating embeddings"):
                batch_texts = source_texts[i:i+batch_size]
                try:
                    batch_embeddings = get_embeddings_batch(batch_texts)
                    all_embeddings.extend(batch_embeddings)
                except Exception as e:
                    failed_batches += 1
                    logger.warning(f"Failed batch {i//batch_size + 1}: {str(e)[:100]}")
                    # 为失败的批次填充空向量
                    all_embeddings.extend([[] for _ in batch_texts])
                    
                    # 如果连续失败太多，提示用户
                    if failed_batches >= 5 and i < len(source_texts) // 2:
                        logger.error(f"Too many failed batches ({failed_batches}). Consider:")
                        logger.error("1. Reducing --embedding-batch-size (try 10 or 5)")
                        logger.error("2. Checking API service status")
                        logger.error("3. Running without --use-embeddings")
                        logger.info("Continuing with remaining batches...")
            
            # 将嵌入向量添加到对应的条目
            for tm_entry, embedding in zip(tm_entries, all_embeddings):
                if embedding:  # 只在成功生成时添加
                    tm_entry['source_vector'] = embedding
                    
            success_count = sum(1 for e in all_embeddings if e)
            logger.info(f"Successfully generated {success_count}/{len(all_embeddings)} embeddings ({success_count/len(all_embeddings)*100:.1f}%)")
            if failed_batches > 0:
                logger.warning(f"Failed batches: {failed_batches}/{(len(source_texts) + batch_size - 1) // batch_size}")
        except Exception as e:
            logger.error(f"Batch embedding generation failed: {e}")
            logger.info("Continuing without embeddings...")
    
    # 批量导入
    logger.info(f"Importing {len(tm_entries)} entries to TM database...")
    success_count = tm_db.batch_add_entries(tm_entries)
    
    logger.info(f"Successfully imported {success_count}/{len(tm_entries)} entries")
    return success_count


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description='导入翻译记忆到数据库')
    parser.add_argument('input_file', type=Path, help='输入数据文件（JSON格式）')
    parser.add_argument('--source-lang', '-s', default='zh', help='源语言代码 (默认: zh)')
    parser.add_argument('--target-lang', '-t', default='en', help='目标语言代码 (默认: en)')
    parser.add_argument('--domain', '-d', default='legal', help='法律领域 (默认: legal)')
    parser.add_argument('--max-entries', '-m', type=int, default=-1, help='最大导入条数 (-1=全部)')
    parser.add_argument('--use-embeddings', action='store_true', help='生成并存储嵌入向量')
    parser.add_argument('--embedding-batch-size', type=int, default=50, help='嵌入生成批次大小 (默认: 50)')
    parser.add_argument('--tm-db-path', default='tm_bm25_index.json', help='TM BM25 索引路径')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not args.input_file.exists():
        logger.error(f"Input file not found: {args.input_file}")
        return 1
    
    # 初始化 TM 数据库
    # tm_db = TranslationMemoryDB(bm25_index_path=args.tm_db_path)
    # 初始化 TM 数据库
    # 从 .env 读取要操作的集合名称
    collection_name = os.getenv("TM_COLLECTION", "tm_zh_en_v2")  # 默认用 tm_zh_en_v2
    if collection_name == "translation_memory":
        logger.warning("警告：正在使用默认的 'translation_memory' 集合，请在 .env 中设置 TM_COLLECTION=tm_zh_en_v2")

    logger.info(f"正在使用 Milvus 集合: {collection_name}")
    tm_db = TranslationMemoryDB(collection_name=collection_name, bm25_index_path=args.tm_db_path)
    # 导入数据
    success_count = import_from_parallel_corpus(
        file_path=args.input_file,
        tm_db=tm_db,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        domain=args.domain,
        max_entries=args.max_entries,
        use_embeddings=args.use_embeddings,
        batch_size=args.embedding_batch_size
    )
    
    # 显示统计信息
    stats = tm_db.get_stats()
    print(f"\n{'='*60}")
    print(f"TM 数据库统计")
    print(f"{'='*60}")
    print(f"BM25 条目数: {stats['bm25_entries']}")
    print(f"BM25 可用: {stats['bm25_available']}")
    print(f"Milvus 可用: {stats['milvus_available']}")
    if stats['milvus_available']:
        print(f"Milvus 条目数: {stats.get('milvus_entries', 0)}")
    print(f"{'='*60}\n")
    
    return 0 if success_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())

