#!/usr/bin/env python3
"""
术语导入脚本 - 将提取的术语导入SQLite数据库
"""
import json
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.lib.term_db import TermDatabase, Term, import_terms_from_dict

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def import_terms_from_json(json_file: str, db_path: str = "terms.db", force_recreate: bool = False) -> Dict[str, Any]:
    """
    从JSON文件导入术语到数据库
    
    Args:
        json_file: JSON文件路径
        db_path: 数据库文件路径
        force_recreate: 是否强制重建数据库
    
    Returns:
        导入统计信息
    """
    logger.info(f"开始从 {json_file} 导入术语")
    
    # 如果强制重建，删除现有数据库
    if force_recreate:
        db_file = Path(db_path)
        if db_file.exists():
            db_file.unlink()
            logger.info(f"已删除现有数据库: {db_path}")
    
    # 读取JSON文件
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"读取JSON文件失败: {e}")
        return {"success": False, "error": str(e)}
    
    # 解析元数据和术语列表（支持两种格式）
    if isinstance(data, list):
        # 格式1：直接是术语列表
        terms_list = data
        source_lang = 'zh'  # 默认值
        target_lang = 'en'  # 默认值
        stage = 'unknown'
        # 尝试从第一个术语推断语言
        if terms_list:
            first_term = terms_list[0]
            source_lang = first_term.get('source_lang', source_lang)
            target_lang = first_term.get('target_lang', target_lang)
        total_terms = len(terms_list)
        logger.info(f"检测到格式1 (术语列表): {len(terms_list)} 个术语")
    else:
        # 格式2：包含metadata的对象
        metadata = data.get('metadata', {})
        source_lang = metadata.get('source_lang', 'zh')
        target_lang = metadata.get('target_lang', 'en')
        total_terms = metadata.get('total_terms', 0)
        stage = metadata.get('stage', 'unknown')
        terms_list = data.get('terms', [])
        logger.info(f"检测到格式2 (带metadata): {len(terms_list)} 个术语")
    
    logger.info(f"语言对: {source_lang} -> {target_lang}, 预期术语数: {total_terms}, 阶段: {stage if 'stage' in locals() else 'N/A'}")
    if not terms_list:
        logger.warning("JSON文件中没有术语")
        return {"success": False, "error": "No terms found"}
    
    logger.info(f"找到 {len(terms_list)} 个术语")
    
    # 初始化数据库
    db = TermDatabase(db_path)
    
    # 导入术语
    logger.info("开始导入术语...")
    added_count = import_terms_from_dict(terms_list, source_lang, target_lang, db_path)
    
    # 获取统计信息
    stats = db.get_term_stats()
    
    logger.info(f"✅ 导入完成！成功导入 {added_count} 个术语")
    logger.info(f"数据库统计: {stats}")
    
    return {
        "success": True,
        "added_count": added_count,
        "total_in_file": len(terms_list),
        "source_lang": source_lang,
        "target_lang": target_lang,
        "stage": stage,
        "database_stats": stats
    }


def main():
    parser = argparse.ArgumentParser(description='导入术语到SQLite数据库')
    parser.add_argument('json_file', help='术语JSON文件路径')
    parser.add_argument('-d', '--database', default='terms.db', help='数据库文件路径 (默认: terms.db)')
    parser.add_argument('-f', '--force', action='store_true', help='强制重建数据库')
    parser.add_argument('-o', '--output-dir', default=None, help='数据库输出目录（默认为当前目录）')
    
    args = parser.parse_args()
    
    # 确定数据库路径
    if args.output_dir:
        db_path = Path(args.output_dir) / args.database
        db_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        db_path = Path(args.database)
    
    logger.info("=" * 80)
    logger.info("术语导入工具")
    logger.info("=" * 80)
    logger.info(f"JSON文件: {args.json_file}")
    logger.info(f"数据库: {db_path}")
    logger.info(f"强制重建: {args.force}")
    logger.info("=" * 80)
    
    # 导入术语
    result = import_terms_from_json(args.json_file, str(db_path), args.force)
    
    if result.get('success'):
        logger.info("")
        logger.info("=" * 80)
        logger.info("导入摘要")
        logger.info("=" * 80)
        logger.info(f"文件中的术语数: {result['total_in_file']}")
        logger.info(f"成功导入: {result['added_count']}")
        logger.info(f"语言对: {result['source_lang']} -> {result['target_lang']}")
        logger.info(f"处理阶段: {result['stage']}")
        logger.info("")
        
        db_stats = result.get('database_stats', {})
        logger.info("数据库统计:")
        logger.info(f"  总术语数: {db_stats.get('total_terms', 0)}")
        
        lang_pairs = db_stats.get('language_pairs', [])
        if lang_pairs:
            logger.info("  语言对分布:")
            for lp in lang_pairs:
                logger.info(f"    {lp['source']} -> {lp['target']}: {lp['count']} 个术语")
        
        domains = db_stats.get('domains', [])
        if domains:
            logger.info("  领域分布 (前10):")
            for domain in domains[:10]:
                logger.info(f"    {domain['domain']}: {domain['count']} 个术语")
        
        logger.info("=" * 80)
        logger.info(f"✅ 导入成功！数据库文件: {db_path}")
    else:
        logger.error(f"❌ 导入失败: {result.get('error')}")
        sys.exit(1)


if __name__ == '__main__':
    main()

