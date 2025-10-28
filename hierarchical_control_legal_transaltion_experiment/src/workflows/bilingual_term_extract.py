"""
双语术语提取工作流 - 从双语文本中提取平行术语翻译（包含质量检验）
"""
from typing import Dict, Any, List, Optional
import logging

# 兼容相对/绝对导入（从项目根 or backend 目录运行）
try:
    from ..models import TranslationConfig
except ImportError:  # running as top-level module
    from models import TranslationConfig

from ..agents.preprocess.bilingual_term_extract import BilingualTermExtractAgent
from ..agents.preprocess.bilingual_term_quality_check import BilingualTermQualityCheckAgent
from ..agents.preprocess.bilingual_term_normalization import TermNormalizationAgent

logger = logging.getLogger(__name__)


async def run_bilingual_term_extract_workflow(
    orchestrator, 
    job_id: str, 
    source_text: str, 
    target_text: str,
    src_lang: str = 'zh',
    tgt_lang: str = 'en'
) -> Dict[str, Any]:
    """运行双语术语提取工作流（包含质量检验）"""
    
    logger.info(f"开始双语术语提取工作流 - 作业ID: {job_id}")
    
    # 第一阶段：双语术语提取
    logger.info("第一阶段：双语术语提取")
    bi_extract_agent = BilingualTermExtractAgent(locale=src_lang)
    extract_input_data = {
        'source_text': source_text,
        'target_text': target_text,
        'src_lang': src_lang,
        'tgt_lang': tgt_lang
    }
    extracted_terms = await bi_extract_agent.run(extract_input_data, None)
    
    logger.info(f"初级提取完成，获得 {len(extracted_terms)} 个术语对")
    
    # 第二阶段：质量检验和过滤
    logger.info("第二阶段：质量检验和过滤")
    quality_check_agent = BilingualTermQualityCheckAgent(locale=src_lang)
    quality_input_data = {
        'terms': [term.__dict__ for term in extracted_terms],
        'source_text': source_text,
        'target_text': target_text,
        'src_lang': src_lang,
        'tgt_lang': tgt_lang
    }
    filtered_terms = await quality_check_agent.run(quality_input_data, None)
    
    logger.info(f"质量检验完成，过滤后保留 {len(filtered_terms)} 个高质量术语对")
    
    # 第三阶段：术语归一化
    logger.info("第三阶段：术语归一化")
    normalization_agent = TermNormalizationAgent(locale=src_lang)
    normalization_input_data = {
        'terms': [term.__dict__ for term in filtered_terms],
        'src_lang': src_lang,
        'tgt_lang': tgt_lang
    }
    normalized_terms = await normalization_agent.run(normalization_input_data, None)
    
    logger.info(f"术语归一化完成，归一化后保留 {len(normalized_terms)} 个标准化术语对")
    
    return {
        "extractedTerms": [item.__dict__ for item in normalized_terms],
        "rawExtractedTerms": [item.__dict__ for item in extracted_terms],  # 原始提取结果
        "filteredTerms": [item.__dict__ for item in filtered_terms],  # 质量过滤后的结果
        "normalizedTerms": [item.__dict__ for item in normalized_terms],  # 归一化后的结果
        "qualityStats": {
            "totalExtracted": len(extracted_terms),
            "totalFiltered": len(filtered_terms),
            "totalNormalized": len(normalized_terms),
            "filterRate": (len(extracted_terms) - len(filtered_terms)) / len(extracted_terms) if extracted_terms else 0,
            "normalizationRate": (len(filtered_terms) - len(normalized_terms)) / len(filtered_terms) if filtered_terms else 0
        },
        "sourceText": source_text,
        "targetText": target_text,
        "srcLang": src_lang,
        "tgtLang": tgt_lang
    }


async def run_bilingual_term_extract_from_config(
    orchestrator, 
    job_id: str, 
    config: TranslationConfig,
    target_text: Optional[str] = None
) -> Dict[str, Any]:
    """从配置对象运行双语术语提取工作流"""
    
    if not target_text:
        raise ValueError("目标文本不能为空")
    
    return await run_bilingual_term_extract_workflow(
        orchestrator=orchestrator,
        job_id=job_id,
        source_text=config.source,
        target_text=target_text,
        src_lang=config.src_lang,
        tgt_lang=config.tgt_lang
    )
