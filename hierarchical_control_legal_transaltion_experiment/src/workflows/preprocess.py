"""
文档预处理工作流 - 文档级术语预提取
"""
from typing import Dict, Any

# 兼容相对/绝对导入（从项目根 or backend 目录运行）
try:
    from ..models import TranslationConfig
except ImportError:  # running as top-level module
    from models import TranslationConfig

from ..agents.preprocess.document_term_extract_simple import DocumentTermExtractAgent, DocumentTermExtractOptions


async def run_preprocess_workflow(orchestrator, job_id: str, config: TranslationConfig, verbose: bool = False) -> Dict[str, Any]:
    """运行文档预处理工作流：文档级术语预提取"""
    text = config.source
    src_lang = config.src_lang
    tgt_lang = config.tgt_lang

    if verbose:
        print(f"   📝 预处理：提取文档级术语...")
        print(f"   文档长度: {len(text)} 字符")

    # 文档级术语预提取
    term_extract_agent = DocumentTermExtractAgent(locale=src_lang)
    options = DocumentTermExtractOptions(max_terms=200)
    extracted_terms = await term_extract_agent.run({"text": text, "options": options}, None)

    if verbose:
        print(f"   提取到 {len(extracted_terms)} 个文档级术语")
        if extracted_terms:
            print(f"\n   【详细】前10个文档术语:")
            for i, term in enumerate(extracted_terms[:10], 1):
                print(f"   {i}. {term.term} (重要性: {term.importance:.2f})")
            if len(extracted_terms) > 10:
                print(f"   ... 还有 {len(extracted_terms) - 10} 个术语\n")

    return {
        "extractedTerms": [item.__dict__ for item in extracted_terms],
        "output": text
    }