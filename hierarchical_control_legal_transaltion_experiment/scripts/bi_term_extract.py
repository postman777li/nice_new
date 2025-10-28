#!/usr/bin/env python3
"""
多并发双语术语提取脚本 - 不改变现有Agent架构，通过并发控制实现高并发
"""
import json
import argparse
import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import sys
import os
import aiofiles
from dataclasses import dataclass
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.agents.preprocess.bilingual_term_extract import BilingualTermExtractAgent
from backend.agents.preprocess.bilingual_term_quality_check import BilingualTermQualityCheckAgent
from backend.agents.preprocess.bilingual_term_normalization import TermNormalizationAgent
from backend.agents.preprocess.bilingual_term_standardization import BilingualTermStandardizationAgent

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ConcurrentConfig:
    """并发配置"""
    batch_size: int = 10   # 每批处理的条目数
    max_concurrent_requests: int = 40  # 最大并发请求数
    timeout: int = 300    # 超时时间（秒）
    save_interval: int = 2  # 每处理多少批次保存一次中间结果
    checkpoint_file: str = "checkpoint.json"  # 检查点文件
    stage_dir: Optional[str] = None  # 分阶段快照输出目录（默认与输出文件同目录）
    
    # Agent 批量处理配置
    extraction_batch_size: int = 5      # 术语提取：每次处理的法条对数量（推荐 5-10）
    quality_check_batch_size: int = 10  # 质量检验：每次处理的术语数量（推荐 8-15）
    normalization_batch_size: int = 10  # 术语归一化：每次处理的术语数量（推荐 10-20）
    
    # 标准化配置
    max_targets_per_source: int = 3     # 每个source_term最多保留的target_term数量
    confidence_weight: float = 0.4      # confidence权重
    quality_weight: float = 0.6         # quality_score权重


def round_floats(obj, decimals=2):
    """递归地将所有浮点数四舍五入到指定小数位数"""
    if isinstance(obj, float):
        return round(obj, decimals)
    elif isinstance(obj, dict):
        return {k: round_floats(v, decimals) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [round_floats(item, decimals) for item in obj]
    else:
        return obj


class ConcurrentBilingualTermExtractor:
    """多并发双语术语提取器（不改变现有Agent架构）"""
    
    def __init__(self, config: ConcurrentConfig):
        self.config = config
        self.semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        self.stats = {
            'total_entries': 0,         # 输入条目数
            'total_terms': 0,           # 最终术语数
            'extracted_terms': 0,       # 提取的术语数
            'filtered_terms': 0,        # 过滤后的术语数
            'normalized_terms': 0,      # 归一化后的术语数
            'start_time': None,
            'end_time': None,
            'processed_batches': 0,
            'last_save_time': None,
            # 兼容旧版本
            'total_processed': 0,
            'successful': 0
        }
        self.checkpoint_data = {
            'processed_batches': [],
            'all_terms': [],
            'all_extracted_terms': [],
            'all_filtered_terms': [],
            'all_normalized_terms': [],
            'all_standardized_terms': [],  # 阶段4：标准化后的术语
            'stats': self.stats
        }
    
    def save_checkpoint(self):
        """保存检查点"""
        try:
            self.checkpoint_data['stats'] = self.stats
            # 格式化所有浮点数到2位小数
            formatted_data = round_floats(self.checkpoint_data, decimals=2)
            with open(self.config.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(formatted_data, f, ensure_ascii=False, indent=2)
            # 同时写出分阶段快照，便于快速恢复/检查
            base = Path(self.config.checkpoint_file)
            # 计算阶段输出目录
            stage_dir = Path(self.config.stage_dir) if self.config.stage_dir else base.parent
            stage_dir.mkdir(parents=True, exist_ok=True)
            try:
                with open(stage_dir / f"{base.stem}_stage1_extracted.json", 'w', encoding='utf-8') as f1:
                    json.dump(round_floats(self.checkpoint_data.get('all_extracted_terms', [])), f1, ensure_ascii=False, indent=2)
                with open(stage_dir / f"{base.stem}_stage2_filtered.json", 'w', encoding='utf-8') as f2:
                    json.dump(round_floats(self.checkpoint_data.get('all_filtered_terms', [])), f2, ensure_ascii=False, indent=2)
                with open(stage_dir / f"{base.stem}_stage3_normalized.json", 'w', encoding='utf-8') as f3:
                    json.dump(round_floats(self.checkpoint_data.get('all_normalized_terms', [])), f3, ensure_ascii=False, indent=2)
                with open(stage_dir / f"{base.stem}_stage4_standardized.json", 'w', encoding='utf-8') as f4:
                    json.dump(round_floats(self.checkpoint_data.get('all_standardized_terms', [])), f4, ensure_ascii=False, indent=2)
                with open(stage_dir / f"{base.stem}_final_terms.json", 'w', encoding='utf-8') as f5:
                    json.dump(round_floats(self.checkpoint_data.get('all_terms', [])), f5, ensure_ascii=False, indent=2)
            except Exception as se:
                logger.warning(f"写入分阶段检查点失败: {se}")
            self.stats['last_save_time'] = time.time()
            logger.info(f"检查点已保存: {self.config.checkpoint_file}")
        except Exception as e:
            logger.error(f"保存检查点失败: {e}")
    
    def load_checkpoint(self):
        """加载检查点"""
        try:
            if os.path.exists(self.config.checkpoint_file):
                with open(self.config.checkpoint_file, 'r', encoding='utf-8') as f:
                    self.checkpoint_data = json.load(f)
                self.stats = self.checkpoint_data.get('stats', self.stats)
                logger.info(f"检查点已加载: {self.config.checkpoint_file}")
                logger.info(f"已处理批次: {len(self.checkpoint_data.get('processed_batches', []))}")
                return True
        except Exception as e:
            logger.error(f"加载检查点失败: {e}")
        return False
    
    def add_batch_result(self, batch_result: Dict[str, Any], batch_index: int):
        """添加批次结果到检查点数据"""
        if batch_result:
            self.checkpoint_data['all_terms'].extend(batch_result.get('final_terms', []))
            self.checkpoint_data['all_extracted_terms'].extend(batch_result.get('extracted_terms', []))
            self.checkpoint_data['all_filtered_terms'].extend(batch_result.get('filtered_terms', []))
            self.checkpoint_data['all_normalized_terms'].extend(batch_result.get('normalized_terms', []))
            self.checkpoint_data['all_standardized_terms'].extend(batch_result.get('standardized_terms', []))
            self.checkpoint_data['processed_batches'].append(batch_index)
            self.stats['processed_batches'] += 1
            self.stats['successful'] = len(self.checkpoint_data['all_terms'])
    
    async def extract_terms_concurrent(self, json_file_path: str, max_entries: Optional[int] = None, resume: bool = True) -> Dict[str, Any]:
        """多并发提取术语（支持断点续传）"""
        logger.info(f"开始多并发处理文件: {json_file_path}")
        self.stats['start_time'] = time.time()
        
        # 尝试加载检查点
        if resume and self.load_checkpoint():
            logger.info("从检查点恢复处理...")
            processed_batches = set(self.checkpoint_data.get('processed_batches', []))
        else:
            processed_batches = set()
        
        # 读取JSON文件
        async with aiofiles.open(json_file_path, 'r', encoding='utf-8') as f:
            content = await f.read()
            data = json.loads(content)
        
        # 从metadata中检测语言对
        metadata = data.get('metadata', {})
        lang_pair = metadata.get('pair', 'zh-en')  # 默认中英
        src_lang, tgt_lang = lang_pair.split('-')
        logger.info(f"检测到语言对: {src_lang} -> {tgt_lang}")
        
        # 保存语言对信息到配置中
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
        entries = data.get('entries', [])
        if max_entries:
            entries = entries[:max_entries]
        
        logger.info(f"共找到 {len(entries)} 条双语条目")
        
        # 分批处理
        batches = [entries[i:i + self.config.batch_size] 
                  for i in range(0, len(entries), self.config.batch_size)]
        
        logger.info(f"分为 {len(batches)} 个批次，每批 {self.config.batch_size} 条")
        
        # 过滤已处理的批次（用于第1阶段）
        remaining_batches = [(i, batch) for i, batch in enumerate(batches) if i not in processed_batches]
        logger.info(f"剩余待处理批次: {len(remaining_batches)}/{len(batches)}")

        # =========================
        # 阶段 1：术语提取（覆盖所有输入）
        # =========================
        extracted_terms_all: List[Dict[str, Any]] = self.checkpoint_data.get('all_extracted_terms', [])
        if remaining_batches:
            logger.info("开始阶段1：提取术语（覆盖所有剩余批次）")
            stage1_results = await self._stage1_extract_concurrent(remaining_batches)
            # 扁平化并合并结果
            for batch_terms in stage1_results:
                if batch_terms:
                    extracted_terms_all.extend(batch_terms)
            # 写入检查点
            self.checkpoint_data['all_extracted_terms'] = extracted_terms_all
            self.save_checkpoint()
        else:
            logger.info("阶段1跳过：所有批次已处理完成")

        # =========================
        # 阶段 2：质量检验（基于阶段1的所有结果）
        # =========================
        filtered_terms_all: List[Dict[str, Any]] = self.checkpoint_data.get('all_filtered_terms', [])
        
        # 检查阶段2是否已经完成
        if filtered_terms_all:
            logger.info(f"阶段2跳过：质量检验已完成，已有 {len(filtered_terms_all)} 个过滤术语")
        else:
            remaining_for_qc = extracted_terms_all if extracted_terms_all else self.checkpoint_data.get('all_extracted_terms', [])
            if remaining_for_qc:
                logger.info("开始阶段2：质量检验（覆盖所有输入，按条目分组）")
                stage2_results = await self._stage2_quality_concurrent(batches, remaining_for_qc)
                # ❌ 不要在这里再次extend和保存！
                # _stage2_quality_concurrent 已经在内部去重并保存到 checkpoint_data['all_filtered_terms']
                filtered_terms_all = self.checkpoint_data.get('all_filtered_terms', [])
            else:
                logger.info("阶段2跳过：无可用的提取结果")

        # =========================
        # 阶段 3：归一化（基于阶段2的所有结果）
        # =========================
        normalized_terms_all: List[Dict[str, Any]] = self.checkpoint_data.get('all_normalized_terms', [])
        
        # 检查阶段3是否已经完成
        if normalized_terms_all:
            logger.info(f"阶段3跳过：归一化已完成，已有 {len(normalized_terms_all)} 个归一化术语")
        else:
            remaining_for_norm = filtered_terms_all if filtered_terms_all else self.checkpoint_data.get('all_filtered_terms', [])
            if remaining_for_norm:
                logger.info("开始阶段3：归一化（覆盖所有过滤后的术语，以批次分块）")
                stage3_results = await self._stage3_normalize_concurrent(remaining_for_norm) 
                # 结果已经保存在checkpoint_data中，直接读取即可
                normalized_terms_all = self.checkpoint_data.get('all_normalized_terms', [])
                logger.info(f"阶段3完成，获得 {len(normalized_terms_all)} 个归一化术语")
            else:
                logger.info("阶段3跳过：无可用的过滤结果")

        # =========================
        # 阶段 4：标准化（基于阶段3的所有结果）
        # =========================
        standardized_terms_all: List[Dict[str, Any]] = self.checkpoint_data.get('all_standardized_terms', [])
        
        # 检查阶段4是否已经完成
        if standardized_terms_all:
            logger.info(f"阶段4跳过：标准化已完成，已有 {len(standardized_terms_all)} 个标准化术语")
        else:
            remaining_for_std = normalized_terms_all if normalized_terms_all else self.checkpoint_data.get('all_normalized_terms', [])
            if remaining_for_std:
                logger.info("开始阶段4：标准化（去重、排序、清理）")
                standardized_terms_all = await self._stage4_standardize(remaining_for_std)
                self.checkpoint_data['all_standardized_terms'] = standardized_terms_all
                self.save_checkpoint()
            else:
                logger.info("阶段4跳过：无可用的归一化结果")

        # 生成最终术语（使用标准化后的结果）
        all_terms = self.checkpoint_data.get('all_terms', [])
        if standardized_terms_all:
            all_terms = standardized_terms_all  # 直接使用标准化结果
            self.checkpoint_data['all_terms'] = all_terms
            self.save_checkpoint()
        elif normalized_terms_all:
            final_terms = []
            for term in normalized_terms_all:
                # term 已经是 dict（归一化阶段保存的字典）
                final_terms.append({
                    'source_term': term.get('source_term', ''),
                    'target_term': term.get('target_term', ''),
                    'normalized_source': term.get('normalized_source'),
                    'normalized_target': term.get('normalized_target'),
                    'confidence': term.get('confidence', 0.0),
                    'category': term.get('category', ''),
                    'source_context': term.get('source_context', ''),
                    'target_context': term.get('target_context', ''),
                    'quality_score': term.get('quality_score', 0.0),
                    'is_valid': term.get('is_valid', False),
                    'law': term.get('law', ''),
                    'domain': term.get('domain', ''),
                    'year': term.get('year', ''),
                    'entry_id': term.get('entry_id', ''),
                    'normalization_notes': term.get('normalization_notes')
                })
            all_terms.extend(final_terms)
            self.checkpoint_data['all_terms'] = all_terms
            self.save_checkpoint()
        
        self.stats['end_time'] = time.time()
        self.stats['total_entries'] = len(entries)  # 输入条目数
        self.stats['total_terms'] = len(all_terms)  # 最终术语数
        self.stats['extracted_terms'] = len(self.checkpoint_data.get('all_extracted_terms', []))
        self.stats['filtered_terms'] = len(self.checkpoint_data.get('all_filtered_terms', []))
        self.stats['normalized_terms'] = len(self.checkpoint_data.get('all_normalized_terms', []))
        # 修正：不再计算"失败"，因为术语数和条目数不是同一单位
        
        # 计算耗时（确保start_time不为None）
        if self.stats['start_time'] is not None and self.stats['end_time'] is not None:
            elapsed_time = self.stats['end_time'] - self.stats['start_time']
        else:
            elapsed_time = 0.0
        
        logger.info(f"多并发提取完成，共获得 {len(all_terms)} 个最终术语对")
        logger.info(f"处理统计: {self.stats.get('total_entries', 0)} 个条目 → {self.stats.get('total_terms', 0)} 个最终术语，耗时 {elapsed_time:.2f} 秒")
        logger.info(f"各阶段术语数: 提取 {self.stats.get('extracted_terms', 0)} → 过滤 {self.stats.get('filtered_terms', 0)} → 归一化 {self.stats.get('normalized_terms', 0)} → 最终 {self.stats.get('total_terms', 0)}")
        
        return {
            'final_terms': all_terms,
            'extracted_terms': self.checkpoint_data.get('all_extracted_terms', []),
            'filtered_terms': self.checkpoint_data.get('all_filtered_terms', []),
            'normalized_terms': self.checkpoint_data.get('all_normalized_terms', []),
            'standardized_terms': self.checkpoint_data.get('all_standardized_terms', []),
            'stats': self.stats
        }
    
    async def _process_batches_concurrent(self, batches: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """使用异步并发处理批次"""
        async def process_batch_with_semaphore(batch, batch_index):
            async with self.semaphore:
                return await self._process_single_batch_async(batch, batch_index)
        
        # 创建带进度条的任务
        tasks = [process_batch_with_semaphore(batch, i) for i, batch in enumerate(batches)]
        
        # 使用tqdm显示进度
        results = []
        with tqdm(total=len(tasks), desc="并发处理批次", unit="批次") as pbar:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                pbar.update(1)
                pbar.set_postfix({
                    '已完成': len(results),
                    '总批次': len(tasks)
                })
        
        # 过滤异常结果
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"批次 {i} 处理失败: {result}")
            else:
                valid_results.append(result)
        
        return valid_results
    
    async def _process_batches_concurrent_with_checkpoint(self, remaining_batches: List[Tuple[int, List[Dict[str, Any]]]]) -> List[Dict[str, Any]]:
        """使用异步并发处理批次（带检查点）"""
        async def process_batch_with_semaphore(batch_info):
            batch_index, batch = batch_info
            async with self.semaphore:
                result = await self._process_single_batch_async(batch, batch_index)
                # 立即保存到检查点
                self.add_batch_result(result, batch_index)
                return result
        
        # 创建带进度条的任务
        tasks = [process_batch_with_semaphore(batch_info) for batch_info in remaining_batches]
        
        # 使用tqdm显示进度
        results = []
        with tqdm(total=len(tasks), desc="并发处理批次", unit="批次") as pbar:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                pbar.update(1)
                
                # 定期保存检查点
                if len(results) % self.config.save_interval == 0:
                    self.save_checkpoint()
                
                pbar.set_postfix({
                    '已完成': len(results),
                    '总批次': len(tasks),
                    '累计术语': len(self.checkpoint_data['all_terms'])
                })
        
        # 最终保存检查点
        self.save_checkpoint()
        
        # 过滤异常结果
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"批次 {i} 处理失败: {result}")
            else:
                valid_results.append(result)
        
        return valid_results

    def _term_to_dict(self, term: Any) -> Dict[str, Any]:
        """将术语对象或字典统一转换为字典。"""
        if isinstance(term, dict):
            return term
        # 对象，使用 __dict__
        return getattr(term, '__dict__', {})

    def _get_attr(self, term: Any, key: str, default: Any = '') -> Any:
        """从术语对象或字典中安全获取字段。"""
        if isinstance(term, dict):
            return term.get(key, default)
        return getattr(term, key, default)

    async def _stage1_extract_concurrent(self, remaining_batches: List[Tuple[int, List[Dict[str, Any]]]]) -> List[List[Dict[str, Any]]]:
        """阶段1：对所有剩余批次执行术语提取，并行执行，周期性保存检查点。"""
        async def run_extract(batch_index: int, batch: List[Dict[str, Any]]):
            async with self.semaphore:
                extracted = await self._extract_terms_batch(batch)
                # ❌ 不要在这里保存到checkpoint！会导致重复
                # 应该在所有批次完成后统一保存并去重
                self.checkpoint_data['processed_batches'].append(batch_index)
                self.stats['processed_batches'] += 1
                return extracted

        tasks = [run_extract(i, batch) for i, batch in remaining_batches]
        results: List[List[Dict[str, Any]]] = []
        total_extracted = 0
        
        with tqdm(total=len(tasks), desc="阶段1 提取术语", unit="批次") as pbar:
            completed = 0
            for coro in asyncio.as_completed(tasks):
                res = await coro
                results.append(res)
                total_extracted += len(res)
                completed += 1
                pbar.update(1)
                pbar.set_postfix({'已完成': completed, '累计术语': total_extracted})
        
        # 所有批次完成后，统一保存结果（去重）
        all_extracted = []
        seen_pairs = set()
        for batch_result in results:
            for term in batch_result:
                term_dict = self._term_to_dict(term)
                pair = (term_dict.get('source_term', ''), term_dict.get('target_term', ''))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    all_extracted.append(term_dict)
        
        logger.info(f"提取去重: {total_extracted} -> {len(all_extracted)} 个术语")
        
        # 保存到checkpoint
        self.checkpoint_data['all_extracted_terms'] = all_extracted
        self.save_checkpoint()
        
        return results

    async def _stage2_quality_concurrent(self, batches: List[List[Dict[str, Any]]], extracted_terms_all: List[Any]) -> List[List[Dict[str, Any]]]:
        """阶段2：对所有术语按固定批次大小执行质量检验，并行执行，周期性保存检查点。
        
        注意：这里按固定批次大小处理所有术语，不考虑条目边界，以确保批次大小均匀。
        """
        if not extracted_terms_all:
            return []
        
        # 将所有术语转换为字典格式
        terms_dicts = [self._term_to_dict(t) for t in extracted_terms_all]
        
        # 按固定批次大小分块
        quality_batch_size = self.config.quality_check_batch_size
        term_chunks: List[List[Dict[str, Any]]] = [
            terms_dicts[i:i + quality_batch_size] 
            for i in range(0, len(terms_dicts), quality_batch_size)
        ]
        
        logger.info(f"质量检验：共 {len(terms_dicts)} 个术语，分成 {len(term_chunks)} 个批次，每批 {quality_batch_size} 个")
        
        async def run_qc_batch(chunk_index: int, chunk: List[Dict[str, Any]]):
            """对一个批次的术语进行质量检验"""
            async with self.semaphore:
                # 由于术语来自不同条目，我们需要为每个术语找到对应的原文
                # 这里使用一个简化策略：使用术语的上下文或者整个语料库的代表性文本
                # 更精确的方法是为每个术语单独找到对应的条目文本，但会更复杂
                
                # 收集这批术语涉及的所有条目ID
                entry_ids = set()
                for term in chunk:
                    entry_id = term.get('entry_id', '')
                    if entry_id:
                        entry_ids.add(entry_id)
                
                # 构建一个综合的上下文文本（来自相关条目）
                source_context = ""
                target_context = ""
                
                # 从batches中找到对应的条目
                for batch in batches:
                    for entry in batch:
                        if entry.get('id', '') in entry_ids:
                            source_context += entry.get(self.src_lang, '') + " "
                            target_context += entry.get(self.tgt_lang, '') + " "
                            if len(source_context) > 5000:  # 限制上下文长度
                                break
                    if len(source_context) > 5000:
                        break
                
                # 如果没有找到上下文，使用默认文本
                if not source_context:
                    source_context = "法律文本"
                    target_context = "Legal text"
                
                quality_check_agent = BilingualTermQualityCheckAgent(locale='zh')
                quality_input_data = {
                    'terms': chunk,
                    'source_text': source_context[:5000],  # 限制长度
                    'target_text': target_context[:5000],
                    'src_lang': 'zh',
                    'tgt_lang': 'en',
                    'batch_mode': True,
                    'batch_size': quality_batch_size
                }
                
                filtered_terms = await quality_check_agent.run(quality_input_data, None)
                filtered_dicts = [self._term_to_dict(t) for t in (filtered_terms or [])]
                
                # ❌ 不要在这里保存到checkpoint！会导致重复
                # 应该在所有批次完成后统一保存
                
                logger.info(f"质量检验批次 {chunk_index+1}/{len(term_chunks)}: "
                          f"处理 {len(chunk)} 个术语，通过 {len(filtered_dicts)} 个")
                
                return filtered_dicts
        
        tasks = [run_qc_batch(i, chunk) for i, chunk in enumerate(term_chunks)]
        results: List[List[Dict[str, Any]]] = []
        
        with tqdm(total=len(tasks), desc="阶段2 质量检验", unit="批次") as pbar:
            completed = 0
            total_filtered = 0
            for coro in asyncio.as_completed(tasks):
                res = await coro
                results.append(res)
                total_filtered += len(res)
                completed += 1
                pbar.update(1)
                pbar.set_postfix({'已完成': completed, '累计过滤术语': total_filtered})
        
        # 所有批次完成后，统一保存结果（去重）
        all_filtered = []
        seen_pairs = set()
        for batch_result in results:
            for term in batch_result:
                pair = (term.get('source_term', ''), term.get('target_term', ''))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    all_filtered.append(term)
        
        logger.info(f"质量检验去重: {total_filtered} -> {len(all_filtered)} 个术语")
        
        # 保存到checkpoint
        self.checkpoint_data['all_filtered_terms'] = all_filtered
        self.save_checkpoint()
        
        return results

    async def _stage3_normalize_concurrent(self, filtered_terms_all: List[Any]) -> List[List[Dict[str, Any]]]:
        """阶段3：对所有过滤术语分块归一化，并行执行，周期性保存检查点。"""
        if not filtered_terms_all:
            return []
        # 将过滤术语（可能为对象或字典）统一成字典列表
        filtered_dicts: List[Dict[str, Any]] = [self._term_to_dict(t) for t in filtered_terms_all]
        
        # 按 source_term 排序，让相同或相似的术语聚集在一起，便于归一化识别重复和变体
        logger.info(f"归一化前排序：按 source_term 对 {len(filtered_dicts)} 个术语进行排序")
        filtered_dicts.sort(key=lambda x: x.get('source_term', ''))
        logger.info(f"排序完成")
        
        # 去重：保留质量最高的前3个术语（按 source_term + target_term 组合去重）
        logger.info(f"归一化前去重：开始去重，保留每个术语对的前3个最高质量版本...")
        term_groups = {}
        for term in filtered_dicts:
            key = (term.get('source_term', ''), term.get('target_term', ''))
            if key not in term_groups:
                term_groups[key] = []
            term_groups[key].append(term)
        
        # 对每组术语按质量分数排序，保留前3个
        top_terms = []
        for key, terms in term_groups.items():
            # 按质量分数降序排序
            terms.sort(key=lambda x: x.get('quality_score', 0.0), reverse=True)
            # 保留前3个
            top_3 = terms[:3]
            # 合并 entry_id 元数据
            all_entry_ids = set()
            for term in terms:
                entry_id = term.get('entry_id', '')
                if entry_id:
                    # 确保entry_id是字符串
                    entry_id_str = str(entry_id)
                    all_entry_ids.update(entry_id_str.split(','))
            
            # 为前3个术语添加合并的 entry_id
            for term in top_3:
                term['entry_id'] = ','.join(filter(None, all_entry_ids))
            
            top_terms.extend(top_3)
        
        original_count = len([self._term_to_dict(t) for t in filtered_terms_all])
        filtered_dicts = top_terms
        logger.info(f"去重完成：{len(filtered_dicts)} 个术语（保留前3个，原始：{original_count} 个，唯一术语对：{len(term_groups)} 个）")
        
        # 再次排序（保持顺序）
        filtered_dicts.sort(key=lambda x: x.get('source_term', ''))
        
        # 分块 - 使用归一化批次大小
        chunk_size = max(1, self.config.normalization_batch_size)
        chunks: List[List[Dict[str, Any]]] = [
            filtered_dicts[i:i + chunk_size] for i in range(0, len(filtered_dicts), chunk_size)
        ]

        async def run_norm(chunk: List[Dict[str, Any]]):
            async with self.semaphore:
                normalization_agent = TermNormalizationAgent(locale='zh')
                normalization_input_data = {
                    'terms': chunk,
                    'src_lang': self.src_lang,
                    'tgt_lang': self.tgt_lang,
                    'batch_size': self.config.normalization_batch_size
                }
                normalized_terms = await normalization_agent.run(normalization_input_data, None)
                # 存为字典，并从输入 chunk 回填条目元数据
                normalized_list = (normalized_terms or [])
                normalized_dicts = [self._term_to_dict(t) for t in normalized_list]
                
                # 创建原始术语的元数据映射（基于 source_term + target_term）
                metadata_map = {}
                for src_term in chunk:
                    key = (src_term.get('source_term', ''), src_term.get('target_term', ''))
                    metadata_map[key] = {
                        'entry_id': src_term.get('entry_id', ''),
                        'law': src_term.get('law', ''),
                        'domain': src_term.get('domain', ''),
                        'year': src_term.get('year', '')
                    }
                
                # 基于术语内容匹配元数据，而不是索引
                for nt in normalized_dicts:
                    key = (nt.get('source_term', ''), nt.get('target_term', ''))
                    if key in metadata_map:
                        meta = metadata_map[key]
                        for k, v in meta.items():
                            if k not in nt or not nt[k]:
                                nt[k] = v
                    else:
                        # 如果找不到精确匹配，记录警告
                        logger.debug(f"归一化术语找不到元数据匹配: {key}")
                
                # ❌ 不要在这里保存到checkpoint！会导致重复
                # 应该在所有批次完成后统一保存
                return normalized_dicts

        tasks = [run_norm(chunk) for chunk in chunks]
        results: List[List[Dict[str, Any]]] = []
        total_normalized = 0
        
        with tqdm(total=len(tasks), desc="阶段3 归一化", unit="分块") as pbar:
            completed = 0
            for coro in asyncio.as_completed(tasks):
                res = await coro
                results.append(res)
                total_normalized += len(res)
                completed += 1
                pbar.update(1)
                pbar.set_postfix({'已完成': completed, '累计归一化': total_normalized})
        
        # 所有批次完成后，统一保存结果（去重）
        all_normalized = []
        seen_pairs = set()
        for batch_result in results:
            for term in batch_result:
                pair = (term.get('source_term', ''), term.get('target_term', ''))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    all_normalized.append(term)
        
        logger.info(f"归一化去重: {total_normalized} -> {len(all_normalized)} 个术语")
        
        # 保存到checkpoint
        self.checkpoint_data['all_normalized_terms'] = all_normalized
        self.save_checkpoint()
        
        return results
    
    async def _stage4_standardize(self, normalized_terms_all: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """阶段4：标准化（去重、排序、清理）- 纯逻辑处理，不调用LLM"""
        if not normalized_terms_all:
            return []
        
        logger.info(f"标准化：开始处理 {len(normalized_terms_all)} 个归一化术语")
        
        try:
            # 创建标准化Agent
            standardization_agent = BilingualTermStandardizationAgent(locale='zh')
            
            # 执行标准化（纯逻辑处理）
            standardization_input_data = {
                'terms': normalized_terms_all,
                'max_targets_per_source': self.config.max_targets_per_source,
                'confidence_weight': self.config.confidence_weight,
                'quality_weight': self.config.quality_weight
            }
            
            standardized_terms = await standardization_agent.execute(standardization_input_data, None)
            
            logger.info(f"标准化完成：{len(normalized_terms_all)} -> {len(standardized_terms)} 个术语")
            logger.info(f"减少了 {len(normalized_terms_all) - len(standardized_terms)} 个术语 "
                       f"({(1 - len(standardized_terms)/len(normalized_terms_all))*100:.2f}% 压缩率)")
            
            return standardized_terms
            
        except Exception as e:
            logger.error(f"标准化时出错: {e}")
            # 如果标准化失败，返回原始归一化结果
            logger.warning("标准化失败，使用归一化结果")
            return normalized_terms_all
    
    
    async def _process_single_batch_async(self, batch: List[Dict[str, Any]], batch_index: int = 0) -> Dict[str, Any]:
        """异步处理单个批次 - 分阶段处理"""
        batch_results = {
            'extracted_terms': [],
            'filtered_terms': [],
            'normalized_terms': [],
            'final_terms': []
        }
        
        # 第一阶段：批量提取术语
        logger.info(f"批次 {batch_index}: 开始提取术语...")
        extracted_terms_all = await self._extract_terms_batch(batch)
        batch_results['extracted_terms'] = [term.__dict__ for term in extracted_terms_all]
        logger.info(f"批次 {batch_index}: 提取了 {len(extracted_terms_all)} 个术语")
        
        # 第二阶段：批量质量检验
        if extracted_terms_all:
            logger.info(f"批次 {batch_index}: 开始质量检验...")
            filtered_terms_all = await self._quality_check_batch(extracted_terms_all, batch)
            batch_results['filtered_terms'] = [term.__dict__ for term in filtered_terms_all]
            logger.info(f"批次 {batch_index}: 质量检验后剩余 {len(filtered_terms_all)} 个术语")
        else:
            filtered_terms_all = []
            batch_results['filtered_terms'] = []
        
        # 第三阶段：批量归一化
        if filtered_terms_all:
            logger.info(f"批次 {batch_index}: 开始归一化...")
            normalized_terms_all = await self._normalize_terms_batch(filtered_terms_all)
            batch_results['normalized_terms'] = [term.__dict__ for term in normalized_terms_all]
            logger.info(f"批次 {batch_index}: 归一化后得到 {len(normalized_terms_all)} 个术语")
            
            # 转换为最终格式
            final_terms = []
            for term in normalized_terms_all:
                term_dict = {
                    'source_term': term.source_term,
                    'target_term': term.target_term,
                    'normalized_source': term.normalized_source,
                    'normalized_target': term.normalized_target,
                    'confidence': term.confidence,
                    'category': term.category,
                    'source_context': term.source_context,
                    'target_context': term.target_context,
                    'quality_score': term.quality_score,
                    'is_valid': term.is_valid,
                    'normalization_notes': term.normalization_notes
                }
                final_terms.append(term_dict)
            batch_results['final_terms'] = final_terms
        else:
            batch_results['normalized_terms'] = []
            batch_results['final_terms'] = []
        
        return batch_results
    
    async def _extract_terms_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """第一阶段：批量提取术语；返回带条目元数据的字典列表。
        
        使用 extraction_batch_size 控制每次批量处理的法条对数量。
        例如：extraction_batch_size=3 表示每次向 LLM 发送 3 个法条对进行批量提取。
        """
        all_extracted_terms: List[Dict[str, Any]] = []
        
        # 按 extraction_batch_size 分组条目
        extraction_batch_size = self.config.extraction_batch_size
        
        # 将条目按 extraction_batch_size 分成小批次
        for i in range(0, len(batch), extraction_batch_size):
            mini_batch = batch[i:i + extraction_batch_size]
            
            # 过滤掉空文本的条目
            # 支持两种格式：(1) 使用语言代码键 zh/en/ja  (2) 使用通用键 source/target
            valid_entries = []
            for entry in mini_batch:
                src_text = entry.get(self.src_lang, '') or entry.get('source', '')
                tgt_text = entry.get(self.tgt_lang, '') or entry.get('target', '')
                if src_text and tgt_text:
                    # 确保entry包含标准化的语言键（用于后续处理）
                    entry[self.src_lang] = src_text
                    entry[self.tgt_lang] = tgt_text
                    valid_entries.append(entry)
            
            if not valid_entries:
                continue
            
            try:
                bi_extract_agent = BilingualTermExtractAgent(locale='zh')
                
                # 如果只有一个条目，使用单个模式
                if len(valid_entries) == 1:
                    entry = valid_entries[0]
                    extract_input_data = {
                        'source_text': entry.get(self.src_lang, ''),
                        'target_text': entry.get(self.tgt_lang, ''),
                        'src_lang': self.src_lang,
                        'tgt_lang': self.tgt_lang
                    }
                    extracted_terms = await bi_extract_agent.run(extract_input_data, None)
                    
                    # 转换为字典并附带条目元数据
                    for t in (extracted_terms or []):
                        t_dict = self._term_to_dict(t)
                        t_dict.setdefault('entry_id', entry.get('id', ''))
                        t_dict.setdefault('law', entry.get('law', ''))
                        t_dict.setdefault('domain', entry.get('domain', ''))
                        t_dict.setdefault('year', entry.get('year', ''))
                        all_extracted_terms.append(t_dict)
                
                # 如果有多个条目，使用批量模式
                else:
                    text_pairs = [
                        {
                            'source_text': entry.get(self.src_lang, ''),
                            'target_text': entry.get(self.tgt_lang, ''),
                            'entry_id': entry.get('id', ''),
                            'law': entry.get('law', ''),
                            'domain': entry.get('domain', ''),
                            'year': entry.get('year', '')
                        }
                        for entry in valid_entries
                    ]
                    
                    extract_input_data = {
                        'text_pairs': text_pairs,
                        'src_lang': self.src_lang,
                        'tgt_lang': self.tgt_lang,
                        'batch_mode': True,
                        'batch_size': extraction_batch_size
                    }
                    
                    extracted_terms = await bi_extract_agent.run(extract_input_data, None)
                    
                    # 转换为字典并附带条目元数据
                    # 由于批量提取无法精确匹配每个术语到具体条目，使用简单策略：
                    # 为所有提取的术语添加第一个条目的元数据（可以后续通过文本匹配优化）
                    for t in (extracted_terms or []):
                        t_dict = self._term_to_dict(t)
                        # 尝试根据源术语匹配到具体条目
                        matched_entry = None
                        source_term = t_dict.get('source_term', '')
                        for entry in valid_entries:
                            if source_term in entry.get(self.src_lang, ''):
                                matched_entry = entry
                                break
                        
                        # 如果没有匹配到，使用第一个条目的元数据
                        if not matched_entry:
                            matched_entry = valid_entries[0]
                        
                        t_dict.setdefault('entry_id', matched_entry.get('id', ''))
                        t_dict.setdefault('law', matched_entry.get('law', ''))
                        t_dict.setdefault('domain', matched_entry.get('domain', ''))
                        t_dict.setdefault('year', matched_entry.get('year', ''))
                        all_extracted_terms.append(t_dict)
                    
                    logger.info(f"批量提取: 处理 {len(valid_entries)} 个法条对，提取 {len(extracted_terms or [])} 个术语")
                
            except Exception as e:
                logger.error(f"批量提取术语时出错: {e}")
                continue
        
        return all_extracted_terms
    
    async def _quality_check_batch(self, extracted_terms: List, batch: List[Dict[str, Any]]) -> List:
        """第二阶段：批量质量检验
        
        使用 quality_check_batch_size 控制每次批量检验的术语数量。
        例如：quality_check_batch_size=10 表示每次向 LLM 发送最多 10 个术语进行质量检验。
        """
        if not extracted_terms:
            return []
        
        all_filtered_terms = []
        quality_check_batch_size = self.config.quality_check_batch_size
        
        # 按条目分组术语
        terms_by_entry = {}
        for i, entry in enumerate(batch):
            src_text = entry.get(self.src_lang, '')
            tgt_text = entry.get(self.tgt_lang, '')
            if src_text and tgt_text:
                # 找到属于该条目的术语
                entry_terms = [term for term in extracted_terms if self._is_term_from_entry(term, src_text, tgt_text)]
                terms_by_entry[i] = entry_terms
        
        # 对每个条目的术语进行质量检验
        for i, entry_terms in terms_by_entry.items():
            if not entry_terms:
                continue
                
            try:
                entry = batch[i]
                src_text = entry.get(self.src_lang, '')
                tgt_text = entry.get(self.tgt_lang, '')
                
                # 如果术语数量超过 quality_check_batch_size，分批处理
                for batch_start in range(0, len(entry_terms), quality_check_batch_size):
                    batch_end = min(batch_start + quality_check_batch_size, len(entry_terms))
                    terms_chunk = entry_terms[batch_start:batch_end]
                    
                    quality_check_agent = BilingualTermQualityCheckAgent(locale='zh')
                    quality_input_data = {
                        'terms': [term.__dict__ if hasattr(term, '__dict__') else term for term in terms_chunk],
                        'source_text': src_text,
                        'target_text': tgt_text,
                        'src_lang': self.src_lang,
                        'tgt_lang': self.tgt_lang,
                        'batch_mode': True,  # 启用批量模式
                        'batch_size': quality_check_batch_size
                    }
                    filtered_terms = await quality_check_agent.run(quality_input_data, None)
                    
                    if filtered_terms:
                        all_filtered_terms.extend(filtered_terms)
                    
                    logger.info(f"质量检验: 条目 {i+1}, 批次 {batch_start//quality_check_batch_size + 1}, "
                              f"检验 {len(terms_chunk)} 个术语，通过 {len(filtered_terms or [])} 个")
                
            except Exception as e:
                logger.error(f"质量检验时出错: {e}")
                continue
        
        return all_filtered_terms
    
    async def _normalize_terms_batch(self, filtered_terms: List) -> List:
        """第三阶段：批量归一化
        
        使用 normalization_batch_size 控制每次归一化的术语数量。
        例如：normalization_batch_size=10 表示每次向 LLM 发送最多 10 个术语进行归一化。
        注意：这个方法通常被 _stage3_normalize_concurrent 调用，那里已经做了分块处理。
        """
        if not filtered_terms:
            return []
        
        try:
            normalization_agent = TermNormalizationAgent(locale='zh')
            normalization_input_data = {
                'terms': [term.__dict__ for term in filtered_terms],
                'src_lang': 'zh',
                'tgt_lang': 'en',
                'batch_size': self.config.normalization_batch_size
            }
            normalized_terms = await normalization_agent.run(normalization_input_data, None)
            logger.info(f"归一化: 处理 {len(filtered_terms)} 个术语，归一化后得到 {len(normalized_terms or [])} 个")
            return normalized_terms or []
        except Exception as e:
            logger.error(f"归一化时出错: {e}")
            return []
    
    def _is_term_from_entry(self, term, zh_text: str, en_text: str) -> bool:
        """判断术语是否来自特定条目"""
        t = self._term_to_dict(term)
        source = t.get('source_term', '')
        target = t.get('target_term', '')
        return (source in zh_text and target in en_text)
    
    
    def save_results(self, results: Dict[str, Any], output_file: str):
        """保存结果到文件"""
        timestamp = int(time.time())
        
        # 确保输出文件路径是文件而不是目录
        output_path = Path(output_file)
        if output_path.is_dir():
            # 如果是目录，在目录下创建默认文件名
            output_file = output_path / f"extracted_terms_{timestamp}.json"
            logger.warning(f"输出路径是目录，使用默认文件名: {output_file}")
        
        # 确保输出目录存在
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        base_name = output_path.stem
        base_dir = output_path.parent
        
        # 保存最终结果
        final_data = {
            'metadata': {
                'source_lang': 'zh',
                'target_lang': 'en',
                'total_terms': len(results['final_terms']),
                'extraction_time': timestamp,
                'processing_config': {
                    'batch_size': self.config.batch_size,
                    'max_concurrent_requests': self.config.max_concurrent_requests,
                    'save_interval': self.config.save_interval
                },
                'stats': results['stats']
            },
            'terms': results['final_terms']
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"结果已保存到: {output_file}")
        
        # 保存中间结果
        self._save_intermediate_results(results, base_dir, base_name, timestamp)
    
    def _save_intermediate_results(self, results: Dict[str, Any], base_dir: Path, base_name: str, timestamp: int):
        """保存中间结果"""
        stages = [
            ('extracted', results['extracted_terms']),
            ('filtered', results['filtered_terms']),
            ('normalized', results['normalized_terms']),
            ('standardized', results['standardized_terms'])
        ]
        
        for stage_name, terms in stages:
            stage_file = base_dir / f"{base_name}_{stage_name}_{timestamp}.json"
            stage_data = {
                'metadata': {
                    'source_lang': 'zh',
                    'target_lang': 'en',
                    'total_terms': len(terms),
                    'stage': stage_name,
                    'extraction_time': timestamp
                },
                'terms': terms
            }
            
            with open(stage_file, 'w', encoding='utf-8') as f:
                json.dump(stage_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"{stage_name} 结果已保存到: {stage_file}")




async def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='多并发双语术语提取（4阶段流程）',
        epilog="""
使用示例:
  # 完整运行所有阶段
  python %(prog)s input.json -o output.json
  
  # 从阶段3（归一化）开始重新运行
  python %(prog)s input.json -o output.json --start-from-stage 3
  
  # 从阶段4（标准化）开始，调整参数
  python %(prog)s input.json -o output.json --start-from-stage 4 --max-targets-per-source 5

阶段说明:
  阶段1: 术语提取 (Extraction)
  阶段2: 质量检验 (Quality Check)
  阶段3: 归一化 (Normalization) - 按source_term分组归一化target_term
  阶段4: 标准化 (Standardization) - 去重、排序、清理
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('input_file', help='输入的双语JSON文件路径')
    parser.add_argument('-o', '--output', help='输出文件路径', default='concurrent_extracted_terms_v2.json')
    parser.add_argument('--max-entries', type=int, help='最大处理条目数（用于测试）')
    parser.add_argument('--batch-size', type=int, default=20, help='批处理大小')
    parser.add_argument('--max-concurrent', type=int, default=40, help='最大远端LLM并发请求数')
    parser.add_argument('--timeout', type=int, default=300, help='超时时间（秒）')
    parser.add_argument('--save-interval', type=int, default=5, help='每处理多少批次保存一次检查点')
    parser.add_argument('--checkpoint', type=str, default='outputs/checkpoint.json', help='检查点文件路径（默认写入 outputs/）')
    parser.add_argument('--stage-dir', type=str, help='分阶段快照输出目录（默认与checkpoint同目录）')
    parser.add_argument('--no-resume', action='store_true', help='不恢复检查点，重新开始')
    parser.add_argument('--clean-checkpoint', action='store_true', help='清理检查点文件')
    parser.add_argument('--start-from-stage', type=int, choices=[1, 2, 3, 4], 
                       help='从指定阶段开始执行（1=提取, 2=质量检验, 3=归一化, 4=标准化），会清空该阶段及之后的数据')
    
    # Agent 批量处理配置
    parser.add_argument('--extraction-batch-size', type=int, default=3, help='术语提取批次大小（每次处理的法条对数量，推荐 3-5，默认 5）')
    parser.add_argument('--quality-check-batch-size', type=int, default=30, help='质量检验批次大小（每次处理的术语数量，推荐 30-50，默认 50）')
    parser.add_argument('--normalization-batch-size', type=int, default=50, help='归一化批次大小（每次处理的术语数量，推荐 30-50，默认 50）')
    
    # 标准化配置
    parser.add_argument('--max-targets-per-source', type=int, default=3, help='每个source_term最多保留的target_term数量（默认 3）')
    parser.add_argument('--confidence-weight', type=float, default=0.4, help='confidence权重（默认 0.4）')
    parser.add_argument('--quality-weight', type=float, default=0.6, help='quality_score权重（默认 0.6）')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input_file):
        logger.error(f"输入文件不存在: {args.input_file}")
        return
    
    # 清理检查点文件
    if args.clean_checkpoint:
        if os.path.exists(args.checkpoint):
            os.remove(args.checkpoint)
            logger.info(f"检查点文件已清理: {args.checkpoint}")
        else:
            logger.info("检查点文件不存在，无需清理")
        return
    
    # 创建并发配置
    config = ConcurrentConfig(
        batch_size=args.batch_size,
        timeout=args.timeout,
        max_concurrent_requests=args.max_concurrent,
        save_interval=args.save_interval,
        checkpoint_file=args.checkpoint,
        stage_dir=args.stage_dir,
        extraction_batch_size=args.extraction_batch_size,
        quality_check_batch_size=args.quality_check_batch_size,
        normalization_batch_size=args.normalization_batch_size,
        max_targets_per_source=args.max_targets_per_source,
        confidence_weight=args.confidence_weight,
        quality_weight=args.quality_weight
    )
    
    # 显示配置信息
    logger.info(f"Agent 批量处理配置:")
    logger.info(f"  - 术语提取批次大小: {config.extraction_batch_size} 个法条对")
    logger.info(f"  - 质量检验批次大小: {config.quality_check_batch_size} 个术语")
    logger.info(f"  - 归一化批次大小: {config.normalization_batch_size} 个术语")
    logger.info(f"标准化配置:")
    logger.info(f"  - 每个source_term最多保留: {config.max_targets_per_source} 个target_term")
    logger.info(f"  - Confidence权重: {config.confidence_weight}")
    logger.info(f"  - Quality权重: {config.quality_weight}")
    
    # 创建提取器
    extractor = ConcurrentBilingualTermExtractor(config)
    
    # 处理 --start-from-stage 参数
    if args.start_from_stage:
        stage = args.start_from_stage
        logger.info(f"⚠️  强制从阶段{stage}开始执行，将清空阶段{stage}及之后的数据")
        
        # 尝试加载现有检查点
        if extractor.load_checkpoint():
            stage_names = {
                1: "提取（Extraction）",
                2: "质量检验（Quality Check）",
                3: "归一化（Normalization）",
                4: "标准化（Standardization）"
            }
            
            # 根据起始阶段清空数据
            if stage <= 1:
                extractor.checkpoint_data['processed_batches'] = []
                extractor.checkpoint_data['all_extracted_terms'] = []
                extractor.checkpoint_data['all_filtered_terms'] = []
                extractor.checkpoint_data['all_normalized_terms'] = []
                extractor.checkpoint_data['all_standardized_terms'] = []
                extractor.checkpoint_data['all_terms'] = []
                logger.info(f"  ✓ 清空所有阶段数据")
            elif stage == 2:
                extractor.checkpoint_data['all_filtered_terms'] = []
                extractor.checkpoint_data['all_normalized_terms'] = []
                extractor.checkpoint_data['all_standardized_terms'] = []
                extractor.checkpoint_data['all_terms'] = []
                logger.info(f"  ✓ 保留阶段1，清空阶段2/3/4")
            elif stage == 3:
                extractor.checkpoint_data['all_normalized_terms'] = []
                extractor.checkpoint_data['all_standardized_terms'] = []
                extractor.checkpoint_data['all_terms'] = []
                logger.info(f"  ✓ 保留阶段1/2，清空阶段3/4")
            elif stage == 4:
                extractor.checkpoint_data['all_standardized_terms'] = []
                extractor.checkpoint_data['all_terms'] = []
                logger.info(f"  ✓ 保留阶段1/2/3，清空阶段4")
            
            # 保存修改后的检查点
            extractor.save_checkpoint()
            logger.info(f"  ✓ 检查点已更新")
        else:
            logger.warning(f"  未找到检查点文件，将从头开始执行")
    
    try:
        # 提取术语（支持断点续传）
        resume = not args.no_resume
        results = await extractor.extract_terms_concurrent(args.input_file, args.max_entries, resume=resume)
        
        # 保存结果
        extractor.save_results(results, args.output)
        
        # 保留检查点与分阶段快照，便于后续查看/手工汇总
        logger.info(f"检查点与分阶段快照已保留在: {args.checkpoint} （阶段目录: {args.stage_dir or Path(args.checkpoint).parent}）")
        
        # 显示统计信息
        stats = results['stats']
        logger.info(f"多并发提取完成！")
        logger.info(f"输入条目数: {stats.get('total_entries', stats.get('total_processed', 0))}")
        logger.info(f"最终术语数: {stats.get('total_terms', stats.get('successful', 0))}")
        logger.info(f"\n各阶段统计:")
        logger.info(f"  Stage 1 (术语提取): {stats.get('extracted_terms', 0)} 个")
        logger.info(f"  Stage 2 (质量检验): {stats.get('filtered_terms', 0)} 个")
        logger.info(f"  Stage 3 (归一化): {stats.get('normalized_terms', 0)} 个")
        logger.info(f"  Stage 4 (标准化): {stats.get('total_terms', 0)} 个（最终）")
        
        # 计算耗时和速度（确保时间值不为None）
        if stats.get('start_time') and stats.get('end_time'):
            elapsed = stats['end_time'] - stats['start_time']
            logger.info(f"总耗时: {elapsed:.2f} 秒")
            total_entries = stats.get('total_entries', stats.get('total_processed', 0))
            if elapsed > 0 and total_entries > 0:
                logger.info(f"平均速度: {total_entries / elapsed:.2f} 条目/秒")
                logger.info(f"术语生成速度: {stats.get('total_terms', 0) / elapsed:.2f} 术语/秒")
        else:
            logger.info(f"总耗时: N/A (从检查点恢复)")
        
        # 显示前几个术语作为示例
        if results['final_terms']:
            logger.info("前5个最终术语示例:")
            for i, term in enumerate(results['final_terms'][:5]):
                normalized_info = f" (归一化: {term.get('normalized_source', 'N/A')} -> {term.get('normalized_target', 'N/A')})" if 'normalized_source' in term else ""
                logger.info(f"  {i+1}. {term['source_term']} -> {term['target_term']}{normalized_info} (置信度: {term['confidence']:.2f}, 质量分数: {term.get('quality_score', 'N/A')})")
    
    except KeyboardInterrupt:
        logger.info("用户中断处理，检查点已保存，可以稍后使用 --resume 恢复")
        extractor.save_checkpoint()
    except Exception as e:
        logger.error(f"多并发提取过程中出错: {e}")
        logger.info("检查点已保存，可以稍后使用 --resume 恢复")
        extractor.save_checkpoint()
        raise


if __name__ == '__main__':
    asyncio.run(main())
