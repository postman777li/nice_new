"""
双语术语标准化智能体 - 去重、排序、清理（纯逻辑处理，不调用LLM）
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import logging

from ..base import BaseAgent, AgentConfig, AgentRunContext

logger = logging.getLogger(__name__)


@dataclass
class StandardizedTerm:
    """标准化后的术语"""
    source_term: str
    target_term: str
    category: str
    confidence: float
    quality_score: float
    law: str
    domain: str
    year: str
    entry_id: str
    source_context: str
    target_context: str
    combined_score: float  # 综合评分
    occurrence_count: int  # 出现次数


class BilingualTermStandardizationAgent(BaseAgent):
    """双语术语标准化智能体（纯逻辑处理）"""
    
    def __init__(self, locale: str = 'zh'):
        super().__init__(AgentConfig(
            name='preprocess:bilingual-term-standardization',
            role='bilingual_terminology_standardization',
            domain='preprocess',
            specialty='双语术语标准化清理',
            quality='review',
            locale=locale
        ))
    
    async def execute(self, input_data: Dict[str, Any], ctx: Optional[AgentRunContext] = None) -> List[Dict[str, Any]]:
        """执行术语标准化（纯逻辑，不调用LLM）"""
        terms = input_data.get('terms', [])
        max_targets_per_source = input_data.get('max_targets_per_source', 3)  # 每个source_term最多保留3个target_term
        confidence_weight = input_data.get('confidence_weight', 0.4)  # confidence权重
        quality_weight = input_data.get('quality_weight', 0.6)  # quality_score权重
        
        if not terms:
            return []
        
        logger.info(f"开始标准化处理: {len(terms)} 个术语")
        logger.info(f"配置: 每个source_term最多保留 {max_targets_per_source} 个target_term")
        logger.info(f"评分权重: confidence={confidence_weight}, quality={quality_weight}")
        
        # 步骤1: 计算综合评分
        terms_with_score = self._calculate_combined_scores(terms, confidence_weight, quality_weight)
        logger.info(f"已计算综合评分")
        
        # 步骤2: 修复无效的normalized字段
        cleaned_terms = self._clean_normalized_fields(terms_with_score)
        logger.info(f"已清理无效归一化字段")
        
        # 步骤3: 按normalized术语去重，选择最佳target_term
        deduplicated = self._deduplicate_by_normalized(cleaned_terms)
        logger.info(f"归一化去重后: {len(deduplicated)} 个术语")
        
        # 步骤4: 按source_term分组，每组最多保留N个不同的target_term
        final_terms = self._limit_targets_per_source(deduplicated, max_targets_per_source)
        logger.info(f"限制每source最多{max_targets_per_source}个target后: {len(final_terms)} 个术语")
        
        # 步骤5: 转换为最终输出格式
        output_terms = self._format_output(final_terms)
        
        # 步骤6: 按综合评分排序
        output_terms.sort(key=lambda x: (
            x['source_term'],
            -x['combined_score'],
            -x['confidence'],
            -x['quality_score']
        ))
        
        logger.info(f"标准化完成: 最终输出 {len(output_terms)} 个术语")
        
        return output_terms
    
    def _calculate_combined_scores(self, terms: List[Dict[str, Any]], conf_weight: float, qual_weight: float) -> List[Dict[str, Any]]:
        """计算综合评分 = confidence * conf_weight + quality_score * qual_weight"""
        for term in terms:
            confidence = term.get('confidence', 0.0)
            quality = term.get('quality_score', 0.0)
            combined = confidence * conf_weight + quality * qual_weight
            term['combined_score'] = combined
        return terms
    
    def _clean_normalized_fields(self, terms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """清理无效的normalized字段"""
        cleaned = []
        
        for term in terms:
            # 验证并修正normalized_source
            source_term = term.get('source_term', '')
            normalized_source = term.get('normalized_source', source_term)
            
            if not self._is_valid_normalization(source_term, normalized_source, is_english=False):
                logger.warning(f"⚠️ 无效归一化: '{source_term}' -> '{normalized_source}', 使用原始术语")
                normalized_source = source_term
            
            # 验证并修正normalized_target
            target_term = term.get('target_term', '')
            normalized_target = term.get('normalized_target', target_term)
            
            if not self._is_valid_normalization(target_term, normalized_target, is_english=True):
                logger.warning(f"⚠️ 无效归一化: '{target_term}' -> '{normalized_target}', 使用原始术语")
                normalized_target = target_term
            
            # 使用验证后的值更新术语
            term['normalized_source'] = normalized_source
            term['normalized_target'] = normalized_target
            
            cleaned.append(term)
        
        return cleaned
    
    def _is_valid_normalization(self, original: str, normalized: str, is_english: bool = False) -> bool:
        """检查归一化是否有效"""
        if not original or not normalized:
            return False
        
        if original == normalized:
            return True
        
        if is_english:
            # 英文特殊规则
            orig_lower = original.lower().strip()
            norm_lower = normalized.lower().strip()
            
            # 规则1：允许添加括号标记 (s), (es), (ies)等
            if norm_lower.replace('(s)', '').replace('(es)', '').replace('(ies)', '') == orig_lower:
                return True
            if orig_lower.replace('(s)', '').replace('(es)', '').replace('(ies)', '') == norm_lower:
                return True
            
            # 规则2：允许单复数变化（直接包含关系）
            # e.g., accomplice <-> accomplices, mediator <-> mediators
            if orig_lower in norm_lower or norm_lower in orig_lower:
                # 确保不是完全不相关的词（长度差异不能太大）
                len_diff = abs(len(orig_lower) - len(norm_lower))
                if len_diff <= max(len(orig_lower), len(norm_lower)) * 0.5:
                    return True
            
            # 规则3：检查单词重叠（放宽到20%）
            orig_words = set(orig_lower.split())
            norm_words = set(norm_lower.split())
            if orig_words and norm_words:
                overlap = len(orig_words & norm_words)
                min_words = min(len(orig_words), len(norm_words))
                if overlap >= min_words * 0.2:  # 从30%降低到20%
                    return True
            
            # 规则4：检查字符重叠（对于词性变化等，如 mediate -> mediation）
            orig_chars = set(orig_lower.replace(' ', '').replace('-', ''))
            norm_chars = set(norm_lower.replace(' ', '').replace('-', ''))
            if orig_chars and norm_chars:
                overlap = len(orig_chars & norm_chars)
                min_chars = min(len(orig_chars), len(norm_chars))
                if overlap >= min_chars * 0.5:  # 50%字符重叠
                    return True
            
            return False
        else:
            # 中文：检查字符重叠（保持30%）
            orig_chars = set(original)
            norm_chars = set(normalized)
            overlap = len(orig_chars & norm_chars)
            min_chars = min(len(orig_chars), len(norm_chars))
            return overlap >= min_chars * 0.3
    
    def _deduplicate_by_normalized(self, terms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        按normalized术语对去重，并智能合并单复数变体
        """
        # 第一步：按精确的normalized_source和normalized_target分组
        exact_groups = defaultdict(list)
        
        for term in terms:
            key = (
                term.get('normalized_source', ''),
                term.get('normalized_target', '')
            )
            exact_groups[key].append(term)
        
        # 第二步：对每组选择最佳的
        result = []
        for key, group_terms in exact_groups.items():
            # 按综合评分排序
            group_terms.sort(key=lambda x: -x['combined_score'])
            best_term = group_terms[0]
            
            # 记录出现次数和合并entry_ids
            best_term['occurrence_count'] = len(group_terms)
            all_entry_ids = set()
            for t in group_terms:
                entry_id = str(t.get('entry_id', ''))
                if entry_id:
                    all_entry_ids.update(entry_id.split(','))
            best_term['merged_entry_ids'] = ','.join(sorted(filter(None, all_entry_ids)))
            
            result.append(best_term)
        
        logger.info(f"精确去重: {len(terms)} -> {len(result)} (去掉 {len(terms) - len(result)} 个完全相同的)")
        
        # 第三步：合并同一source下 target 的单数与复合形式（例如："trade union" 与 "trade union/trade unions"）
        merged_result = self._merge_singular_with_composite_targets(result)
        if len(merged_result) != len(result):
            logger.info(f"合并单数与复合形式后: {len(result)} -> {len(merged_result)}")
        
        return merged_result

    def _merge_singular_with_composite_targets(self, terms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        在同一 normalized_source 内，将 target 为单一形式的项与其对应的
        复合形式（"singular/plural"）合并，保留复合形式且选择综合评分更高的版本。
        """
        if not terms:
            return terms
        
        # 按 normalized_source 分组
        from collections import defaultdict as _dd
        source_groups = _dd(list)
        for t in terms:
            source_key = t.get('normalized_source', t.get('source_term', ''))
            source_groups[source_key].append(t)
        
        merged_terms: List[Dict[str, Any]] = []
        
        for source_key, group in source_groups.items():
            # 1) 为每个复合 target（包含'/'）按其 parts（小写、修剪、去多空格）挑选综合评分最高的一个
            parts_to_best_composite: Dict[tuple, Dict[str, Any]] = {}
            for item in group:
                target_val = item.get('normalized_target', item.get('target_term', ''))
                normalized_target_val = ' '.join(target_val.lower().split())
                if '/' in normalized_target_val:
                    parts = tuple(p.strip() for p in normalized_target_val.split('/') if p.strip())
                    prev = parts_to_best_composite.get(parts)
                    if prev is None or item.get('combined_score', 0.0) > prev.get('combined_score', 0.0):
                        parts_to_best_composite[parts] = item
            
            # 2) 合并：若存在单一形式的 target，且恰好是某复合形式的组成部分，则将单一形式合并进该复合形式
            to_drop: set = set()
            for item in group:
                target_val = item.get('normalized_target', item.get('target_term', ''))
                normalized_target_val = ' '.join(target_val.lower().split())
                if '/' not in normalized_target_val:
                    for parts, best_comp in parts_to_best_composite.items():
                        if normalized_target_val in parts:
                            # 将该单一项合并进复合项
                            best_comp['occurrence_count'] = best_comp.get('occurrence_count', 1) + item.get('occurrence_count', 1)
                            # 合并 entry ids
                            eid_a = str(best_comp.get('merged_entry_ids', best_comp.get('entry_id', '')))
                            eid_b = str(item.get('merged_entry_ids', item.get('entry_id', '')))
                            ids = set(map(str.strip, filter(None, eid_a.split(','))))
                            ids.update(map(str.strip, filter(None, eid_b.split(','))))
                            best_comp['merged_entry_ids'] = ','.join(sorted(filter(None, ids)))
                            to_drop.add(id(item))
                            break
            
            # 3) 对于同一 parts 的多个复合形式，保留综合评分最高者，其余合并到最佳者
            for parts, best_comp in parts_to_best_composite.items():
                for item in group:
                    if id(item) == id(best_comp):
                        continue
                    target_val = item.get('normalized_target', item.get('target_term', ''))
                    normalized_target_val = ' '.join(target_val.lower().split())
                    if '/' in normalized_target_val:
                        item_parts = tuple(p.strip() for p in normalized_target_val.split('/') if p.strip())
                        if item_parts == parts:
                            # 合并同 parts 的复合项
                            best_comp['occurrence_count'] = best_comp.get('occurrence_count', 1) + item.get('occurrence_count', 1)
                            eid_a = str(best_comp.get('merged_entry_ids', best_comp.get('entry_id', '')))
                            eid_b = str(item.get('merged_entry_ids', item.get('entry_id', '')))
                            ids = set(map(str.strip, filter(None, eid_a.split(','))))
                            ids.update(map(str.strip, filter(None, eid_b.split(','))))
                            best_comp['merged_entry_ids'] = ','.join(sorted(filter(None, ids)))
                            to_drop.add(id(item))
            
            # 4) 输出分组内保留的项
            for item in group:
                if id(item) in to_drop:
                    continue
                merged_terms.append(item)
        
        return merged_terms
    
    def _limit_targets_per_source(self, terms: List[Dict[str, Any]], max_targets: int) -> List[Dict[str, Any]]:
        """
        限制每个source_term最多保留N个不同的target_term
        选择综合评分最高的N个
        """
        # 按source_term分组
        source_groups = defaultdict(list)
        
        for term in terms:
            source = term.get('normalized_source', term.get('source_term', ''))
            source_groups[source].append(term)
        
        result = []
        for source, group_terms in source_groups.items():
            # 按target_term去重（选择每个target的最佳版本）
            target_groups = defaultdict(list)
            for term in group_terms:
                target = term.get('normalized_target', term.get('target_term', ''))
                target_groups[target].append(term)
            
            # 为每个target选择最佳版本
            best_per_target = []
            for target, target_terms in target_groups.items():
                target_terms.sort(key=lambda x: -x['combined_score'])
                best_per_target.append(target_terms[0])
            
            # 按综合评分排序，保留前N个
            best_per_target.sort(key=lambda x: -x['combined_score'])
            top_n = best_per_target[:max_targets]
            
            if len(best_per_target) > max_targets:
                logger.debug(f"source_term '{source}' 有 {len(best_per_target)} 个target，保留前 {max_targets} 个")
            
            result.extend(top_n)
        
        logger.info(f"限制每source最多{max_targets}个target: {len(terms)} -> {len(result)}")
        
        return result
    
    def _format_output(self, terms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """格式化为最终输出格式（使用归一化后的术语作为主要术语，原始术语作为备份）"""
        output = []
        
        for term in terms:
            # 使用归一化后的术语作为主要的source_term和target_term
            normalized_source = term.get('normalized_source', term.get('source_term', ''))
            normalized_target = term.get('normalized_target', term.get('target_term', ''))
            original_source = term.get('source_term', '')
            original_target = term.get('target_term', '')
            
            output_term = {
                'source_term': normalized_source,
                'target_term': normalized_target,
                'original_source_term': original_source,
                'original_target_term': original_target,
                'category': term.get('category', ''),
                'confidence': round(term.get('confidence', 0.0), 2),
                'quality_score': round(term.get('quality_score', 0.0), 2),
                'combined_score': round(term.get('combined_score', 0.0), 2),
                'law': term.get('law', ''),
                'domain': term.get('domain', ''),
                'year': term.get('year', ''),
                'entry_id': term.get('merged_entry_ids', term.get('entry_id', '')),
                'source_context': term.get('source_context', ''),
                'target_context': term.get('target_context', ''),
                'occurrence_count': term.get('occurrence_count', 1)
            }
            output.append(output_term)
        
        return output

 