"""
指标计算模块
"""
import re
import json
from typing import List, Dict, Any, Tuple
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class LegalTranslationMetrics:
    """法律翻译指标计算器"""
    
    def __init__(self):
        # 情态动词映射表
        self.deontic_mapping = {
            'zh': {
                '必须': 'must',
                '须': 'must', 
                '应当': 'should',
                '可以': 'may',
                '不得': 'shall_not',
                '禁止': 'prohibit'
            },
            'en': {
                'must': '必须',
                'shall': '必须',
                'should': '应当',
                'may': '可以',
                'shall not': '不得',
                'must not': '不得'
            }
        }
        
        # 条件连接词
        self.conditional_markers = {
            'zh': ['如果', '若', '假如', '倘若', '除非', '但书'],
            'en': ['if', 'unless', 'provided that', 'in case', 'when', 'where']
        }
    
    def calculate_termbase_accuracy(self, source: str, target: str, term_table: List[Dict]) -> float:
        """计算术语准确性"""
        if not term_table:
            return 0.0
        
        correct_matches = 0
        total_terms = len(term_table)
        
        for term in term_table:
            source_term = term.get('source', '')
            target_term = term.get('target', '')
            confidence = term.get('confidence', 0.0)
            
            # 检查源术语是否在源文中出现
            if source_term in source:
                # 检查目标术语是否在目标文中出现
                if target_term in target:
                    correct_matches += 1
        
        return correct_matches / total_terms if total_terms > 0 else 0.0
    
    def calculate_deontic_preservation(self, source: str, target: str, src_lang: str, tgt_lang: str) -> float:
        """计算情态一致性"""
        # 提取源文情态动词
        source_modals = self._extract_modals(source, src_lang)
        target_modals = self._extract_modals(target, tgt_lang)
        
        if not source_modals:
            return 1.0  # 没有情态动词，认为一致
        
        correct_mappings = 0
        total_modals = len(source_modals)
        
        for src_modal in source_modals:
            # 查找对应的目标情态动词
            expected_tgt = self.deontic_mapping.get(src_lang, {}).get(src_modal, '')
            if expected_tgt and expected_tgt in target_modals:
                correct_mappings += 1
        
        return correct_mappings / total_modals if total_modals > 0 else 0.0
    
    def calculate_conditional_logic_preservation(self, source: str, target: str, src_lang: str, tgt_lang: str) -> float:
        """计算条件逻辑保留"""
        # 提取条件结构
        source_conditionals = self._extract_conditionals(source, src_lang)
        target_conditionals = self._extract_conditionals(target, tgt_lang)
        
        if not source_conditionals:
            return 1.0  # 没有条件结构，认为保留
        
        # 检查条件结构数量是否匹配
        if len(source_conditionals) != len(target_conditionals):
            return 0.0
        
        # 检查条件结构类型是否匹配
        correct_structures = 0
        for src_cond, tgt_cond in zip(source_conditionals, target_conditionals):
            if self._is_conditional_equivalent(src_cond, tgt_cond, src_lang, tgt_lang):
                correct_structures += 1
        
        return correct_structures / len(source_conditionals) if source_conditionals else 0.0
    
    def _extract_modals(self, text: str, lang: str) -> List[str]:
        """提取情态动词"""
        modals = []
        if lang == 'zh':
            for modal in self.deontic_mapping['zh'].keys():
                if modal in text:
                    modals.append(modal)
        elif lang == 'en':
            for modal in self.deontic_mapping['en'].keys():
                if modal in text.lower():
                    modals.append(modal)
        return modals
    
    def _extract_conditionals(self, text: str, lang: str) -> List[str]:
        """提取条件结构"""
        conditionals = []
        markers = self.conditional_markers.get(lang, [])
        
        for marker in markers:
            if marker in text:
                conditionals.append(marker)
        
        return conditionals
    
    def _is_conditional_equivalent(self, src_cond: str, tgt_cond: str, src_lang: str, tgt_lang: str) -> bool:
        """检查条件结构是否等价"""
        # 简化的等价性检查
        if src_lang == 'zh' and tgt_lang == 'en':
            return (src_cond == '如果' and tgt_cond == 'if') or \
                   (src_cond == '除非' and tgt_cond == 'unless')
        elif src_lang == 'en' and tgt_lang == 'zh':
            return (src_cond == 'if' and tgt_cond == '如果') or \
                   (src_cond == 'unless' and tgt_cond == '除非')
        
        return src_cond == tgt_cond
    
    def calculate_comet_score(self, source: str, target: str, reference: str) -> float:
        """计算COMET分数（简化版）"""
        # 这里应该调用实际的COMET模型
        # 暂时返回基于长度和词汇重叠的简化分数
        source_words = set(source.lower().split())
        target_words = set(target.lower().split())
        reference_words = set(reference.lower().split())
        
        # 计算词汇重叠度
        target_ref_overlap = len(target_words & reference_words) / len(reference_words) if reference_words else 0
        source_target_overlap = len(source_words & target_words) / len(source_words) if source_words else 0
        
        # 简化的COMET分数
        comet_score = (target_ref_overlap * 0.7 + source_target_overlap * 0.3)
        return min(comet_score, 1.0)
    
    def calculate_all_metrics(self, source: str, target: str, reference: str, 
                            src_lang: str, tgt_lang: str, term_table: List[Dict]) -> Dict[str, float]:
        """计算所有指标"""
        return {
            'termbase_accuracy': self.calculate_termbase_accuracy(source, target, term_table),
            'deontic_preservation': self.calculate_deontic_preservation(source, target, src_lang, tgt_lang),
            'conditional_logic_preservation': self.calculate_conditional_logic_preservation(source, target, src_lang, tgt_lang),
            'comet_score': self.calculate_comet_score(source, target, reference)
        }
    
    def generate_confusion_matrix(self, results: List[Dict]) -> Dict[str, Any]:
        """生成混淆矩阵"""
        # 情态动词混淆矩阵
        deontic_predictions = []
        deontic_actual = []
        
        for result in results:
            if 'deontic_actual' in result and 'deontic_predicted' in result:
                deontic_actual.append(result['deontic_actual'])
                deontic_predictions.append(result['deontic_predicted'])
        
        if deontic_actual and deontic_predictions:
            cm = confusion_matrix(deontic_actual, deontic_predictions)
            return {
                'deontic_confusion_matrix': cm.tolist(),
                'deontic_accuracy': accuracy_score(deontic_actual, deontic_predictions)
            }
        
        return {}


# 使用示例
def test_metrics():
    """测试指标计算"""
    metrics = LegalTranslationMetrics()
    
    source = "合同当事人应当按照约定履行义务"
    target = "Contracting parties shall perform their obligations in accordance with the agreement"
    reference = "Contracting parties must perform their obligations according to the agreement"
    term_table = [
        {"source": "合同", "target": "contract", "confidence": 0.95},
        {"source": "当事人", "target": "parties", "confidence": 0.92},
        {"source": "义务", "target": "obligations", "confidence": 0.88}
    ]
    
    results = metrics.calculate_all_metrics(
        source, target, reference, "zh", "en", term_table
    )
    
    print("指标结果:")
    for metric, score in results.items():
        print(f"  {metric}: {score:.3f}")


if __name__ == "__main__":
    test_metrics()
