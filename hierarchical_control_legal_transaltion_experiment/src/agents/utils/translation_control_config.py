"""
统一的翻译控制配置 - 整合门控和候选选择

一个配置类管理两种控制机制：
1. 门控（Gating）：输入级别过滤，决定是否应用检索到的建议
2. 候选选择（Selection）：输出级别筛选，从多个翻译候选中选择最佳
"""
from typing import Set, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class TranslationControlConfig:
    """统一的翻译控制配置"""
    
    # ==================== 候选选择配置 ====================
    # 启用候选选择的层级（'terminology', 'syntax', 'discourse'）
    selection_enabled_layers: Set[str] = field(default_factory=set)
    
    # 每层生成的候选数量（可以为每层单独配置）
    num_candidates: int = 3
    
    # ==================== 门控配置 ====================
    # 启用门控的层级（'terminology', 'syntax', 'discourse'）
    gating_enabled_layers: Set[str] = field(default_factory=set)
    
    # 各层门控阈值
    terminology_threshold: float = 0.6    # 术语置信度阈值（低于此值的术语被过滤）
    syntax_threshold: float = 0.85        # 句法评估分数阈值（高于此值不修改）
    discourse_threshold: float = 0.75     # 篇章评估分数阈值（高于此值不修改）
    tm_similarity_threshold: float = 0.7  # TM相似度阈值（低于此值的TM被过滤）
    
    # ==================== 候选选择方法 ====================
    def is_selection_enabled(self, layer_name: str) -> bool:
        """检查指定层级是否启用候选选择"""
        return layer_name in self.selection_enabled_layers
    
    def get_num_candidates(self, layer_name: str) -> int:
        """获取指定层级的候选数量"""
        return self.num_candidates
    
    # ==================== 门控方法 ====================
    def is_gating_enabled(self, layer_name: str) -> bool:
        """检查指定层级是否启用门控"""
        return layer_name in self.gating_enabled_layers
    
    def get_gating_threshold(self, layer_name: str) -> float:
        """获取指定层级的门控阈值"""
        if layer_name == 'terminology':
            return self.terminology_threshold
        elif layer_name == 'syntax':
            return self.syntax_threshold
        elif layer_name == 'discourse':
            return self.discourse_threshold
        else:
            logger.warning(f"Unknown layer: {layer_name}, returning default 0.5")
            return 0.5
    
    def should_apply_terminology(self, confidence: float) -> bool:
        """判断术语是否应该被应用（基于置信度）"""
        if not self.is_gating_enabled('terminology'):
            return True  # 门控未启用，应用所有术语
        return confidence >= self.terminology_threshold
    
    def should_apply_syntax_modification(self, evaluation_score: float) -> bool:
        """判断是否应该应用句法修改（基于评估分数）"""
        if not self.is_gating_enabled('syntax'):
            return True  # 门控未启用，总是修改
        # 分数高于阈值，说明翻译已经很好，不需要修改
        return evaluation_score < self.syntax_threshold
    
    def should_apply_discourse_modification(self, evaluation_score: Optional[float] = None) -> bool:
        """判断是否应该应用篇章修改（基于评估分数）"""
        if not self.is_gating_enabled('discourse'):
            return True  # 门控未启用，总是修改
        if evaluation_score is None:
            return True  # 无评估分数，允许修改
        # 分数高于阈值，说明翻译已经很好，不需要修改
        return evaluation_score < self.discourse_threshold
    
    def should_use_tm_reference(self, similarity: float) -> bool:
        """判断TM参考是否应该被使用（基于相似度）"""
        if not self.is_gating_enabled('discourse'):
            return True  # 门控未启用，使用所有TM
        return similarity >= self.tm_similarity_threshold
    
    @classmethod
    def from_args(
        cls,
        selection_layers: str = 'none',
        num_candidates: int = 3,
        gating_layers: str = 'none',
        term_threshold: float = 0.6,
        syntax_threshold: float = 0.85,
        discourse_threshold: float = 0.75,
        tm_threshold: float = 0.7
    ) -> 'TranslationControlConfig':
        """从命令行参数创建配置
        
        Args:
            selection_layers: 启用候选选择的层级 (none/all/last/terminology,syntax,discourse)
            num_candidates: 生成的候选数量
            gating_layers: 启用门控的层级 (none/all/terminology,syntax,discourse)
            term_threshold: 术语置信度门控阈值
            syntax_threshold: 句法评估分数门控阈值
            discourse_threshold: 篇章评估分数门控阈值
            tm_threshold: TM相似度门控阈值
        
        Returns:
            TranslationControlConfig实例
        """
        # 解析候选选择层级
        selection_enabled = cls._parse_layers(selection_layers, allow_special=['last'])
        
        # 解析门控层级
        gating_enabled = cls._parse_layers(gating_layers, allow_special=[])
        
        return cls(
            selection_enabled_layers=selection_enabled,
            num_candidates=num_candidates,
            gating_enabled_layers=gating_enabled,
            terminology_threshold=term_threshold,
            syntax_threshold=syntax_threshold,
            discourse_threshold=discourse_threshold,
            tm_similarity_threshold=tm_threshold
        )
    
    @staticmethod
    def _parse_layers(layers_str: str, allow_special: list = None) -> Set[str]:
        """解析层级字符串"""
        if not layers_str or layers_str.strip() in ['', 'none']:
            return set()
        
        layers_str = layers_str.strip().lower()
        
        # 处理特殊值
        if layers_str == 'all':
            return {'terminology', 'syntax', 'discourse'}
        
        if allow_special and layers_str == 'last':
            return {'discourse'}
        
        # 解析逗号分隔的层级
        enabled = set(layer.strip() for layer in layers_str.split(',') if layer.strip())
        
        # 验证层级名称
        valid_layers = {'terminology', 'syntax', 'discourse'}
        invalid = enabled - valid_layers
        if invalid:
            logger.warning(f"Invalid layers: {invalid}, ignoring them")
            enabled = enabled & valid_layers
        
        return enabled
    
    def __repr__(self) -> str:
        parts = []
        
        # 候选选择部分
        if self.selection_enabled_layers:
            sel_layers = ','.join(sorted(self.selection_enabled_layers))
            parts.append(f"Selection({sel_layers}, n={self.num_candidates})")
        else:
            parts.append("Selection(disabled)")
        
        # 门控部分
        if self.gating_enabled_layers:
            gate_layers = ','.join(sorted(self.gating_enabled_layers))
            thresholds = []
            if 'terminology' in self.gating_enabled_layers:
                thresholds.append(f"term≥{self.terminology_threshold}")
            if 'syntax' in self.gating_enabled_layers:
                thresholds.append(f"syntax<{self.syntax_threshold}")
            if 'discourse' in self.gating_enabled_layers:
                thresholds.append(f"discourse<{self.discourse_threshold}")
            parts.append(f"Gating({gate_layers}: {', '.join(thresholds)})")
        else:
            parts.append("Gating(disabled)")
        
        return f"TranslationControlConfig({' | '.join(parts)})"


# 全局配置实例
_global_control_config: Optional[TranslationControlConfig] = None


def set_global_control_config(config: TranslationControlConfig) -> None:
    """设置全局翻译控制配置"""
    global _global_control_config
    _global_control_config = config
    logger.info(f"✓ 全局翻译控制配置已设置: {config}")


def get_global_control_config() -> Optional[TranslationControlConfig]:
    """获取全局翻译控制配置"""
    return _global_control_config


# 预设配置
class ControlConfigPresets:
    """翻译控制配置预设"""
    
    @staticmethod
    def disabled() -> TranslationControlConfig:
        """完全禁用"""
        return TranslationControlConfig(
            selection_enabled_layers=set(),
            gating_enabled_layers=set()
        )
    
    @staticmethod
    def selection_only(layers: str = 'all', num_candidates: int = 3) -> TranslationControlConfig:
        """仅启用候选选择"""
        return TranslationControlConfig.from_args(
            selection_layers=layers,
            num_candidates=num_candidates,
            gating_layers='none'
        )
    
    @staticmethod
    def gating_only(layers: str = 'all', mode: str = 'balanced') -> TranslationControlConfig:
        """仅启用门控"""
        if mode == 'conservative':
            return TranslationControlConfig.from_args(
                selection_layers='none',
                gating_layers=layers,
                term_threshold=0.7,
                syntax_threshold=0.90,
                discourse_threshold=0.85,
                tm_threshold=0.8
            )
        elif mode == 'aggressive':
            return TranslationControlConfig.from_args(
                selection_layers='none',
                gating_layers=layers,
                term_threshold=0.5,
                syntax_threshold=0.95,
                discourse_threshold=0.90,
                tm_threshold=0.6
            )
        else:  # balanced
            return TranslationControlConfig.from_args(
                selection_layers='none',
                gating_layers=layers,
                term_threshold=0.6,
                syntax_threshold=0.85,
                discourse_threshold=0.75,
                tm_threshold=0.7
            )
    
    @staticmethod
    def full_control(
        selection_layers: str = 'syntax,discourse',
        gating_layers: str = 'all',
        mode: str = 'balanced'
    ) -> TranslationControlConfig:
        """启用门控 + 候选选择"""
        config = ControlConfigPresets.gating_only(gating_layers, mode)
        config.selection_enabled_layers = TranslationControlConfig._parse_layers(selection_layers)
        config.num_candidates = 3
        return config


# ==================== 向后兼容别名 ====================
# 为了向后兼容，保留旧的类名作为别名
CandidateSelectionConfig = TranslationControlConfig
SelectionConfigPresets = ControlConfigPresets
COMETSelectionConfig = TranslationControlConfig  # 历史遗留
COMETConfigPresets = ControlConfigPresets  # 历史遗留

# 全局配置的别名
set_global_config = set_global_control_config
get_global_config = get_global_control_config
set_global_gating_config = set_global_control_config
get_global_gating_config = get_global_control_config

