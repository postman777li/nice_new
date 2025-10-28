"""
Agent工具类模块
"""
# LLM选择器
from .llm_selector import LLMSelector

# 统一的翻译控制配置（整合了门控和候选选择）
from .translation_control_config import (
    TranslationControlConfig,
    ControlConfigPresets,
    set_global_control_config,
    get_global_control_config,
    # 向后兼容的别名
    CandidateSelectionConfig,
    SelectionConfigPresets,
    set_global_config,
    get_global_config,
    set_global_gating_config,
    get_global_gating_config,
    COMETSelectionConfig,
    COMETConfigPresets
)

__all__ = [
    # LLM选择器
    'LLMSelector',
    # 统一的翻译控制配置
    'TranslationControlConfig',
    'ControlConfigPresets',
    'set_global_control_config',
    'get_global_control_config',
    # 向后兼容别名
    'CandidateSelectionConfig',
    'SelectionConfigPresets',
    'set_global_config',
    'get_global_config',
    'set_global_gating_config',
    'get_global_gating_config',
    'COMETSelectionConfig',
    'COMETConfigPresets'
]

