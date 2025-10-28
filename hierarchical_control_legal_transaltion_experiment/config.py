"""
统一的实验配置管理
集中管理所有默认参数，避免硬编码
"""

# ==================== 文件路径配置 ====================
DEFAULT_TERM_DB = 'terms.db'
DEFAULT_TEST_SET = 'dataset/processed/test_set_zh_en.json'
DEFAULT_OUTPUT_DIR = 'outputs'

# ==================== 并发控制 ====================
DEFAULT_MAX_CONCURRENT = 10

# ==================== 控制机制说明 ====================
# LLM候选选择和门控阈值配置已移除，统一通过命令行参数设置
# 参见 run_experiment.py 中的 --selection-layers, --gating-layers 等参数

# ==================== 消融实验配置 ====================
ABLATION_CONFIGS = {
    'baseline': {
        'name': '基线（纯LLM）',
        'hierarchical': False,
        'useTermBase': False,
        'useTM': False,
        'max_rounds': 1
    },
    'terminology': {
        'name': '术语控制',
        'hierarchical': True,
        'useTermBase': True,
        'useTM': False,
        'max_rounds': 1  # 只运行术语层
    },
    'terminology_syntax': {
        'name': '术语+句法控制',
        'hierarchical': True,
        'useTermBase': True,
        'useTM': False,
        'max_rounds': 2  # 运行术语层和句法层
    },
    'full': {
        'name': '完整系统（术语+句法+篇章）',
        'hierarchical': True,
        'useTermBase': True,
        'useTM': True,
        'max_rounds': 3  # 运行所有三层
    }
}

# ==================== 评估指标名称 ====================
METRIC_NAMES = [
    'termbase_accuracy',
    'deontic_preservation',
    'conditional_logic_preservation',
    'comet_score'
]

# ==================== 帮助信息 ====================
# 控制机制的帮助信息已移至命令行参数的help文本中

