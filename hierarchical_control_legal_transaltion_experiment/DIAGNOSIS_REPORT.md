# 实验问题诊断报告

## 问题描述
运行实验后，三个配置（full, terminology, terminology_syntax）的翻译结果和评估得分完全相同（100%相同率）。

## 根本原因分析

### 1. **中间层翻译被门控过滤**

**现象**：
- Full配置中，r2（句法层）和r3（篇章层）都被标记为`gated: True`
- r2评估分数：0.94（高于默认阈值0.85）
- r3评估分数：0.83（高于默认阈值0.75）
- 所有样本的r1, r2, r3输出完全相同

**原因**：
虽然命令行参数`--gating-layers`默认为`'none'`（应该禁用门控），但门控逻辑仍然被触发。这导致：
- 句法层：因为评估分数0.94 > 阈值0.85，被认为"翻译已经很好"，跳过修改
- 篇章层：因为评估分数0.83 > 阈值0.75，被认为"翻译已经很好"，跳过修改

**代码位置**：
- `src/workflows/syntax.py:84-101`
- `src/workflows/discourse.py:106-135`

### 2. **Terminology和Terminology_syntax结果来自Full实验的提取**

**现象**：
- Terminology和terminology_syntax配置的结果没有trace信息
- 只有full配置有完整的trace
- 三个配置的结果100%相同

**原因**：
在`run_experiment.py:634-685`中，如果只运行full配置且启用了`--save-intermediate`，代码会从full实验的intermediate字段自动提取terminology和terminology_syntax的结果，而不是独立运行这些配置。

由于r2和r3被门控，导致r1, r2, r3的输出相同，从而提取出的所有层结果也相同。

### 3. **Discourse.py中的Bug**（已修复）

**Bug**：
在`discourse.py:119`，使用了未定义的变量`selected_references`（该变量在第155行才定义）。

**修复**：
将`selected_references`改为`top_references`（在第75行已定义）。

## 实际问题

### 配置传递链路
```
命令行参数（--gating-layers='none'）
  ↓
run_experiment.py:434 set_global_control_config()
  ↓
run_experiment.py:60 创建SimpleTranslator(config)
  ↓
run_translation.py:49 set_global_control_config()  # 覆盖全局配置
  ↓
workflows使用get_global_control_config()
```

**问题**：全局配置被多次设置，可能导致配置不一致。

### 门控触发条件错误

**期望行为**：
- `gating_layers='none'` → `is_gating_enabled()` 返回 `False` → 不应该触发门控

**实际行为**：
- 门控逻辑被触发，导致r2和r3的修改被跳过

**可能原因**：
1. 全局配置被覆盖
2. 配置解析错误
3. 默认配置有问题

## 建议修复方案

### 方案1：禁用门控功能（快速修复）
确保默认情况下门控完全禁用：

```python
# src/agents/utils/translation_control_config.py
# 确保默认 gating_enabled_layers 为空集
gating_enabled_layers: Set[str] = field(default_factory=set)
```

### 方案2：修复配置传递
在SimpleTranslator初始化时，不覆盖全局配置，而是继承：

```python
# run_translation.py
def __init__(self, config: Dict[str, Any], verbose: bool = False):
    # ... 现有代码 ...
    
    # 检查是否已有全局配置
    existing_config = get_global_control_config()
    if existing_config is None:
        # 只有在没有全局配置时才设置
        set_global_control_config(control_config)
    else:
        # 使用现有的全局配置
        self.selection_config = existing_config
```

### 方案3：使实验独立运行
修改run_experiment.py，让terminology和terminology_syntax配置独立运行，而不是从full提取：

```python
# run_experiment.py:597
# 为每个配置独立运行实验
for ablation_name in args.ablations:
    # 独立运行，不从full提取
    results = await runner.run_ablation(...)
    all_results[ablation_name] = results
```

## 验证步骤

1. 修复discourse.py的bug（✅ 已完成）
2. 运行小规模测试（1-2个样本）
3. 检查trace中的gated标记
4. 确认r1, r2, r3的输出不同
5. 运行完整实验

## 快速测试命令

```bash
# 测试1个样本，禁用门控，verbose输出
python run_experiment.py \
  --samples 1 \
  --ablations full \
  --save-intermediate \
  --gating-layers none \
  --verbose

# 检查结果
python -c "
import json
with open('outputs/experiment_results_*.json') as f:
    r = json.load(f)
    t = r['full'][0]['trace']
    print('R1:', t['r1']['output'][:50])
    print('R2:', t['r2']['output'][:50], 'gated:', t['r2'].get('gated'))
    print('R3:', t['r3']['output'][:50], 'gated:', t['r3'].get('gated'))
"
```

## 下一步行动

1. ✅ 修复discourse.py的bug
2. 🔄 验证配置传递是否正确
3. ⏳ 运行小规模测试
4. ⏳ 根据测试结果决定采用哪个修复方案

