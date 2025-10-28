# 消融实验设计 / Ablation Experiment Design

## 四种渐进式实验配置

本系统设计了四种消融实验，用于验证层次化控制策略的有效性：

### 1. **baseline** - 基线（纯LLM）

直接使用大语言模型翻译，不使用任何控制策略。

**配置**:
```python
{
    'name': '基线（纯LLM）',
    'hierarchical': False,
    'useTermBase': False,
    'max_rounds': 1
}
```

**流程**:
```
源文本 → LLM翻译 → 译文
```

**用途**: 作为基线对照，评估原始LLM的翻译能力。

---

### 2. **terminology** - 术语控制

使用术语层控制，确保法律术语的准确性。

**配置**:
```python
{
    'name': '术语控制',
    'hierarchical': True,
    'useTermBase': True,
    'max_rounds': 1
}
```

**流程**:
```
源文本 
  ↓
术语层：提取术语 → 检索术语库 → 评估 → 带术语约束的翻译
  ↓
译文
```

**改进重点**:
- ✅ 法律术语的准确翻译
- ✅ 专业词汇的一致性
- ✅ 领域知识的应用

---

### 3. **terminology_syntax** - 术语+句法控制

在术语控制基础上，增加句法层控制。

**配置**:
```python
{
    'name': '术语+句法控制',
    'hierarchical': True,
    'useTermBase': True,
    'max_rounds': 2
}
```

**流程**:
```
源文本 
  ↓
术语层：术语控制翻译
  ↓
句法层：提取句法模式 → 评估保真度 → 句法优化翻译
  ↓
译文
```

**改进重点**:
- ✅ 术语层的所有优势
- ✅ 情态动词的准确翻译（shall/must/may/should）
- ✅ 连接词的精确对应
- ✅ 条件逻辑的完整性
- ✅ 法律句式的规范性

---

### 4. **full** - 完整系统（术语+句法+篇章）

完整的三层层次化控制系统。

**配置**:
```python
{
    'name': '完整系统（术语+句法+篇章）',
    'hierarchical': True,
    'useTermBase': True,
    'max_rounds': 3
}
```

**流程**:
```
源文本 
  ↓
术语层：术语控制翻译
  ↓
句法层：句法优化翻译
  ↓
篇章层：检索翻译记忆 → 分析差异 → 风格一致化
  ↓
最终译文
```

**改进重点**:
- ✅ 术语层和句法层的所有优势
- ✅ 翻译风格的一致性
- ✅ 用词选择与历史翻译对齐
- ✅ 句法模式与参考保持一致
- ✅ 整体译文质量的提升

---

## 运行实验

### 运行单个实验

```bash
# 1. 只运行基线实验
python run_experiment.py --samples 10 --ablations baseline

# 2. 只运行术语控制
python run_experiment.py --samples 10 --ablations terminology

# 3. 只运行术语+句法
python run_experiment.py --samples 10 --ablations terminology_syntax

# 4. 只运行完整系统
python run_experiment.py --samples 10 --ablations full
```

### 运行全部四种实验（推荐）

```bash
# 运行所有消融实验
python run_experiment.py \
  --samples 50 \
  --ablations baseline terminology terminology_syntax full \
  --output-dir experiments/ablation_study \
  --verbose
```

### 快速测试

```bash
# 使用3个样本快速测试所有配置
python run_experiment.py \
  --samples 3 \
  --ablations baseline terminology terminology_syntax full
```

---

## 评估指标

每个实验会计算以下指标：

### 1. **术语准确性** (termbase_accuracy)
- 衡量术语翻译的准确率
- 基线实验中该指标较低
- 术语控制后显著提升

### 2. **情态动词保真度** (deontic_preservation)
- 衡量法律情态动词的翻译准确性
- shall/must/may/should 的正确使用
- 句法层介入后提升

### 3. **条件逻辑维护** (conditional_logic_preservation)
- 衡量条件句和逻辑关系的保持
- 句法层的重要改进点

### 4. **COMET分数** (comet_score)
- 基于语义相似度的整体质量评分
- 篇章层介入后达到最优

---

## 预期结果

根据层次化控制理论，预期结果趋势：

```
指标提升趋势：
baseline < terminology < terminology_syntax < full

术语准确性：
baseline: ~0.60  →  terminology: ~0.85  →  terminology_syntax: ~0.87  →  full: ~0.90

情态动词保真度：
baseline: ~0.55  →  terminology: ~0.60  →  terminology_syntax: ~0.85  →  full: ~0.88

条件逻辑维护：
baseline: ~0.50  →  terminology: ~0.55  →  terminology_syntax: ~0.80  →  full: ~0.83

整体质量(COMET)：
baseline: ~0.65  →  terminology: ~0.75  →  terminology_syntax: ~0.82  →  full: ~0.88
```

---

## 结果分析

### 查看实验结果

```bash
# 查看输出目录
ls -la experiments/ablation_study/

# 使用jq分析JSON结果
cat experiments/ablation_study/experiment_results_*.json | jq '
{
  baseline: .baseline[0].metrics,
  terminology: .terminology[0].metrics,
  terminology_syntax: .terminology_syntax[0].metrics,
  full: .full[0].metrics
}'
```

### 生成对比报告

创建 `analyze_results.py`:

```python
import json
import sys

with open(sys.argv[1], 'r') as f:
    data = json.load(f)

for exp_name in ['baseline', 'terminology', 'terminology_syntax', 'full']:
    results = data.get(exp_name, [])
    if results:
        valid = [r for r in results if r['success'] and r['metrics']]
        if valid:
            print(f"\n{exp_name}:")
            metrics = valid[0]['metrics']
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.3f}")
```

运行分析：
```bash
python analyze_results.py experiments/ablation_study/experiment_results_*.json
```

---

## 研究问题

这些消融实验可以回答以下研究问题：

### RQ1: 术语控制的效果
**对比**: baseline vs terminology

**评估**: 术语准确性提升幅度

### RQ2: 句法控制的增量贡献
**对比**: terminology vs terminology_syntax

**评估**: 情态动词和条件逻辑指标的提升

### RQ3: 篇章层的作用
**对比**: terminology_syntax vs full

**评估**: 整体质量和风格一致性的提升

### RQ4: 层次化策略的整体效果
**对比**: baseline vs full

**评估**: 所有指标的综合提升幅度

---

## 统计显著性检验

建议使用配对 t 检验验证改进的显著性：

```python
from scipy import stats

baseline_scores = [r['metrics']['comet_score'] for r in baseline_results]
full_scores = [r['metrics']['comet_score'] for r in full_results]

t_stat, p_value = stats.ttest_rel(baseline_scores, full_scores)
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")
```

---

## 可视化建议

### 1. 雷达图
展示四种配置在各指标上的表现

### 2. 柱状图
对比每个指标在四种配置下的得分

### 3. 改进百分比图
显示相对于baseline的改进幅度

### 4. 箱型图
展示每种配置的分数分布和稳定性

---

## 注意事项

1. **样本数量**: 建议至少50个样本以确保统计可靠性
2. **随机种子**: 使用固定随机种子确保可重复性
3. **多次运行**: 建议运行3-5次取平均值
4. **错误分析**: 保存失败样本进行人工分析
5. **公平对比**: 确保所有实验使用相同的LLM和参数

---

## 扩展实验

### 其他可能的消融配置

```python
# 只使用句法（不使用术语库）
'syntax_only': {
    'hierarchical': True,
    'useTermBase': False,
    'max_rounds': 2
}

# 只使用篇章（不使用术语库）
'discourse_only': {
    'hierarchical': True,
    'useTermBase': False,
    'max_rounds': 3
}

# 启用翻译记忆
'full_with_tm': {
    'hierarchical': True,
    'useTermBase': True,
    'useTM': True,
    'max_rounds': 3
}
```

添加这些配置可以进一步分析各层的独立贡献。

---

**开始实验** 🚀

```bash
python run_experiment.py \
  --samples 50 \
  --ablations baseline terminology terminology_syntax full \
  --output-dir experiments/main_ablation \
  --verbose
```

