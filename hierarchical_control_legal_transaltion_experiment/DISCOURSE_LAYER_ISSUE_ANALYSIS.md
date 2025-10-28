# 篇章层BLEU分数下降问题分析

## 问题概述

实验数据（20个样本）显示：
- **术语层**: BLEU = 42.08
- **术语+句法层**: BLEU = 41.85
- **术语+句法+篇章层**: BLEU = **35.85** ❌ （下降6.2分）

## 根本原因

### 1. TM检索策略问题

篇章层在所有20个样本中都使用了TM（翻译记忆），检索到的参考译文可能来自：
- **不同的翻译版本**（如官方译本 vs 学术译本）
- **不同的翻译时期**（术语标准可能变化）
- **不同的翻译风格**（直译 vs 意译）

### 2. 过度改写

为了与TM参考保持风格一致，篇章层大量改写了用词和句式：

#### 样本1: 用词替换
```
句法层: "fully consider"
篇章层: "give full consideration to"  ← TM风格
参考:   "fully consider"  ✓ 句法层正确！
```

#### 样本2: 句式重构
```
句法层: "When several People's Courts..."
篇章层: "Cases over which several People's Courts..."  ← TM风格
参考:   "Where two or more people's courts..."  ✓ 句法层更接近
```

#### 样本3: 术语替换
```
句法层: "emergency danger aversion"
篇章层: "seeking to avoid a peril in response to an emergency"  ← TM参考
参考:   "conduct of necessity"  ✗ 两者都不对，但句法层更简洁
```

### 3. 为什么其他指标下降较少？

| 指标 | 下降幅度 | 原因 |
|------|---------|------|
| BLEU | -6.2 | 基于精确n-gram匹配，用词变化影响大 |
| chrF | -3.8 | 字符级匹配，也受用词影响 |
| BERTScore | -0.006 | 基于语义，改写后语义基本保持 |
| COMET | -0.012 | 神经网络评估质量，语义正确影响小 |

**结论**: 篇章层的改写在语义上是正确的，但与参考译文的用词不一致。

## 解决方案

### 方案1: 保守修改策略（推荐）

修改 `src/agents/discourse/discourse_translation.py`，让篇章层只在以下情况修改：

```python
# 在篇章翻译prompt中强调
if tm_matches_low_confidence or has_obvious_errors:
    # 允许修改
else:
    # 保持原翻译，只调整用词一致性
```

### 方案2: 改进TM检索

修改 `src/agents/discourse/discourse_query.py`：

1. **提高相似度阈值**：只使用高度相似的TM
2. **限制TM数量**：只使用top-1或top-2，避免混合风格
3. **同源过滤**：优先检索同一数据源的TM

### 方案3: 修改Prompt策略

在篇章层提示中强调：

```
You should primarily focus on:
1. Maintaining terminology consistency
2. Improving discourse coherence
3. **Minimizing unnecessary paraphrasing**
4. **Preserving working translations unless there are clear issues**

Reference translations are provided for style guidance only.
Do NOT force-fit your translation to match their wording.
```

### 方案4: 条件性启用篇章层

```python
# 只在以下情况启用篇章层改写
if (
    terminology_confidence < 0.8 or
    syntax_confidence < 0.8 or
    has_inconsistencies
):
    apply_discourse_refinement()
else:
    return_previous_layer_output()
```

## 实验验证

### 测试方案A: 禁用TM
```bash
# 修改discourse workflow，临时禁用TM检索
python run_experiment.py \
  --test-set dataset/processed/test_set_zh_en_sample20.json \
  --ablations full \
  --save-intermediate
```

### 测试方案B: 更保守的Prompt
修改prompt后重新测试，看BLEU是否恢复。

### 测试方案C: 提高TM阈值
```python
# 在discourse_query.py中
similarity_threshold = 0.9  # 从0.7提高到0.9
```

## 建议优先级

1. **短期**（立即可做）：
   - 修改Prompt，强调保守修改
   - 提高TM相似度阈值

2. **中期**（需要测试）：
   - 实现条件性篇章改写
   - 优化TM检索策略

3. **长期**（架构优化）：
   - 引入COMET-Kiwi做质量评估，决定是否应用篇章修改
   - 建立同源TM数据库

## 关键洞察

**篇章层的目标应该是**：
- ✅ 保持术语一致性
- ✅ 改善篇章连贯性
- ✅ 修正明显错误
- ❌ **不应该为了匹配TM风格而过度改写**

**当前问题**：
- 篇章层过于"积极"，把正确的翻译改"错"了（从BLEU角度）
- TM的作用应该是**指导和验证**，而非**强制改写**

## 下一步

1. 先用测试方案A验证禁用TM后的效果
2. 如果BLEU恢复，说明TM策略有问题
3. 然后逐步引入更保守的TM使用策略

