# 质量评估功能使用指南

## 功能简介

`run_translation.py` 新增了翻译质量评估功能，可以对比翻译结果和参考译文，给出详细的质量评分和改进建议。

## 使用方法

### 基本用法

```bash
python run_translation.py \
  --source "劳动者享有平等就业的权利。" \
  --reference "Workers shall have the right to equal employment." \
  --evaluate
```

### 完整示例

```bash
python run_translation.py \
  --source "用人单位招用劳动者，不得扣押劳动者的居民身份证和其他证件。" \
  --reference "When recruiting workers, an employing unit must not seize the workers' resident identity cards or other documents." \
  --evaluate \
  --verbose
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--reference` | 参考译文（标准答案） | 无（必需） |
| `--evaluate` | 启用质量评估 | 关闭 |
| `--verbose` | 显示详细评估报告 | 关闭 |

## 输出示例

### 基本输出

```
============================================================
📊 质量评估（对比参考译文）
============================================================
参考译文: Workers shall have the right to equal employment.

正在评估...

✅ 评估完成！

────────────────────────────────────────────────────────────
📈 评分详情
────────────────────────────────────────────────────────────
  总体评分: 85.00% ⭐⭐⭐⭐
  - 准确性:   90.00%
  - 流畅性:   85.00%
  - 术语:     80.00%
  - 风格:     85.00%

────────────────────────────────────────────────────────────
✨ 翻译优点
────────────────────────────────────────────────────────────
  1. 法律术语"劳动者"翻译为"workers"准确恰当
  2. 句子结构清晰，符合英文法律文本习惯
  3. 关键词"right"和"equal employment"翻译准确

────────────────────────────────────────────────────────────
⚠️  需要改进
────────────────────────────────────────────────────────────
  1. 情态动词使用不当："have"应改为"shall have"以体现法律强制性
  2. 缺少法律文本常用的强调性表达

────────────────────────────────────────────────────────────
💡 改进建议
────────────────────────────────────────────────────────────
  1. 将"have the right"改为"shall have the right"以增强法律效力
  2. 保持与参考译文的术语一致性
  3. 考虑使用更正式的法律表达方式

============================================================
```

### 详细输出（--verbose）

添加 `--verbose` 参数后，还会显示：

```
────────────────────────────────────────────────────────────
📝 详细对比分析
────────────────────────────────────────────────────────────
  翻译结果整体质量良好，准确传达了源文本的法律含义。主要优点是
  术语选择准确，句子结构清晰。需要改进的地方是：1) 情态动词的
  使用需要更符合法律文本的规范性要求；2) 部分法律惯用表达可以
  更加完善。参考译文使用了"shall"这一法律专用情态动词，更能体
  现法律条文的强制性和规范性。
```

## 评分标准

### 评分维度

1. **准确性 (Accuracy)**
   - 是否准确传达源文本意思
   - 是否有误译、漏译、增译

2. **流畅性 (Fluency)**
   - 是否符合目标语言表达习惯
   - 是否自然流畅

3. **术语一致性 (Terminology)**
   - 法律术语是否准确、规范
   - 术语使用是否一致

4. **风格适配 (Style)**
   - 是否符合法律文本正式性要求
   - 是否保持专业性

### 评分等级

| 分数 | 等级 | 说明 |
|------|------|------|
| 90-100% | 优秀 ⭐⭐⭐⭐⭐ | 与参考译文质量相当或更好 |
| 80-89% | 良好 ⭐⭐⭐⭐ | 有小的改进空间 |
| 70-79% | 合格 ⭐⭐⭐ | 存在一些明显问题 |
| 60-69% | 需改进 ⭐⭐ | 有较多问题 |
| <60% | 不合格 ⭐ | 存在严重问题 |

## 与其他功能配合

### 与层级控制配合

```bash
# 评估完整系统的翻译质量
python run_translation.py \
  --source "劳动者享有平等就业的权利。" \
  --reference "Workers shall have the right to equal employment." \
  --hierarchical \
  --use-termbase \
  --evaluate \
  --verbose
```

### 与候选选择配合

```bash
# 评估使用候选选择后的翻译质量
python run_translation.py \
  --source "劳动者享有平等就业的权利。" \
  --reference "Workers shall have the right to equal employment." \
  --selection-layers all \
  --num-candidates 5 \
  --evaluate
```

### 输出到JSON文件

```bash
python run_translation.py \
  --source "劳动者享有平等就业的权利。" \
  --reference "Workers shall have the right to equal employment." \
  --evaluate \
  --output result.json
```

输出的JSON文件会包含 `quality_assessment` 字段：

```json
{
  "source": "劳动者享有平等就业的权利。",
  "final": "Workers have the right to equal employment.",
  "quality_assessment": {
    "overall_score": 0.85,
    "accuracy_score": 0.90,
    "fluency_score": 0.85,
    "terminology_score": 0.80,
    "style_score": 0.85,
    "strengths": [...],
    "weaknesses": [...],
    "suggestions": [...]
  }
}
```

## 应用场景

1. **翻译质量检查**：对比翻译结果和人工译文，找出差距
2. **系统评估**：评估不同配置下的翻译质量
3. **模型改进**：根据评估建议优化翻译策略
4. **翻译学习**：了解专业翻译的标准和要求

## 注意事项

1. **必须提供参考译文**：使用 `--evaluate` 时必须同时提供 `--reference`
2. **API调用**：质量评估会额外调用一次LLM API
3. **准确性**：评估结果依赖LLM的判断能力，仅供参考
4. **对比基准**：评估是相对于参考译文的，不同参考译文会有不同结果

## 示例脚本

```bash
#!/bin/bash
# 批量评估示例

# 测试用例
SOURCE="劳动者享有平等就业的权利。"
REFERENCE="Workers shall have the right to equal employment."

# 评估基线翻译
echo "=== 评估基线翻译 ==="
python run_translation.py \
  --source "$SOURCE" \
  --reference "$REFERENCE" \
  --no-hierarchical \
  --evaluate

# 评估完整系统
echo -e "\n=== 评估完整系统 ==="
python run_translation.py \
  --source "$SOURCE" \
  --reference "$REFERENCE" \
  --hierarchical \
  --use-termbase \
  --use-tm \
  --evaluate
```

