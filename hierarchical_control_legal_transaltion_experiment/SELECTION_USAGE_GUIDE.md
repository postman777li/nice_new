# LLM候选选择器使用指南

## 概述

新增的LLM候选选择器功能允许在翻译过程中生成多个候选译文，然后使用LLM评估并选择最佳的一个。这个功能可以提高翻译质量。

## 架构说明

**独立Agent架构**：
- `LLMSelectorAgent`: 独立的选择器智能体，负责评估和选择最佳候选
- `TranslationAgent`: 负责生成候选翻译（可生成1个或多个）
- `Workflow`: 在workflow层协调翻译Agent和选择器Agent的调用

## 命令行选项

### `run_translation.py` 新增选项

```bash
# 基本语法
python run_translation.py --source "你的文本" [选项]

# LLM选择器相关选项：
--selection-layers <layers>    # 启用选择器的层级
--num-candidates <N>            # 每层生成的候选数量
```

### `--selection-layers` 选项值

| 值 | 说明 | 示例 |
|---|---|---|
| `none` | 不启用选择器（默认） | `--selection-layers none` |
| `terminology` | 只在术语层启用 | `--selection-layers terminology` |
| `syntax` | 只在句法层启用 | `--selection-layers syntax` |
| `discourse` | 只在篇章层启用 | `--selection-layers discourse` |
| `last` | 只在最后一层（篇章层）启用 | `--selection-layers last` |
| `all` | 在所有三个层级启用 | `--selection-layers all` |
| `terminology,syntax` | 在指定的多个层级启用 | `--selection-layers terminology,syntax` |

### `--num-candidates` 选项

指定每层生成的候选翻译数量（默认: 3）

- 建议范围: 3-5
- 候选越多，LLM评估成本越高
- 候选越多，选择质量可能越好

## 使用示例

### 示例1：不使用选择器（传统模式）

```bash
python run_translation.py \
    --source "本合同自双方签字之日起生效。" \
    --src-lang zh \
    --tgt-lang en
```

### 示例2：只在篇章层使用选择器（推荐）

```bash
python run_translation.py \
    --source "本合同自双方签字之日起生效。" \
    --src-lang zh \
    --tgt-lang en \
    --selection-layers last \
    --num-candidates 5
```

**说明**：在最后一层（篇章层）生成5个候选，LLM选择最佳的一个。这是推荐的配置。

### 示例3：在所有层级使用选择器

```bash
python run_translation.py \
    --source "本合同自双方签字之日起生效。" \
    --src-lang zh \
    --tgt-lang en \
    --selection-layers all \
    --num-candidates 3
```

**说明**：在术语层、句法层、篇章层都生成3个候选并选择。

### 示例4：自定义层级

```bash
python run_translation.py \
    --source "本合同自双方签字之日起生效。" \
    --src-lang zh \
    --tgt-lang en \
    --selection-layers terminology,discourse \
    --num-candidates 4
```

**说明**：只在术语层和篇章层使用选择器，句法层不使用。

### 示例5：保存结果到文件

```bash
python run_translation.py \
    --source "本合同自双方签字之日起生效。" \
    --src-lang zh \
    --tgt-lang en \
    --selection-layers last \
    --num-candidates 5 \
    --output result.json \
    --verbose
```

**说明**：启用详细输出，并将完整结果保存到JSON文件。

## 运行实验脚本

`run_experiment.py` 也支持相同的选项：

```bash
# 运行实验并启用选择器
python run_experiment.py \
    --ablation full \
    --num-samples 20 \
    --selection-layers all \
    --num-candidates 3
```

## 工作流程

### 启用选择器后的翻译流程

1. **术语层**（如果启用选择器）：
   ```
   TranslationAgent 生成3个候选译文
   ↓
   LLMSelectorAgent 评估并选择最佳
   ↓
   返回最佳译文
   ```

2. **句法层**（如果启用选择器）：
   ```
   SyntaxTranslationAgent 基于术语层结果生成3个改进候选
   ↓
   LLMSelectorAgent 评估并选择最佳
   ↓
   返回最佳译文
   ```

3. **篇章层**（如果启用选择器）：
   ```
   DiscourseTranslationAgent 基于句法层结果生成5个整合候选
   ↓
   LLMSelectorAgent 评估并选择最佳
   ↓
   返回最终译文
   ```

## 性能和成本考虑

### API调用次数

假设不使用选择器，每个样本需要 **3次** LLM调用（术语+句法+篇章）

启用选择器后：
- 每层需要额外的：
  - N次候选生成调用（N = num_candidates）
  - 1次选择器评估调用

示例：`--selection-layers all --num-candidates 3`
- 术语层: 3次生成 + 1次选择 = 4次
- 句法层: 3次生成 + 1次选择 = 4次
- 篇章层: 3次生成 + 1次选择 = 4次
- **总计**: 12次调用（vs 原来的3次）

### 推荐配置

**平衡质量和成本**：
```bash
--selection-layers last --num-candidates 5
```
- 只在最后一层使用选择器
- 可以获得显著的质量提升
- API调用增加较少（原来3次 → 现在8次）

**追求最高质量**：
```bash
--selection-layers all --num-candidates 5
```
- 在所有层级使用选择器
- 质量最佳
- API调用较多（原来3次 → 现在18次）

## 调试和日志

使用 `--verbose` 选项查看详细信息：

```bash
python run_translation.py \
    --source "你的文本" \
    --selection-layers last \
    --num-candidates 3 \
    --verbose
```

日志会显示：
- 生成的候选数量
- LLM选择器的评估理由
- 每个候选的分数
- 选中的候选编号

## 常见问题

### Q: 选择器和传统模式有什么区别？

A: 传统模式每层只生成1个翻译。启用选择器后，每层生成N个候选，然后用LLM评估选择最佳的一个。

### Q: 推荐在哪些层级启用选择器？

A: 推荐 `--selection-layers last`，即只在篇章层启用。这样可以在质量和成本之间取得平衡。

### Q: num-candidates 设置多少合适？

A: 建议3-5个。太少可能选择空间不足，太多会增加成本且收益递减。

### Q: 选择器是否支持COMET评分？

A: 当前版本只支持LLM选择器。COMET选择器已被移除，因为LLM选择器更快、更灵活。

### Q: 如何查看选择器的决策过程？

A: 使用 `--verbose` 选项，会输出详细的候选评估信息和选择理由。

## 进一步阅读

- 查看 `src/agents/selector/llm_selector_agent.py` 了解选择器实现
- 查看 `src/workflows/*.py` 了解workflow如何调用选择器
- 运行 `python test_llm_selector.py` 测试选择器功能
