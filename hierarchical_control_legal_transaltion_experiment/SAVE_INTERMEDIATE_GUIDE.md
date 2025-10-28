# `--save-intermediate` 功能使用指南

## 🎯 功能说明

`--save-intermediate` 参数允许你在运行 `full` 消融实验时，**自动保存中间层的翻译结果**（术语层、句法层），无需分别运行 `terminology` 和 `terminology_syntax` 实验。

### 优势

✅ **节省时间**：只需运行1次 `full` 实验，自动生成3个消融实验的结果  
✅ **节省成本**：避免重复调用LLM API  
✅ **数据一致性**：确保所有层级的翻译使用相同的输入和术语表  

---

## 🚀 使用方法

### 基础用法

```bash
# 运行full实验并保存中间结果
python run_experiment.py --ablations full --save-intermediate

# 带verbose模式查看详细过程
python run_experiment.py --ablations full --save-intermediate --verbose
```

### 完整示例

```bash
# 运行100个样本的full实验，保存中间结果
python run_experiment.py \
  --ablations full \
  --save-intermediate \
  --samples 100 \
  --verbose
```

---

## 📊 输出结果

### 文件结构

运行后会生成以下文件：

```
outputs/
├── experiment_results_1234567890.json              # 完整结果（包含full）
├── experiment_results_1234567890_terminology.json  # 术语层结果（自动提取）
└── experiment_results_1234567890_terminology_syntax.json  # 术语+句法层结果（自动提取）
```

### 主文件内容

`experiment_results_1234567890.json`:
```json
{
  "full": [
    {
      "sample_id": "sample_001",
      "source": "合同双方应当遵守本协议的所有条款。",
      "target": "...",
      "prediction": "Both parties to the contract shall comply with all provisions of this agreement.",
      "success": true,
      "intermediate": {
        "round1_terminology": {
          "prediction": "The contract parties shall comply with all provisions...",
          "terms_used": 3,
          "confidence": 0.85
        },
        "round2_syntax": {
          "prediction": "Both parties to the contract shall comply with all provisions...",
          "confidence": 0.88
        },
        "round3_discourse": {
          "prediction": "Both parties to the contract shall comply with all provisions of this agreement.",
          "tm_used": true,
          "confidence": 0.92
        }
      },
      "trace": { ... }
    }
  ],
  "terminology": [
    {
      "sample_id": "sample_001",
      "source": "合同双方应当遵守本协议的所有条款。",
      "prediction": "The contract parties shall comply with all provisions...",
      "terms_used": 3,
      "confidence": 0.85
    }
  ],
  "terminology_syntax": [
    {
      "sample_id": "sample_001",
      "source": "合同双方应当遵守本协议的所有条款。",
      "prediction": "Both parties to the contract shall comply with all provisions...",
      "confidence": 0.88
    }
  ]
}
```

---

## 🔍 调试模式

### 查看详细过程

```bash
python run_experiment.py --ablations full --save-intermediate --verbose
```

### 输出示例

```
运行消融实验: full - 完整三层翻译
============================================================
样本数: 100
层级控制: max_rounds=3
使用术语库: True
并发模式: 批量并发 (最大并发: 10)
💾 保存中间层结果: 是

[1/100] 💾 提取中间结果: trace包含 ['r1', 'r2', 'r3']
  ✓ 提取了 round1_terminology
  ✓ 提取了 round2_syntax
  ✓ 提取了 round3_discourse
  💾 中间结果包含: ['round1_terminology', 'round2_syntax', 'round3_discourse']

...

============================================================
从full实验中提取中间层结果...
============================================================
📊 包含intermediate字段的样本: 95/100
✓ 提取了 95 个术语层结果
✓ 提取了 95 个术语+句法层结果
✓ 从1次full实验自动生成了 3 个消融实验结果！

✅ 结果已保存到: outputs/experiment_results_1234567890.json
  ✅ terminology层结果已单独保存到: outputs/experiment_results_1234567890_terminology.json
  ✅ terminology_syntax层结果已单独保存到: outputs/experiment_results_1234567890_terminology_syntax.json
  💾 共保存了 2 个中间层结果文件
```

---

## ⚠️ 常见问题

### Q1: 为什么有些样本没有包含intermediate字段？

**可能原因：**
1. 样本翻译失败（`success: false`）
2. trace数据不完整（某些round缺失）

**解决方法：**
- 使用 `--verbose` 查看详细日志
- 检查是否有错误信息
- 确保full实验的max_rounds=3

### Q2: 提示"未提取到中间结果"怎么办？

**调试步骤：**
```bash
# 1. 运行带verbose的测试
python run_experiment.py --ablations full --save-intermediate --verbose --samples 5

# 2. 检查输出中的警告信息
#    - "r1存在但无output"
#    - "save_intermediate=True 但未提取到任何中间结果"

# 3. 查看trace数据结构
# 在输出的JSON文件中检查 full 结果的 trace 字段
```

### Q3: 只想保存特定层的结果可以吗？

目前 `--save-intermediate` 会保存所有可用的中间层结果。如果只需要特定层，可以：

**方法1：从JSON文件中提取**
```python
import json

with open('experiment_results_1234567890.json') as f:
    data = json.load(f)

# 只保存terminology层
terminology_only = {
    'terminology': data['terminology']
}

with open('terminology_only.json', 'w') as f:
    json.dump(terminology_only, f, ensure_ascii=False, indent=2)
```

**方法2：分别运行实验**
```bash
# 如果确实只需要terminology层，直接运行
python run_experiment.py --ablations terminology
```

### Q4: 中间结果文件和完整文件有什么区别？

**完整文件** (`experiment_results_1234567890.json`):
- 包含所有消融实验的结果
- full 条目中包含 `intermediate` 和 `trace` 字段
- 文件较大

**中间结果文件** (`experiment_results_1234567890_terminology.json`):
- 只包含特定层的结果
- 没有 `trace` 字段（更轻量）
- 便于单独分析和对比

---

## 📈 性能对比

### 不使用 --save-intermediate

```bash
# 需要运行3次实验
python run_experiment.py --ablations terminology        # 耗时: 10分钟
python run_experiment.py --ablations terminology_syntax # 耗时: 15分钟
python run_experiment.py --ablations full               # 耗时: 20分钟
# 总计: 45分钟，3次API调用
```

### 使用 --save-intermediate ⭐

```bash
# 只需运行1次实验
python run_experiment.py --ablations full --save-intermediate
# 总计: 20分钟，1次API调用
# 节省: 25分钟 (55%) + 减少2次重复API调用
```

---

## 💡 最佳实践

### 1. 开发阶段

```bash
# 小样本测试，确保功能正常
python run_experiment.py \
  --ablations full \
  --save-intermediate \
  --verbose \
  --samples 10
```

### 2. 正式实验

```bash
# 完整数据集，保存中间结果
python run_experiment.py \
  --ablations full \
  --save-intermediate \
  --samples 0  # 0表示使用全部样本
```

### 3. 结合COMET选择

```bash
# 如果已实现COMET选择功能
python run_experiment.py \
  --ablations full \
  --save-intermediate \
  --comet-layers discourse \
  --comet-candidates 5
```

### 4. 批量对比实验

```bash
# 运行多个配置，自动保存中间结果
for config in baseline full; do
  if [ "$config" = "full" ]; then
    python run_experiment.py --ablations $config --save-intermediate
  else
    python run_experiment.py --ablations $config
  fi
done
```

---

## 🔧 技术实现

### 数据流

```
输入样本
    ↓
[Round 1: 术语层]
    ├─ output → round1_terminology.prediction
    └─ termTable → round1_terminology.terms_used
    ↓
[Round 2: 句法层]
    ├─ output → round2_syntax.prediction
    └─ confidence → round2_syntax.confidence
    ↓
[Round 3: 篇章层]
    ├─ output → round3_discourse.prediction
    └─ tm_used → round3_discourse.tm_used
    ↓
保存到 result['intermediate']
    ↓
从 intermediate 提取 terminology 和 terminology_syntax
    ↓
保存到独立文件
```

### 关键代码

```python
# 提取中间结果（run_experiment.py 第67-136行）
if save_intermediate and result['success']:
    trace = result.get('trace', {})
    
    # Round 1: 术语层
    if 'r1' in trace and trace['r1'].get('output'):
        intermediate_results['round1_terminology'] = {
            'prediction': trace['r1']['output'],
            'terms_used': len(trace['r1'].get('termTable', [])),
            'confidence': trace['r1'].get('confidence', 0.0)
        }
    
    # ... 其他层类似

# 从full中提取并保存（run_experiment.py 第436-501行）
if save_intermediate and ablation_name == 'full':
    for sample in results:
        if 'intermediate' in sample and 'round1_terminology' in sample['intermediate']:
            terminology_results.append({
                'prediction': sample['intermediate']['round1_terminology']['prediction'],
                # ...
            })
```

---

## 📚 相关文档

- **实验脚本**: [run_experiment.py](./run_experiment.py)
- **实验指南**: [README.md](./README.md)
- **消融实验文档**: [ABLATION_EXPERIMENTS.md](./ABLATION_EXPERIMENTS.md)

---

## 📞 故障排查

如果遇到问题：

1. **检查日志输出**：使用 `--verbose` 查看详细信息
2. **检查trace结构**：确保full实验的trace包含r1, r2, r3
3. **确认成功率**：查看有多少样本成功翻译
4. **查看JSON文件**：手动检查输出文件的数据结构

如果问题仍然存在，请提供：
- 完整的运行命令
- 错误信息或日志输出
- 输出JSON文件的样例（1-2个样本即可）

---

**更新日期**: 2024-10-12  
**功能版本**: v2.0

