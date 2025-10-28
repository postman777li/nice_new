# 🚀 质量评估功能 - 快速开始

## 一行命令开始使用

```bash
python run_translation.py \
  --source "你的源文本" \
  --reference "参考译文" \
  --evaluate
```

## 常用命令

### 1️⃣ 基本评估
```bash
python run_translation.py \
  --source "劳动者享有平等就业的权利。" \
  --reference "Workers shall have the right to equal employment." \
  --evaluate
```

### 2️⃣ 完整系统评估（推荐）
```bash
python run_translation.py \
  --source "劳动者享有平等就业的权利。" \
  --reference "Workers shall have the right to equal employment." \
  --hierarchical \
  --use-termbase \
  --evaluate \
  --verbose
```

### 3️⃣ 保存评估结果
```bash
python run_translation.py \
  --source "劳动者享有平等就业的权利。" \
  --reference "Workers shall have the right to equal employment." \
  --evaluate \
  --output result.json
```

## 输出说明

```
📈 评分详情
  总体评分: 85.00% ⭐⭐⭐⭐
  - 准确性:   90.00%  ← 意思是否准确
  - 流畅性:   85.00%  ← 表达是否自然
  - 术语:     80.00%  ← 术语是否规范
  - 风格:     85.00%  ← 风格是否专业

✨ 翻译优点           ← 做得好的地方
⚠️  需要改进          ← 存在的问题
💡 改进建议          ← 具体优化方案
```

## 参数组合

| 场景 | 参数组合 |
|------|----------|
| 快速评估 | `--evaluate --reference "..."` |
| 详细评估 | `--evaluate --reference "..." --verbose` |
| 评估完整系统 | `--hierarchical --use-termbase --evaluate --reference "..."` |
| 评估基线 | `--no-hierarchical --evaluate --reference "..."` |

## 更多信息

- 📖 详细文档: [QUALITY_ASSESSMENT_USAGE.md](QUALITY_ASSESSMENT_USAGE.md)
- 📝 更新日志: [CHANGELOG_QUALITY_ASSESSMENT.md](CHANGELOG_QUALITY_ASSESSMENT.md)
- 🧪 运行测试: `./test_quality_assessment.sh`
- 💡 查看示例: `./examples/quality_assessment_example.sh`
