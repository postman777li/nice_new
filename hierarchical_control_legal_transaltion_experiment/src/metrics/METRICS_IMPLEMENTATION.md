# 现代机器翻译评估指标实现总结

## 📋 已实现的指标

基于最新的机器翻译评估研究（2023-2024），我们实现了以下5类评估指标：

### 1. BLEU (Bilingual Evaluation Understudy)
- **文件**: `src/metrics/bleu.py`
- **依赖**: `sacrebleu>=2.3.1`
- **特点**: 传统n-gram精确匹配，快速计算
- **用途**: 基线对比、快速评估

### 2. chrF++ (Character n-gram F-score)
- **文件**: `src/metrics/chrf.py`
- **依赖**: `sacrebleu>=2.3.1`
- **特点**: 字符级评估，特别适合中文
- **用途**: 多语言评估、不需要分词

### 3. BERTScore
- **文件**: `src/metrics/bertscore.py`
- **依赖**: `bert-score>=0.3.13`
- **特点**: 基于BERT的语义相似度
- **用途**: 语义质量评估

### 4. COMET (Crosslingual Optimized Metric for Evaluation of Translation)
- **文件**: `src/metrics/comet.py`
- **依赖**: `unbabel-comet>=2.2.0`
- **特点**: WMT2022最佳指标，高度相关人类判断
- **用途**: 高质量评估

### 5. GEMBA (GPT Estimation Metric Based Assessment)
- **文件**: `src/metrics/gemba_mqm.py`
- **依赖**: OpenAI API (GPT-4)
- **官方实现**: https://github.com/MicrosoftTranslator/GEMBA
- **两种方法**:
  - **GEMBA-MQM**: 详细错误检测和MQM评分
  - **GEMBA-DA**: 直接质量评估（推荐）
- **特点**: WMT2023最佳指标，最接近人工评估
- **用途**: 最终评估、错误诊断

## 🏗️ 项目结构

```
src/metrics/
├── __init__.py           # 模块导出
├── bleu.py              # BLEU指标
├── chrf.py              # chrF++指标
├── bertscore.py         # BERTScore指标
├── comet.py             # COMET指标
├── gemba_mqm.py         # GEMBA-MQM和GEMBA-DA指标
├── metric_suite.py      # 统一的指标套件
└── README.md            # 详细文档

test_metrics.py          # 测试脚本
requirements.txt         # 依赖声明（已更新）
```

## 🚀 快速使用

### 基础使用

```python
from src.metrics import MetricSuite

# 创建指标套件（快速指标）
suite = MetricSuite(metrics=['bleu', 'chrf'])

# 计算分数
scores = suite.compute(
    source="合同双方应当遵守本协议的所有条款。",
    prediction="The parties shall comply with all terms of this agreement.",
    reference="Contracting parties must comply with all provisions of this agreement."
)

print(scores)
# {'bleu': 45.23, 'chrf': 67.89}
```

### GEMBA使用（推荐）

```python
from src.metrics import GEMBAMetric

# GEMBA-DA（更快，推荐用于批量评估）
gemba_da = GEMBAMetric(method="GEMBA-DA", model="gpt-4")
score = gemba_da.sentence_score(
    source="合同双方应当遵守本协议的所有条款。",
    prediction="The parties shall comply with all terms.",
    source_lang="Chinese",
    target_lang="English"
)
print(f"GEMBA-DA: {score:.2f}/100")

# GEMBA-MQM（详细错误分析）
gemba_mqm = GEMBAMetric(method="GEMBA-MQM", model="gpt-4")
result = gemba_mqm.compute(
    sources=["合同双方应当遵守本协议的所有条款。"],
    predictions=["The parties shall comply with all terms."],
    source_lang="Chinese",
    target_lang="English"
)
print(f"分数: {result['mean']:.2f}")
print(f"错误: {result['results'][0]['errors']}")
```

## 📊 指标对比

| 指标 | 速度 | 资源 | 人类相关性 | WMT排名 | 推荐场景 |
|------|------|------|-----------|---------|----------|
| BLEU | ⚡⚡⚡ | 无 | ⭐⭐ | 基线 | 快速开发 |
| chrF++ | ⚡⚡⚡ | 无 | ⭐⭐⭐ | 良好 | 多语言 |
| BERTScore | ⚡⚡ | 1.4GB | ⭐⭐⭐⭐ | 优秀 | 语义评估 |
| COMET | ⚡⚡ | 2.3GB | ⭐⭐⭐⭐⭐ | WMT2022#1 | 高质量评估 |
| GEMBA-DA | ⚡ | GPT-4 | ⭐⭐⭐⭐⭐ | WMT2023#1 | 最终评估 |
| GEMBA-MQM | ⚡ | GPT-4 | ⭐⭐⭐⭐⭐ | WMT2023 | 错误诊断 |

## 💡 推荐配置

### 开发阶段（快速迭代）
```python
suite = MetricSuite(metrics=['bleu', 'chrf'])
```

### 验证阶段（平衡质量）
```python
suite = MetricSuite(metrics=['bleu', 'chrf', 'comet'])
```

### 最终评估（WMT标准）
```python
suite = MetricSuite(
    metrics=['comet', 'gemba'],
    gemba_method='GEMBA-DA'
)
```

### 错误分析（详细诊断）
```python
gemba_mqm = GEMBAMetric(method="GEMBA-MQM", model="gpt-4")
```

## 🔧 安装依赖

```bash
# 安装所有依赖
pip install -r requirements.txt

# 或分别安装
pip install sacrebleu              # BLEU, chrF++
pip install bert-score             # BERTScore
pip install unbabel-comet          # COMET
# GEMBA使用现有的OpenAI API配置
```

## 📝 在实验中集成

要在现有的 `metrics.py` 中集成这些指标：

```python
from src.metrics import MetricSuite

class LegalTranslationMetrics:
    def __init__(self):
        # 原有的法律专用指标
        self.deontic_mapping = {...}
        
        # 添加现代MT指标
        self.mt_metrics = MetricSuite(
            metrics=['bleu', 'chrf', 'comet'],
            lang='zh'
        )
    
    def calculate_all_metrics(self, source, target, reference, 
                             src_lang, tgt_lang, term_table):
        # 法律专用指标
        legal_metrics = {
            'termbase_accuracy': self.calculate_termbase_accuracy(...),
            'deontic_preservation': self.calculate_deontic_preservation(...),
            'conditional_logic': self.calculate_conditional_logic_preservation(...)
        }
        
        # 现代MT指标
        mt_metrics = self.mt_metrics.compute(source, target, reference)
        
        # 合并返回
        return {**legal_metrics, **mt_metrics}
```

## 🧪 测试

运行测试脚本：

```bash
python test_metrics.py
```

测试包括：
1. ✅ 基础指标（BLEU, chrF++）
2. ✅ 指标套件
3. ⚠️  高级指标（BERTScore, COMET）- 需要下载模型
4. ⚠️  GEMBA指标 - 需要OpenAI API

## 📚 参考文献

### 论文引用

1. **GEMBA-DA** (WMT2023最佳)
   ```
   Kocmi, T., & Federmann, C. (2023).
   Large Language Models Are State-of-the-Art Evaluators of Translation Quality.
   EAMT 2023.
   ```

2. **GEMBA-MQM** (详细错误分析)
   ```
   Kocmi, T., & Federmann, C. (2023).
   GEMBA-MQM: Detecting Translation Quality Error Spans with GPT-4.
   WMT 2023.
   ```

3. **COMET** (WMT2022最佳)
   ```
   Rei, R., et al. (2020).
   COMET: A Neural Framework for MT Evaluation.
   EMNLP 2020.
   ```

### 官方链接

- **GEMBA**: https://github.com/MicrosoftTranslator/GEMBA
- **COMET**: https://github.com/Unbabel/COMET
- **BERTScore**: https://github.com/Tiiiger/bert_score
- **SacreBLEU**: https://github.com/mjpost/sacrebleu

## ⚠️ 重要说明

1. **模型下载**: BERTScore和COMET首次使用会自动下载模型（较大）
2. **GPU推荐**: BERTScore和COMET支持GPU，推荐使用以提高速度
3. **API成本**: GEMBA使用GPT-4 API，会产生费用
   - GEMBA-DA: ~500-800 tokens/条
   - GEMBA-MQM: ~800-1200 tokens/条
4. **评估时间**: GEMBA较慢，建议用于最终评估或抽样分析

## 🎯 最佳实践

1. **开发时**: 使用BLEU/chrF快速迭代
2. **验证时**: 添加COMET确保质量
3. **发布前**: 使用GEMBA-DA进行最终评估
4. **错误分析**: 使用GEMBA-MQM诊断问题

## 📞 支持

- 详细文档: `src/metrics/README.md`
- 测试脚本: `test_metrics.py`
- 官方仓库: https://github.com/MicrosoftTranslator/GEMBA

