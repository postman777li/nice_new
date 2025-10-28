# 机器翻译评估指标模块

本模块实现了最新的机器翻译评估指标（2024），用于全面评估翻译质量。

## 📊 支持的指标

### 1. BLEU (Bilingual Evaluation Understudy)
- **类型**: 传统n-gram重叠指标
- **范围**: 0-100
- **特点**: 
  - 快速计算
  - 基于精确匹配
  - 适合作为基线
- **使用场景**: 快速评估、基线对比

### 2. chrF / chrF++
- **类型**: 字符级F-score
- **范围**: 0-100
- **特点**:
  - 字符级n-gram
  - 特别适合中文、日语等语言
  - 不需要分词
- **使用场景**: 多语言评估、形态丰富语言

### 3. BERTScore
- **类型**: 基于预训练模型的语义相似度
- **范围**: 0-1
- **特点**:
  - 使用XLM-RoBERTa等模型
  - 捕捉语义相似性
  - 支持多语言
- **使用场景**: 语义质量评估

### 4. COMET
- **类型**: 神经网络翻译质量估计
- **范围**: 约0-1（可能超出）
- **特点**:
  - WMT2022最佳指标
  - 高度相关人类判断
  - 需要源文本和参考
- **使用场景**: 高质量评估、与人类判断对齐

### 5. GEMBA (MQM & DA)
- **类型**: 基于LLM的翻译质量评估
- **范围**: 0-100
- **官方实现**: [Microsoft/GEMBA](https://github.com/MicrosoftTranslator/GEMBA)
- **两种方法**:
  - **GEMBA-MQM**: 基于MQM框架的详细错误检测和评分
    - 识别翻译错误并分类（Accuracy, Fluency, Terminology, Style, Locale）
    - 标注错误严重程度（Minor, Major, Critical）
    - 使用MQM公式计算分数（100分制，扣分制）
  - **GEMBA-DA**: 直接评估（Direct Assessment）
    - 直接输出0-100的质量分数
    - 更快速，适合批量评估
    - 基于准确性、流畅性、充分性评估
- **特点**:
  - 使用GPT-4（WMT2023最佳评估指标）
  - 最接近人工评估（人类相关性最高）
  - 提供可解释的评估结果
- **使用场景**: 
  - GEMBA-MQM: 详细错误分析、质量诊断
  - GEMBA-DA: 快速质量评估、与人类判断对齐

## 🚀 快速开始

### 安装依赖

```bash
# 基础指标
pip install sacrebleu

# BERTScore
pip install bert-score

# COMET
pip install unbabel-comet

# 所有指标
pip install -r requirements.txt
```

### 基本使用

```python
from src.metrics import MetricSuite

# 创建指标套件
suite = MetricSuite(metrics=['bleu', 'chrf', 'bertscore', 'comet'])

# 计算分数
scores = suite.compute(
    source="合同双方应当遵守本协议的所有条款。",
    prediction="The parties shall comply with all terms of this agreement.",
    reference="Contracting parties must comply with all provisions of this agreement."
)

print(scores)
# 输出: {
#   'bleu': 45.23,
#   'chrf': 67.89,
#   'bertscore_f1': 0.8234,
#   'comet': 0.7654
# }
```

### 单个指标使用

```python
from src.metrics import BLEUMetric, ChrFMetric, COMETMetric, GEMBAMetric

# BLEU
bleu = BLEUMetric(tokenize='zh')
score = bleu.sentence_score(prediction, reference)

# chrF++
chrf = ChrFMetric()
score = chrf.sentence_score(prediction, reference)

# COMET
comet = COMETMetric()
score = comet.sentence_score(source, prediction, reference)

# GEMBA-DA (推荐用于快速评估)
gemba_da = GEMBAMetric(method="GEMBA-DA", model="gpt-4")
score = gemba_da.sentence_score(source, prediction, "Chinese", "English")

# GEMBA-MQM (详细错误分析)
gemba_mqm = GEMBAMetric(method="GEMBA-MQM", model="gpt-4")
result = gemba_mqm.compute([source], [prediction], "Chinese", "English")
print(f"分数: {result['mean']}")
print(f"错误: {result['results'][0]['errors']}")
```

## 📝 在实验中使用

### 更新 metrics.py

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
        
        # 合并
        return {**legal_metrics, **mt_metrics}
```

### 在实验中使用

```python
# run_experiment.py 会自动使用新指标
python run_experiment.py \
  --samples 50 \
  --ablations baseline terminology terminology_syntax full
```

## 🎯 指标选择建议

### 快速评估（推荐）
```python
suite = MetricSuite(metrics=['bleu', 'chrf'])
```
- 计算快速
- 资源占用少
- 适合大规模实验

### 平衡评估（推荐）
```python
suite = MetricSuite(metrics=['bleu', 'chrf', 'comet'])
```
- 包含传统和神经指标
- 计算速度适中
- 覆盖多个维度

### 完整评估
```python
suite = MetricSuite(
    metrics=['bleu', 'chrf', 'bertscore', 'comet', 'gemba'],
    gemba_method='GEMBA-DA'  # 或 'GEMBA-MQM'
)
```
- 最全面的评估
- 计算时间较长
- 适合最终评估
- GEMBA-DA更快，GEMBA-MQM提供详细错误分析

## 📊 指标对比

| 指标 | 速度 | GPU | 资源 | 与人类相关性 | 适用场景 |
|------|------|-----|------|-------------|----------|
| BLEU | ⚡⚡⚡ | ❌ | 无 | ⭐⭐ | 基线、快速 |
| chrF++ | ⚡⚡⚡ | ❌ | 无 | ⭐⭐⭐ | 多语言 |
| BERTScore | ⚡⚡ | ✅ | 1.4GB | ⭐⭐⭐⭐ | 语义评估 |
| COMET | ⚡⚡ | ✅ | 2.3GB | ⭐⭐⭐⭐⭐ | WMT2022最佳 |
| GEMBA-DA | ⚡ | ❌ | GPT-4 | ⭐⭐⭐⭐⭐ | WMT2023最佳 |
| GEMBA-MQM | ⚡ | ❌ | GPT-4 | ⭐⭐⭐⭐⭐ | 错误诊断 |

## 🔧 高级配置

### 自定义COMET模型

```python
from src.metrics import COMETMetric

# 使用不同的COMET模型
comet = COMETMetric(
    model_name="Unbabel/XCOMET-XXL",  # 最大模型
    gpus=1  # 使用GPU
)
```

### 自定义BERTScore模型

```python
from src.metrics import BERTScoreMetric

# 中文特化模型
bertscore = BERTScoreMetric(
    model_type="bert-base-chinese",
    lang="zh"
)
```

### 使用GEMBA指标

```python
from src.metrics import GEMBAMetric

# 方法1: GEMBA-DA (推荐，更快速)
gemba_da = GEMBAMetric(
    method="GEMBA-DA",
    model="gpt-4",
    temperature=0.1  # 官方推荐
)

score = gemba_da.sentence_score(
    source="合同双方应当遵守本协议的所有条款。",
    prediction="The parties shall comply with all terms.",
    source_lang="Chinese",
    target_lang="English"
)
print(f"GEMBA-DA: {score:.2f}/100")

# 方法2: GEMBA-MQM (详细错误分析)
gemba_mqm = GEMBAMetric(
    method="GEMBA-MQM",
    model="gpt-4"
)

result = gemba_mqm.compute(
    sources=["合同双方应当遵守本协议的所有条款。"],
    predictions=["The parties shall comply with all terms."],
    source_lang="Chinese",
    target_lang="English"
)

print(f"GEMBA-MQM: {result['mean']:.2f}/100")
print(f"错误详情: {result['results'][0]['errors']}")
print(f"错误统计: {result['results'][0]['error_count']}")
```

### 批量计算

```python
sources = ["文本1", "文本2", "文本3"]
predictions = ["翻译1", "翻译2", "翻译3"]
references = ["参考1", "参考2", "参考3"]

# 批量COMET
comet = COMETMetric()
result = comet.compute(sources, predictions, references)
print(f"系统级COMET: {result['system_score']:.4f}")

# 批量GEMBA-DA
gemba = GEMBAMetric(method="GEMBA-DA")
result = gemba.compute(sources, predictions, "Chinese", "English")
print(f"系统级GEMBA-DA: {result['system_score']:.2f}/100")
```

## ⚠️ 注意事项

1. **首次使用**: BERTScore和COMET首次使用会下载模型（较大）
   - **已内置镜像加速**: 默认使用 `hf-mirror.com` 加速下载
   - xlm-roberta-large: ~1.4GB
   - wmt22-comet-da: ~2.3GB
2. **GPU加速**: BERTScore和COMET支持GPU，建议使用以提高速度
3. **内存占用**: COMET-XXL等大模型需要较多内存
4. **API调用**: GEMBA指标使用GPT-4 API，会产生费用
   - GEMBA-DA: 每条约500-800 tokens
   - GEMBA-MQM: 每条约800-1200 tokens（更详细）
5. **语言支持**: 
   - GEMBA支持任何语言对（通过语言名称指定）
   - 其他指标确保为目标语言选择合适的模型
6. **温度参数**: GEMBA官方推荐temperature=0.1（已默认设置）
7. **镜像加速**: 
   - 国内用户默认启用 HF 镜像加速（`use_hf_mirror=True`）
   - 可通过参数禁用：`BERTScoreMetric(use_hf_mirror=False)`

## 📚 参考文献

- **BLEU**: [Papineni et al., 2002](https://aclanthology.org/P02-1040/) - 传统n-gram精确匹配
- **chrF**: [Popović, 2015](https://aclanthology.org/W15-3049/) - 字符级F-score
- **BERTScore**: [Zhang et al., 2020](https://arxiv.org/abs/1904.09675) - 基于BERT的语义相似度
- **COMET**: [Rei et al., 2020](https://arxiv.org/abs/2009.09025) - WMT2022最佳指标
- **GEMBA-DA**: [Kocmi & Federmann, 2023](https://aclanthology.org/2023.eamt-1.19) - "Large Language Models Are State-of-the-Art Evaluators of Translation Quality"
- **GEMBA-MQM**: [Kocmi & Federmann, 2023](https://arxiv.org/abs/2310.13988) - "GEMBA-MQM: Detecting Translation Quality Error Spans with GPT-4"
- **官方实现**: [Microsoft/GEMBA](https://github.com/MicrosoftTranslator/GEMBA)

## 🛠️ 故障排查

### 问题：模型下载速度慢（推荐配置）

**方法1: 使用内置镜像（推荐，自动启用）**
```python
# 所有指标默认已启用HF镜像加速
from src.metrics import BERTScoreMetric, COMETMetric, MetricSuite

# 自动使用 hf-mirror.com 镜像
bertscore = BERTScoreMetric()  # use_hf_mirror=True 默认
comet = COMETMetric()  # use_hf_mirror=True 默认
suite = MetricSuite()  # use_hf_mirror=True 默认
```

**方法2: 手动设置环境变量**
```bash
# 在命令行中设置（临时）
export HF_ENDPOINT=https://hf-mirror.com

# 或添加到 ~/.bashrc 或 ~/.zshrc（永久）
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc

# 使用配置脚本
source setup_hf_mirror.sh
```

**方法3: 在代码中设置**
```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

### 问题：COMET模型下载失败
```bash
# 使用镜像手动下载
export HF_ENDPOINT=https://hf-mirror.com
python -c "from comet import download_model; download_model('Unbabel/wmt22-comet-da')"
```

### 问题：BERTScore速度慢
```python
# 使用更小的模型
bertscore = BERTScoreMetric(model_type="bert-base-multilingual-cased")
```

### 问题：内存不足
```python
# 减少batch size
comet = COMETMetric(batch_size=4, gpus=0)  # 使用CPU
```

### 问题：GEMBA API调用失败
```python
# 确保设置了OpenAI API密钥
# 在 .env 文件中:
# OPENAI_API_KEY=your_api_key

# 或在代码中:
import os
os.environ['OPENAI_API_KEY'] = 'your_api_key'
```

### 问题：GEMBA评估速度慢
```python
# 方案1: 使用GEMBA-DA而非GEMBA-MQM（更快）
gemba = GEMBAMetric(method="GEMBA-DA")

# 方案2: 减少样本数量
# 方案3: 使用异步批量评估
results = await gemba.compute_async(sources, predictions, "Chinese", "English")
```

## 💡 最佳实践

### 指标组合推荐

1. **开发阶段** (快速迭代)
   ```python
   suite = MetricSuite(metrics=['bleu', 'chrf'])
   ```

2. **验证阶段** (平衡质量)
   ```python
   suite = MetricSuite(metrics=['bleu', 'chrf', 'comet'])
   ```

3. **最终评估** (完整分析)
   ```python
   suite = MetricSuite(
       metrics=['bleu', 'chrf', 'comet', 'gemba'],
       gemba_method='GEMBA-DA'
   )
   ```

4. **错误分析** (详细诊断)
   ```python
   gemba_mqm = GEMBAMetric(method="GEMBA-MQM")
   # 提供详细的错误分类和严重程度
   ```

### WMT标准组合

根据WMT2023评估任务：
```python
# 官方推荐的指标组合
suite = MetricSuite(metrics=['comet', 'gemba'])
# COMET: 神经网络评估
# GEMBA-DA: LLM评估
# 两者结合可达到最高的人类判断相关性
```

## 📞 支持

如有问题，请查看：
- [官方文档](../../README.md)
- [Issue tracker](https://github.com/MicrosoftTranslator/GEMBA/issues)
- [GEMBA官方仓库](https://github.com/MicrosoftTranslator/GEMBA)

