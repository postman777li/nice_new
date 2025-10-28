# 层次化控制法律翻译实验

一个基于 Agent 架构的高质量法律翻译系统，采用三层层次化控制策略：术语层、句法层和篇章层。

## 📋 目录

- [项目简介](#项目简介)
- [核心特性](#核心特性)
- [系统架构](#系统架构)
- [安装指南](#安装指南)
- [快速开始](#快速开始)
- [详细使用](#详细使用)
- [数据库配置](#数据库配置)
- [项目结构](#项目结构)
- [开发指南](#开发指南)

## 项目简介

本项目实现了一个创新的法律翻译系统，通过三层智能体架构实现对翻译质量的精细化控制：

1. **术语层（Terminology Layer）**：确保法律术语的准确性和一致性
2. **句法层（Syntax Layer）**：保证句法结构和法律表达的保真度
3. **篇章层（Discourse Layer）**：维护翻译风格和上下文的一致性

每一层都采用"提取-评估-翻译"的三步工作流，通过智能体之间的协作实现高质量的法律翻译。

## 核心特性

### 🎯 三层架构
- **术语控制**：基于术语库的专业术语管理
- **句法控制**：双语句法模式提取与保真度评估
- **篇章控制**：翻译记忆检索与风格一致性分析

### 🤖 智能体系统
每层包含3个核心智能体：
- **提取智能体（Extract Agent）**：识别关键特征和模式
- **评估智能体（Evaluate Agent）**：分析质量并发现问题
- **翻译智能体（Translation Agent）**：基于评估结果改进翻译

### 🗄️ 多数据库支持
- **SQLite**：术语库存储和管理
- **Milvus**：向量检索（术语、翻译记忆）
- **BM25**：文本检索（混合检索策略）

### 📊 质量保证
- 实时评估和反馈
- 详细的问题诊断
- 可追溯的改进建议
- 完整的翻译轨迹记录

## 系统架构

```
输入文本
    ↓
┌─────────────────────────────────────┐
│  第一轮：术语层 (Terminology)        │
├─────────────────────────────────────┤
│  1. MonoExtract: 提取术语            │
│  2. Evaluate: 评估术语翻译质量       │
│  3. Translation: 生成术语优化翻译    │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  第二轮：句法层 (Syntax)             │
├─────────────────────────────────────┤
│  1. BiExtract: 提取双语句法模式      │
│  2. Evaluate: 评估句法保真度         │
│  3. Translation: 基于评估改进句法    │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  第三轮：篇章层 (Discourse)          │
├─────────────────────────────────────┤
│  1. Query: 检索相关翻译记忆          │
│  2. Evaluate: 分析与参考的差异       │
│  3. Translation: 基于差异调整风格    │
└─────────────────────────────────────┘
    ↓
输出最终翻译
```

### 差异分析示例（篇章层）

篇章评估智能体会分析当前翻译与历史高质量翻译的差异：

**用词差异**：
- 当前翻译使用 "agreement"，参考翻译使用 "contract"
- 当前翻译使用 "must"，参考翻译统一使用 "shall"

**句法差异**：
- 当前翻译使用主动语态，参考翻译多用被动语态
- 当前翻译条件句使用 "if...then"，参考翻译使用 "where"

**改进建议**：
- 建议将 "agreement" 改为 "contract" 以保持一致
- 建议使用被动语态以符合参考风格

## 安装指南

### 环境要求

- Python 3.8+
- Milvus 2.3+ (用于向量检索)
- 足够的磁盘空间（用于存储翻译记忆和术语库）

### 安装步骤

1. **克隆仓库**
```bash
git clone <repository-url>
cd hierarchical_control_legal_transaltion_experiment
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **配置 Milvus**

启动 Milvus 服务（使用 Docker）：
```bash
docker-compose up -d
```

或参考 [Milvus 官方文档](https://milvus.io/docs/install_standalone-docker.md)

4. **初始化数据库**
```bash
# 创建术语库集合
python scripts/reset_milvus_collections.py

# 导入术语数据（可选）
python scripts/import_terms_to_db.py

# 导入翻译记忆（可选）
python scripts/import_tm_to_db.py
```

## 快速开始

### 基本翻译

```bash
python run_translation.py \
  --source "合同双方应当遵守本协议的所有条款。" \
  --src-lang zh \
  --tgt-lang en \
  --verbose
```

### 批量翻译

```bash
python run_translation.py \
  --input dataset/processed/test_set_zh_en.json \
  --output results/translations.json \
  --src-lang zh \
  --tgt-lang en
```

### 配置选项

```bash
python run_translation.py \
  --source "当事人有权依法申请行政复议或者提起行政诉讼。" \
  --src-lang zh \
  --tgt-lang en \
  --hierarchical \          # 启用三层架构（默认）
  --use-termbase \          # 使用术语库（默认）
  --use-tm \                # 使用翻译记忆
  --verbose                 # 显示详细输出
```

## 详细使用

### 1. 术语层工作流

术语层确保法律术语的准确翻译：

```python
from src.workflows.terminology import run_terminology_workflow

result = await run_terminology_workflow(
    orchestrator=None,
    job_id="job_001",
    config=config,
    input_text="原文文本"
)
```

**输出示例**：
```
📝 提取术语...
  提取到 8 个术语

【详细】提取的术语:
  1. 合同 (contract) - 置信度: 0.95
  2. 当事人 (party) - 置信度: 0.92
  3. 违约责任 (liability for breach) - 置信度: 0.88

🔍 评估术语...
  评估完成 (总分: 0.85)

【详细】术语评估:
  准确性: 0.90
  一致性: 0.85
  完整性: 0.80
```

### 2. 句法层工作流

句法层保证法律句式的准确性：

```python
from src.workflows.syntax import run_syntactic_workflow

result = await run_syntactic_workflow(
    orchestrator=None,
    job_id="job_001",
    config=config,
    input_text="第一轮翻译结果"
)
```

**输出示例**：
```
📝 提取句法模式...
  提取到 5 个句法模式

【详细】句法模式:
  1. 应当 → shall (情态动词, 置信度: 0.95)
  2. 可以 → may (情态动词, 置信度: 0.92)

🔍 评估句法...
  评估完成 (总分: 0.78)

【详细】句法评估:
  情态动词保真度: 0.85
  连接词一致性: 0.75
  条件逻辑维护: 0.75
  发现问题: 情态动词 "must" 应改为 "shall"
```

### 3. 篇章层工作流

篇章层通过参考历史翻译保持风格一致性：

```python
from src.workflows.discourse import run_discourse_workflow

result = await run_discourse_workflow(
    orchestrator=None,
    job_id="job_001",
    config=config,
    input_text="第二轮翻译结果"
)
```

**输出示例**：
```
📝 检索翻译记忆...
  找到 5 个相关翻译记忆

【详细】翻译记忆:
  1. 相似度: 0.88
     源文: 双方应当按照约定履行义务...
     译文: The parties shall perform obligations...

🔍 分析当前翻译与参考例子的差异...
  分析完成 (总分: 0.82)

【详细】篇章一致性分析:
  用词一致性: 0.85
  句法一致性: 0.80
  风格一致性: 0.80
  用词差异: 当前使用 "must"，参考使用 "shall"
  句法差异: 当前使用主动语态，参考多用被动语态
```

## 数据库配置

### 术语库（SQLite）

术语库位于 `terms_zh_en.db`，包含：
- 术语条目（源语言术语、目标语言术语、领域、释义等）
- 术语关系（同义词、相关术语等）

查看术语库：
```bash
python scripts/check_terms_db.py
```

导入新术语：
```bash
python scripts/import_terms_to_db.py --input terms.json
```

### 翻译记忆（Milvus + BM25）

翻译记忆使用混合检索：
- **向量检索（Milvus）**：基于语义相似度
- **BM25 检索**：基于关键词匹配

导入翻译记忆：
```bash
python scripts/import_tm_to_db.py --input dataset/processed/train_set_zh_en.json
```

### 向量数据库集合

系统使用以下 Milvus 集合：
- `legal_terms_zh_en`：中英术语向量
- `legal_tm_zh_en`：中英翻译记忆
- `legal_tm_zh_ja`：中日翻译记忆

重置集合：
```bash
python scripts/reset_milvus_collections.py
```

## 项目结构

```
hierarchical_control_legal_transaltion_experiment/
├── run_translation.py              # 主翻译脚本
├── run_experiment.py               # 批量实验脚本
├── requirements.txt                # 依赖列表
├── configs/
│   └── default.yaml                # 默认配置
├── src/
│   ├── agents/                     # 智能体模块
│   │   ├── terminology/            # 术语层智能体
│   │   │   ├── mono_extract.py    # 单语术语提取
│   │   │   ├── evaluate.py        # 术语评估
│   │   │   └── translation.py     # 术语翻译
│   │   ├── syntax/                 # 句法层智能体
│   │   │   ├── bi_extract.py      # 双语句法提取
│   │   │   ├── syntax_evaluate.py # 句法评估
│   │   │   └── syntax_translation.py # 句法翻译
│   │   └── discourse/              # 篇章层智能体
│   │       ├── discourse_query.py  # 翻译记忆查询
│   │       ├── discourse_evaluate.py # 篇章一致性分析
│   │       └── discourse_translation.py # 篇章整合
│   ├── lib/                        # 核心库
│   │   ├── llm_client.py          # LLM 客户端
│   │   ├── vector_db.py           # 向量数据库
│   │   ├── term_db.py             # 术语库
│   │   └── tm_db.py               # 翻译记忆库
│   └── workflows/                  # 工作流
│       ├── terminology.py          # 术语层工作流
│       ├── syntax.py               # 句法层工作流
│       └── discourse.py            # 篇章层工作流
├── scripts/                        # 工具脚本
│   ├── import_terms_to_db.py      # 导入术语
│   ├── import_tm_to_db.py         # 导入翻译记忆
│   ├── bi_term_extract.py         # 双语术语提取
│   └── reset_milvus_collections.py # 重置 Milvus
├── dataset/                        # 数据集
│   └── processed/                  # 处理后的数据
│       ├── train_set_zh_en.json   # 训练集
│       └── test_set_zh_en.json    # 测试集
└── docs/                           # 文档
    ├── README_terminology_import.md
    └── README_bilingual_extract.md
```

## 开发指南

### 添加新的智能体

1. 在相应的层级目录下创建新的智能体文件
2. 继承 `BaseAgent` 类
3. 实现 `execute()` 方法
4. 在 `__init__.py` 中导出

示例：
```python
from ..base import BaseAgent, AgentConfig

class MyAgent(BaseAgent):
    def __init__(self, locale: str = 'zh'):
        super().__init__(AgentConfig(
            name='my:agent',
            role='my_role',
            domain='terminology',
            specialty='我的专长',
            quality='review',
            locale=locale
        ))
    
    async def execute(self, input_data: Dict[str, Any], ctx: Optional[AgentRunContext] = None):
        # 实现智能体逻辑
        pass
```

### 修改工作流

工作流定义在 `src/workflows/` 目录下，每个工作流对应一个层级。修改工作流时需要确保：

1. 输入输出格式一致
2. 错误处理完善
3. 日志输出清晰
4. 支持 verbose 模式

### 自定义评估指标

评估指标定义在各个 evaluate agent 中。可以通过修改 prompt 或添加新的评估维度来自定义评估标准。

### 测试

运行测试：
```bash
# 测试单个翻译
python run_translation.py --source "测试文本" --src-lang zh --tgt-lang en

# 测试批量处理
python test/test_batch_processing.py

# 测试数据导入
python test/test_import.py
```

## 配置说明

### 环境变量

创建 `.env` 文件并设置以下变量：

```bash
# LLM API 配置
OPENAI_API_KEY=your_api_key
OPENAI_API_BASE=https://api.openai.com/v1
MODEL_NAME=gpt-4

# Milvus 配置
MILVUS_HOST=localhost
MILVUS_PORT=19530

# 向量嵌入配置
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIM=1536
```

### 配置文件

在 `configs/default.yaml` 中配置系统参数：

```yaml
translation:
  hierarchical: true        # 启用层次化架构
  use_termbase: true        # 使用术语库
  use_tm: false            # 使用翻译记忆
  use_rules: false         # 使用规则表
  
models:
  llm_model: "gpt-4"
  embedding_model: "text-embedding-3-small"
  
database:
  milvus_host: "localhost"
  milvus_port: 19530
```

## 常见问题

### Q: 如何提高翻译质量？

A: 
1. 确保术语库数据完整
2. 导入更多高质量的翻译记忆
3. 在 verbose 模式下查看详细评估结果
4. 根据评估建议调整配置

### Q: 如何添加新的语言对？

A:
1. 准备该语言对的术语库
2. 准备该语言对的翻译记忆
3. 在代码中添加相应的语言标识符
4. 导入数据到 Milvus

### Q: 系统运行缓慢怎么办？

A:
1. 检查 Milvus 服务状态
2. 减少 top_k 参数值
3. 考虑使用更快的嵌入模型
4. 启用缓存机制

### Q: 如何调试智能体？

A:
1. 使用 `--verbose` 参数查看详细输出
2. 检查日志文件
3. 在智能体代码中添加断点
4. 使用 trace 信息追踪执行流程

## 许可证

[待添加]

## 贡献

欢迎提交 Issue 和 Pull Request！

## 引用

如果您在研究中使用了本项目，请引用：

```bibtex
[待添加]
```

## 联系方式

[待添加]

---

**注意**：本项目仍在积极开发中，API 可能会有变动。
