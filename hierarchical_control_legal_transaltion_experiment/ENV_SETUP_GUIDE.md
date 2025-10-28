# 环境变量配置指南

## 🚀 快速开始

### 方式1：使用 .env 文件（推荐）⭐

```bash
# 1. 复制示例文件
cp .env.example .env

# 2. 编辑 .env 文件，填入你的 API 密钥
nano .env  # 或使用其他编辑器

# 3. 运行实验
python run_experiment.py --ablations full
```

### 方式2：命令行设置（临时）

```bash
# 仅在当前终端会话有效
export OPENAI_API_KEY='your-api-key-here'
python run_experiment.py --ablations full
```

### 方式3：系统环境变量（永久）

```bash
# 添加到 ~/.bashrc 或 ~/.zshrc
echo 'export OPENAI_API_KEY=your-api-key-here' >> ~/.bashrc
source ~/.bashrc

python run_experiment.py --ablations full
```

---

## 📝 .env 文件格式

在项目根目录创建 `.env` 文件：

```bash
# OpenAI API配置
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxx

# 可选配置
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_MODEL=gpt-4o-mini
LLM_TIMEOUT=300
LLM_MAX_CONCURRENT=10

# COMET模型配置（国内用户）
HF_ENDPOINT=https://hf-mirror.com
```

---

## 🔧 支持的环境变量

### 必需变量

| 变量名 | 说明 | 示例 |
|--------|------|------|
| `OPENAI_API_KEY` | OpenAI API密钥 | `sk-xxxxxxxxxxxxx` |

### 可选变量

| 变量名 | 说明 | 默认值 | 示例 |
|--------|------|--------|------|
| `OPENAI_BASE_URL` | API端点 | `https://api.openai.com/v1` | 火山引擎等 |
| `OPENAI_API_MODEL` | 默认模型 | `gpt-4o-mini` | `gpt-4o` |
| `LLM_TIMEOUT` | 请求超时（秒） | `300` | `600` |
| `LLM_MAX_CONCURRENT` | 最大并发数 | `10` | `20` |
| `HF_ENDPOINT` | HF模型镜像 | - | `https://hf-mirror.com` |

---

## 🌐 第三方API配置

### 火山引擎（豆包）

```bash
OPENAI_API_KEY=your-volcengine-api-key
OPENAI_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
OPENAI_API_MODEL=your-endpoint-id
```

### Azure OpenAI

```bash
OPENAI_API_KEY=your-azure-api-key
OPENAI_BASE_URL=https://your-resource.openai.azure.com/openai/deployments/your-deployment
OPENAI_API_MODEL=your-deployment-name
```

### 其他兼容OpenAI API的服务

只需设置对应的 `OPENAI_BASE_URL` 和 `OPENAI_API_KEY` 即可。

---

## 📍 .env 文件查找顺序

脚本会按以下顺序查找 .env 文件：

1. **项目根目录** - `./hierarchical_control_legal_transaltion_experiment/.env`
2. **当前工作目录** - `$(pwd)/.env`
3. **用户主目录** - `~/.env`

找到第一个存在的文件后停止搜索。

---

## 🔐 安全注意事项

### ✅ 安全做法

- ✅ 使用 `.env` 文件存储密钥（已在 .gitignore 中）
- ✅ 不要将 `.env` 文件提交到版本控制
- ✅ 使用 `.env.example` 作为模板（不含真实密钥）
- ✅ 定期轮换 API 密钥
- ✅ 限制 API 密钥的权限和额度

### ❌ 不安全做法

- ❌ 不要将密钥硬编码在代码中
- ❌ 不要将密钥提交到 git 仓库
- ❌ 不要在公开渠道分享密钥
- ❌ 不要使用过于宽松的密钥权限

---

## 🧪 验证配置

### 检查环境变量

```bash
# 方式1：在Python中检查
python -c "import os; print('OPENAI_API_KEY:', os.getenv('OPENAI_API_KEY', 'Not set')[:10] + '...')"

# 方式2：运行实验（会显示配置信息）
python run_experiment.py --samples 1
```

### 测试API连接

```bash
# 运行单个样本测试
python run_experiment.py --samples 1 --verbose
```

---

## 🐛 常见问题

### Q1: 提示"未设置 OPENAI_API_KEY"？

**解决方法：**
1. 确认 `.env` 文件存在且在正确位置
2. 确认 `.env` 文件中没有多余的空格或引号
3. 确认已安装 `python-dotenv`: `pip install python-dotenv`
4. 尝试使用命令行直接设置测试

### Q2: .env 文件未生效？

**检查清单：**
```bash
# 1. 确认文件存在
ls -la .env

# 2. 查看文件内容
cat .env

# 3. 确认格式正确（KEY=value，无引号）
# 正确: OPENAI_API_KEY=sk-xxxxx
# 错误: OPENAI_API_KEY='sk-xxxxx'  # 不需要引号
# 错误: OPENAI_API_KEY = sk-xxxxx  # 等号前后不要空格

# 4. 确认安装了 python-dotenv
pip list | grep python-dotenv
```

### Q3: 如何使用多个配置文件？

```bash
# 开发环境
cp .env.example .env.dev
# 编辑 .env.dev...

# 生产环境
cp .env.example .env.prod
# 编辑 .env.prod...

# 使用时复制到 .env
cp .env.dev .env
python run_experiment.py
```

### Q4: 环境变量优先级？

优先级（从高到低）：
1. **命令行export** - 最高优先级
2. **系统环境变量** - 已存在的系统变量
3. **.env 文件** - 从文件加载（override=False）

---

## 📚 相关依赖

### 安装 python-dotenv

```bash
# 方式1：单独安装
pip install python-dotenv

# 方式2：从requirements.txt安装（如果包含）
pip install -r requirements.txt
```

### 添加到 requirements.txt

```
python-dotenv>=1.0.0
```

---

## 💡 最佳实践

### 1. 使用 .env 文件管理配置

```bash
# 项目结构
project/
├── .env                 # 实际配置（不提交）
├── .env.example         # 配置模板（提交）
├── .gitignore           # 包含 .env
└── run_experiment.py
```

### 2. 分环境配置

```bash
# 开发环境
.env.dev

# 测试环境
.env.test

# 生产环境
.env.prod
```

### 3. 使用脚本切换环境

```bash
#!/bin/bash
# switch_env.sh

if [ "$1" == "dev" ]; then
    cp .env.dev .env
    echo "✓ 切换到开发环境"
elif [ "$1" == "prod" ]; then
    cp .env.prod .env
    echo "✓ 切换到生产环境"
fi
```

---

## 📞 故障排查

如果遇到问题：

1. **查看启动日志**
   ```bash
   python run_experiment.py
   # 应该看到: "✓ 已加载环境配置: /path/to/.env"
   ```

2. **手动验证**
   ```bash
   python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(os.getenv('OPENAI_API_KEY')[:10])"
   ```

3. **使用verbose模式**
   ```bash
   python run_experiment.py --verbose --samples 1
   ```

---

## 📄 示例配置文件

### 基础配置（OpenAI官方）

```bash
# .env
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxx
```

### 完整配置（自定义端点）

```bash
# .env
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://your-endpoint.com/v1
OPENAI_API_MODEL=gpt-4o
LLM_TIMEOUT=600
LLM_MAX_CONCURRENT=20
HF_ENDPOINT=https://hf-mirror.com
```

### 国内镜像配置

```bash
# .env
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxx
HF_ENDPOINT=https://hf-mirror.com
```

---

**更新日期**: 2024-10-12  
**版本**: v1.0

