# 贡献指南 / Contributing Guide

感谢您对本项目的关注！我们欢迎所有形式的贡献。

Thank you for your interest in contributing to this project! We welcome all forms of contributions.

## 如何贡献 / How to Contribute

### 报告问题 / Reporting Issues

如果您发现了bug或有功能建议，请：
1. 检查是否已有类似的issue
2. 创建新的issue，详细描述问题或建议
3. 如果可能，提供复现步骤

If you find a bug or have a feature suggestion:
1. Check if a similar issue already exists
2. Create a new issue with a detailed description
3. Provide reproduction steps if possible

### 提交代码 / Submitting Code

1. **Fork 项目 / Fork the Repository**
   ```bash
   git clone https://github.com/your-username/hierarchical_control_legal_transaltion_experiment.git
   cd hierarchical_control_legal_transaltion_experiment
   ```

2. **创建分支 / Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # 或 / or
   git checkout -b fix/your-bug-fix
   ```

3. **开发和测试 / Develop and Test**
   - 遵循项目的代码风格
   - 添加必要的测试
   - 确保所有测试通过
   
   Follow the project's code style, add necessary tests, and ensure all tests pass.

4. **提交更改 / Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add new feature" # or "fix: bug description"
   ```

   提交信息格式 / Commit Message Format:
   - `feat:` 新功能 / new feature
   - `fix:` 修复bug / bug fix
   - `docs:` 文档更新 / documentation
   - `style:` 代码格式 / formatting
   - `refactor:` 重构 / refactoring
   - `test:` 测试 / testing
   - `chore:` 其他 / others

5. **推送和创建PR / Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   然后在GitHub上创建Pull Request。
   
   Then create a Pull Request on GitHub.

## 代码规范 / Code Standards

### Python 代码风格 / Python Code Style

- 遵循 PEP 8 规范
- 使用类型提示 (Type Hints)
- 添加适当的文档字符串 (Docstrings)
- 保持函数简洁，单一职责

Follow PEP 8, use type hints, add docstrings, and keep functions concise.

示例 / Example:
```python
from typing import Dict, Any, Optional

async def translate_text(
    source: str,
    src_lang: str,
    tgt_lang: str,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """翻译文本
    
    Args:
        source: 源文本
        src_lang: 源语言代码
        tgt_lang: 目标语言代码
        config: 可选配置
        
    Returns:
        包含翻译结果的字典
    """
    # 实现代码
    pass
```

### 智能体开发 / Agent Development

新增智能体时应该：
- 继承 `BaseAgent` 类
- 实现 `execute()` 方法
- 提供清晰的输入输出格式
- 添加详细的文档

When adding new agents:
- Inherit from `BaseAgent`
- Implement the `execute()` method
- Provide clear input/output formats
- Add detailed documentation

### 测试 / Testing

- 为新功能添加单元测试
- 确保测试覆盖率 > 80%
- 测试文件命名: `test_*.py`

Add unit tests for new features, maintain >80% coverage, and name test files as `test_*.py`.

## 开发环境设置 / Development Setup

1. **安装依赖 / Install Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # 如果有
   ```

2. **配置环境变量 / Configure Environment**
   ```bash
   cp .env.example .env
   # 编辑 .env 文件，填入必要的配置
   ```

3. **启动 Milvus / Start Milvus**
   ```bash
   docker-compose up -d
   ```

4. **运行测试 / Run Tests**
   ```bash
   python -m pytest tests/
   ```

## 项目结构 / Project Structure

```
src/
├── agents/          # 智能体模块
│   ├── terminology/ # 术语层
│   ├── syntax/      # 句法层
│   └── discourse/   # 篇章层
├── lib/            # 核心库
├── workflows/      # 工作流
└── models.py       # 数据模型
```

## Pull Request 检查清单 / PR Checklist

在提交PR前，请确保：
- [ ] 代码遵循项目规范
- [ ] 添加了必要的测试
- [ ] 所有测试通过
- [ ] 更新了相关文档
- [ ] 提交信息清晰明确
- [ ] 没有合并冲突

Before submitting a PR, ensure:
- [ ] Code follows project standards
- [ ] Added necessary tests
- [ ] All tests pass
- [ ] Updated relevant documentation
- [ ] Clear commit messages
- [ ] No merge conflicts

## 社区准则 / Community Guidelines

- 尊重所有贡献者
- 保持友好和专业
- 提供建设性的反馈
- 欢迎新手参与

Be respectful, friendly, professional, provide constructive feedback, and welcome newcomers.

## 许可证 / License

通过贡献代码，您同意您的贡献将在MIT许可证下发布。

By contributing, you agree that your contributions will be licensed under the MIT License.

## 联系方式 / Contact

如有问题，欢迎通过以下方式联系：
- 创建 Issue
- 发送邮件到：[待添加]

For questions, feel free to:
- Create an Issue
- Email: [To be added]

---

再次感谢您的贡献！🎉

Thank you again for your contribution! 🎉

