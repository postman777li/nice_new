"""
句法评估Agent - 评估句法保真度
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import logging

from ..base import BaseAgent, AgentConfig, AgentRunContext

logger = logging.getLogger(__name__)


@dataclass
class SyntaxEvaluateResult:
    modality_preservation: float      # 情态动词准确性
    connective_consistency: float     # 连接词逻辑
    conditional_logic: float          # 条件句规范性
    passive_voice_appropriateness: float  # 被动语态适当性
    overall_score: float
    recommendations: List[str]
    issues: List[str]
    # 新增：具体问题标记
    modality_issues: List[Dict[str, str]] = field(default_factory=list)  # 情态动词问题
    connective_issues: List[Dict[str, str]] = field(default_factory=list)  # 连接词问题
    conditional_issues: List[Dict[str, str]] = field(default_factory=list)  # 条件句问题
    passive_issues: List[Dict[str, str]] = field(default_factory=list)  # 被动语态问题


class SyntaxEvaluateAgent(BaseAgent):
    def __init__(self, locale: str = 'zh'):
        super().__init__(AgentConfig(
            name='syntax:syntax-evaluate',
            role='syntax_evaluator',
            domain='syntax',
            specialty='句法保真度评估',
            quality='review',
            locale=locale
        ))

    async def execute(self, input_data: Dict[str, Any], ctx: Optional[AgentRunContext] = None) -> SyntaxEvaluateResult:
        """评估句法保真度"""
        source_text = input_data.get('source_text', '')
        target_text = input_data.get('target_text', '')
        patterns = input_data.get('patterns', [])
        source_lang = input_data.get('source_lang', 'zh')
        target_lang = input_data.get('target_lang', 'en')
        
        if not source_text or not target_text:
            return SyntaxEvaluateResult(
                modality_preservation=0.0,
                connective_consistency=0.0,
                conditional_logic=0.0,
                overall_score=0.0,
                recommendations=[],
                issues=[]
            )
        
        # 使用LLM评估句法保真度（聚焦具体检查）
        messages = [
            {
                "role": "system",
                "content": f"""你是一个法律句法评估专家。你的任务是**具体检查**翻译中的句法问题。

## 🎯 核心任务：具体检查4个维度

### 1️⃣ 情态动词准确性（Modality Preservation）
**检查内容：**
- ✅ 是否有**缺失**？（源文有"应当/必须/可以"，译文没有对应情态词，但也要考虑源语言的省略）
- ✅ 是否准确表达了源语言的**情态义务强度**？
  - "必须" → must/shall（强制）
  - "应当" → shall（法律义务）
  - "可以" → may（许可）
  - "不得" → shall not/may not（禁止）
- ✅ 情态动词是否与法律语境匹配？

**具体输出：**
```json
"modality_issues": [
  {{
    "source": "应当",
    "target": "should",  // 错误，应该用shall
    "location": "第1句",
    "problem": "法律义务应使用shall而非should",
    "severity": "high"  // high/medium/low
  }}
]
```

### 2️⃣ 连接词逻辑（Connective Consistency）
**检查内容：**
- ✅ 并列关系（和/及/以及）→ and
- ✅ 选择关系（或者）→ or
- ✅ 转折关系（但是/然而）→ but/however
- ✅ 因果关系（因此/所以）→ therefore/thus
- ✅ 条件关系（如果/若）→ if/where/when
- ❌ **逻辑错误**：因果误译为转折，并列误译为选择等

**具体输出：**
```json
"connective_issues": [
  {{
    "source": "但是",
    "target": "and",  // 错误，转折关系译成并列
    "logic_type": "adversative → additive",
    "problem": "转折关系误译为并列关系"
  }}
]
```

### 3️⃣ 条件句规范性（Conditional Logic）
**检查内容：**
- ✅ 条件从句是否完整？（没有遗漏条件）
- ✅ 条件引导词是否符合**目标语言法律表达**？
  - 中文："如果...，应当..." → 英文："Where..., ...shall..."（法律正式）
  - 或 "If..., ...shall..."（稍次正式）
  - ❌ 不宜用 "When"表示假设条件
- ✅ 条件逻辑关系是否清晰？

**具体输出：**
```json
"conditional_issues": [
  {{
    "source_pattern": "如果...，应当...",
    "target_pattern": "When..., should...",
    "problem": "1) 条件句应用Where而非When; 2) 法律义务应用shall",
    "suggestion": "Where..., ...shall..."
  }}
]
```

### 4️⃣ 被动语态适当性（Passive Voice Appropriateness）
**检查内容：**
- ✅ 法律规定是否适当使用被动语态？
  - ✅ "应当给予处罚" → "shall be punished"（被动，合适）
  - ❌ "法院应当审理" → "shall be tried"（被动，但法院是主动施事者，不合适）
- ✅ 被动语态是否符合**目标语言法律表达习惯**？
- ✅ 施事者是否需要明确？

**具体输出：**
```json
"passive_issues": [
  {{
    "source": "公司应当公布财报",
    "target": "Financial reports shall be published",
    "problem": "缺少施事者，应改为'The company shall publish'",
    "active_preferred": true
  }}
]
```

## 📊 评分标准

**各维度评分：**
- **1.0**: 完美无误
- **0.9-0.95**: 极轻微问题（如同义词替换但含义准确）
- **0.85-0.90**: 有1-2个小问题，不影响理解
- **0.75-0.85**: 有明确问题，需要改进
- **< 0.75**: 有严重错误，必须修改

**整体评分 (overall_score)**:
- 加权平均：(情态×0.35 + 连接词×0.25 + 条件句×0.25 + 被动语态×0.15)

## 📤 返回格式

```json
{{
  "modality_preservation": 0.85,
  "connective_consistency": 0.90,
  "conditional_logic": 0.88,
  "passive_voice_appropriateness": 0.92,
  "overall_score": 0.88,
  
  "modality_issues": [...],  // 具体问题列表
  "connective_issues": [...],
  "conditional_issues": [...],
  "passive_issues": [...],
  
  "recommendations": [  // 总体改进建议
    "建议将should改为shall以符合法律文本规范",
    "建议优化条件句引导词，使用Where代替When"
  ],
  
  "issues": [  // 问题总结
    "情态动词: 2处should应改为shall",
    "条件句: 使用When不够正式，建议用Where"
  ]
}}
```"""
            },
            {
                "role": "user",
                "content": f"""请**具体检查**以下{source_lang}到{target_lang}的法律翻译的句法问题：

【源文本】
{source_text}

【译文】
{target_text}

【已识别的句法模式】（参考，但需独立判断）
{self._format_patterns(patterns)}

请按照4个维度**逐一检查**：
1. 情态动词是否准确？有无缺失？强度是否匹配？
2. 连接词逻辑是否正确？有无逻辑错误？
3. 条件句是否规范？引导词是否合适？
4. 被动语态是否适当？施事者是否清晰？

请输出具体的问题列表和评分。"""
            }
        ]
        
        try:
            result = await self.call_llm_json(messages)
            
            return SyntaxEvaluateResult(
                modality_preservation=result.get('modality_preservation', 0.0),
                connective_consistency=result.get('connective_consistency', 0.0),
                conditional_logic=result.get('conditional_logic', 0.0),
                passive_voice_appropriateness=result.get('passive_voice_appropriateness', 1.0),  # 新增，默认1.0
                overall_score=result.get('overall_score', 0.0),
                recommendations=result.get('recommendations', []),
                issues=result.get('issues', []),
                # 新增：具体问题列表
                modality_issues=result.get('modality_issues', []),
                connective_issues=result.get('connective_issues', []),
                conditional_issues=result.get('conditional_issues', []),
                passive_issues=result.get('passive_issues', [])
            )
        except Exception as e:
            logger.error(f"Syntax evaluate failed: {e}")
            import traceback
            traceback.print_exc()
            return SyntaxEvaluateResult(
                modality_preservation=0.0,
                connective_consistency=0.0,
                conditional_logic=0.0,
                passive_voice_appropriateness=0.0,
                overall_score=0.0,
                recommendations=[],
                issues=[]
            )
    
    def _format_patterns(self, patterns: List[Dict[str, Any]]) -> str:
        """格式化句法模式"""
        if not patterns:
            return "无句法模式"
        
        formatted = []
        for pattern in patterns:
            # 支持字典和对象两种格式
            if hasattr(pattern, 'source_pattern'):
                confidence = getattr(pattern, 'confidence', 0.0)
                formatted.append(f"- {pattern.source_pattern} → {pattern.target_pattern} ({pattern.modality_type}, 置信度: {confidence:.2f})")
            else:
                confidence = pattern.get('confidence', 0.0)
                formatted.append(f"- {pattern.get('source_pattern', '')} → {pattern.get('target_pattern', '')} ({pattern.get('modality_type', '')}, 置信度: {confidence:.2f})")
        
        return "\n".join(formatted)