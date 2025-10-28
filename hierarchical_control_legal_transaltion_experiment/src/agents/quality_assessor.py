"""
质量评估智能体
对比翻译结果和参考译文，给出质量评估和改进建议
"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from .base import BaseAgent, AgentConfig, AgentRunContext


@dataclass
class QualityAssessment:
    """质量评估结果"""
    overall_score: float
    accuracy_score: float
    fluency_score: float
    terminology_score: float
    style_score: float
    strengths: List[str]
    weaknesses: List[str]
    suggestions: List[str]
    detailed_comparison: Optional[str] = None


class QualityAssessorAgent(BaseAgent):
    """质量评估智能体：对比翻译和参考译文"""
    
    def __init__(self, locale: str = 'zh'):
        super().__init__(AgentConfig(
            name='quality:assessor',
            role='quality_assessor',
            domain='legal',
            specialty='翻译质量评估',
            quality='review',
            locale=locale
        ))
    
    async def execute(self, params: Dict[str, Any], context: Optional[AgentRunContext] = None) -> QualityAssessment:
        """
        执行质量评估
        
        Args:
            params: {
                'source_text': str - 源文本
                'translation': str - 翻译结果
                'reference': str - 参考译文
                'source_lang': str - 源语言
                'target_lang': str - 目标语言
            }
        
        Returns:
            QualityAssessment: 评估结果
        """
        source_text = params['source_text']
        translation = params['translation']
        reference = params['reference']
        source_lang = params.get('source_lang', 'zh')
        target_lang = params.get('target_lang', 'en')
        
        # 构建提示词
        prompt = self._build_prompt(source_text, translation, reference, source_lang, target_lang)
        
        # 调用LLM（使用BaseAgent提供的方法）
        result = await self.call_llm_json(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        # 验证和修复结果
        result = self._validate_result(result)
        
        return QualityAssessment(
            overall_score=result.get('overall_score', 0.0),
            accuracy_score=result.get('accuracy_score', 0.0),
            fluency_score=result.get('fluency_score', 0.0),
            terminology_score=result.get('terminology_score', 0.0),
            style_score=result.get('style_score', 0.0),
            strengths=result.get('strengths', []),
            weaknesses=result.get('weaknesses', []),
            suggestions=result.get('suggestions', []),
            detailed_comparison=result.get('detailed_comparison', '')
        )
    
    def _build_prompt(self, source: str, translation: str, reference: str, src_lang: str, tgt_lang: str) -> str:
        """构建评估提示词"""
        lang_names = {
            'zh': '中文',
            'en': '英文',
            'ja': '日文',
            'ko': '韩文'
        }
        src_name = lang_names.get(src_lang, src_lang)
        tgt_name = lang_names.get(tgt_lang, tgt_lang)
        
        return f"""你是一位专业的法律翻译质量评估专家。请对比以下翻译结果和参考译文，给出详细的质量评估和改进建议。

**源文本（{src_name}）：**
{source}

**翻译结果（待评估）：**
{translation}

**参考译文（标准答案）：**
{reference}

---

**评估要求：**

请从以下维度对翻译结果进行评分（0-1分制）：

1. **准确性 (Accuracy)**：翻译是否准确传达了源文本的意思，是否有误译、漏译、增译
2. **流畅性 (Fluency)**：译文是否符合目标语言的表达习惯，是否自然流畅
3. **术语一致性 (Terminology)**：法律术语翻译是否准确、规范、一致
4. **风格适配 (Style)**：是否符合法律文本的正式性和专业性要求

**输出格式（JSON）：**

```json
{{
    "overall_score": 0.85,  // 总体评分（0-1）
    "accuracy_score": 0.90,  // 准确性评分
    "fluency_score": 0.85,   // 流畅性评分
    "terminology_score": 0.80,  // 术语评分
    "style_score": 0.85,     // 风格评分
    
    "strengths": [  // 翻译的优点（2-4条）
        "法律术语'劳动者'翻译为'workers'准确恰当",
        "句子结构清晰，符合英文法律文本习惯"
    ],
    
    "weaknesses": [  // 翻译的不足（2-4条）
        "情态动词使用不当：'have'应改为'shall have'以体现法律强制性",
        "缺少'in accordance with law'等法律惯用表达"
    ],
    
    "suggestions": [  // 具体改进建议（3-5条）
        "将'have the right'改为'shall have the right'以增强法律效力",
        "考虑在句尾添加'in accordance with the law'使表达更完整",
        "保持与参考译文的术语一致性"
    ],
    
    "detailed_comparison": "翻译结果整体质量良好，准确传达了源文本的法律含义。主要优点是术语选择准确，句子结构清晰。需要改进的地方是：1) 情态动词的使用需要更符合法律文本的规范性要求；2) 部分法律惯用表达可以更加完善。参考译文使用了'shall'这一法律专用情态动词，更能体现法律条文的强制性和规范性。"
}}
```

**评分标准：**
- 0.90-1.00: 优秀，与参考译文质量相当或更好
- 0.80-0.89: 良好，有小的改进空间
- 0.70-0.79: 合格，存在一些明显问题
- 0.60-0.69: 需要改进，有较多问题
- <0.60: 不合格，存在严重问题

请给出专业、客观、具体的评估。重点关注法律翻译的专业性和准确性。
"""
    
    def _validate_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """验证和修复评估结果"""
        # 处理错误响应
        if 'error' in result:
            return {
                'overall_score': 0.0,
                'accuracy_score': 0.0,
                'fluency_score': 0.0,
                'terminology_score': 0.0,
                'style_score': 0.0,
                'strengths': [],
                'weaknesses': [f"评估失败: {result['error']}"],
                'suggestions': [],
                'detailed_comparison': result.get('raw', '')
            }
        
        # 验证必需字段并设置默认值
        if 'overall_score' not in result:
            result['overall_score'] = 0.75
        if 'accuracy_score' not in result:
            result['accuracy_score'] = 0.75
        if 'fluency_score' not in result:
            result['fluency_score'] = 0.75
        if 'terminology_score' not in result:
            result['terminology_score'] = 0.75
        if 'style_score' not in result:
            result['style_score'] = 0.75
        if 'strengths' not in result or not isinstance(result['strengths'], list):
            result['strengths'] = []
        if 'weaknesses' not in result or not isinstance(result['weaknesses'], list):
            result['weaknesses'] = []
        if 'suggestions' not in result or not isinstance(result['suggestions'], list):
            result['suggestions'] = []
        
        return result

