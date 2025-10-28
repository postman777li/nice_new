"""
术语评估Agent - 评估和验证检索到的翻译
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

from ..base import BaseAgent, AgentConfig, AgentRunContext

logger = logging.getLogger(__name__)


@dataclass
class EvaluateResult:
    term: str
    translation: str
    is_valid: bool
    confidence: float
    reason: str
    suggestions: List[str]


class EvaluateAgent(BaseAgent):
    def __init__(self, locale: str = 'zh'):
        super().__init__(AgentConfig(
            name='terminology:evaluate',
            role='terminology_evaluator',
            domain='terminology',
            specialty='术语翻译评估',
            quality='review',
            locale=locale
        ))

    async def execute(self, input_data: Dict[str, Any], ctx: Optional[AgentRunContext] = None) -> List[EvaluateResult]:
        """评估和验证检索到的翻译"""
        translations = input_data.get('translations', [])
        source_text = input_data.get('source_text', '')
        source_lang = input_data.get('source_lang', 'zh')
        target_lang = input_data.get('target_lang', 'en')
        
        if not translations:
            return []
        
        # 使用LLM评估翻译质量（平衡模式）
        messages = [
            {
                "role": "system",
                "content": f"""你是一个专业的法律术语翻译评估专家。你的职责是客观评估术语翻译的质量，平衡准确性和实用性。

评估原则（平衡模式）：
1. **准确性优先**：确保术语翻译基本正确
2. **实用性兼顾**：基本合适的术语应该保留，不要过于苛刻
3. **语境适配**：考虑术语在当前语境下的适用性
4. **专业标准**：保持法律翻译的专业性

评估标准：

1. 法律准确性（Legal Accuracy）- 核心标准：
   ✓ 通过：法律概念基本准确
   ✓ 通过：专业性合格的翻译
   ⚠️ 谨慎：可能有轻微歧义但可接受
   ❌ 拒绝：法律概念明显错误

2. 术语一致性（Terminology Consistency）：
   ✓ 通过：符合常见法律术语标准
   ✓ 通过：业界认可的译法
   ⚠️ 谨慎：非官方但可接受的译法
   ❌ 拒绝：明显不符合规范

3. 语境适用性（Context Relevance）：
   ✓ 通过：适合当前语境
   ✓ 通过：虽然通用但用法正确
   ⚠️ 谨慎：基本适用但不够精确
   ❌ 拒绝：明显不适合当前语境

4. 专业规范性（Professional Standards）：
   ✓ 通过：符合法律专业标准
   ✓ 通过：正式且规范
   ⚠️ 谨慎：稍欠正式但可接受
   ❌ 拒绝：明显口语化或非正式

判定策略（平衡精确性和召回率）：
- 准确且专业 → is_valid=true, confidence=0.8-1.0
- 基本正确且可用 → is_valid=true, confidence=0.6-0.8 （接受）
- 有小瑕疵但可接受 → is_valid=true, confidence=0.5-0.6 （接受）
- 有明显问题 → is_valid=false, confidence=0.3-0.5 （拒绝）
- 严重错误 → is_valid=false, confidence=0.0-0.3 （拒绝）

评估理由示例：
通过：
- "术语翻译准确，适合法律语境"
- "符合法律翻译标准，建议使用"
- "虽然不够精确，但在当前语境下可接受"

拒绝：
- "法律概念有明显偏差，不建议使用"
- "严重不适合当前法律语境"
- "翻译存在重大错误"

输出要求：
- 客观评估每个术语
- 基本合适的术语应标记为 is_valid=true
- 只拒绝明显有问题的术语
- 给出清晰的评估理由

返回JSON格式：
{{
    "evaluations": [
        {{
            "term": "源术语",
            "translation": "目标翻译",
            "is_valid": true,  # 基本合适就应该通过
            "confidence": 0.75,  # 合理评分
            "reason": "术语翻译准确，适合当前语境",
            "suggestions": []
        }}
    ]
}}

**记住：要平衡准确性和召回率。基本合适的术语应该保留，让翻译Agent有更多可用的术语资源。只拒绝明显有问题的术语。**"""
            },
            {
                "role": "user",
                "content": f"""请客观评估以下{source_lang}到{target_lang}的法律术语翻译质量：

当前句子（需要翻译的句子）：
{source_text}

待评估的术语翻译（从术语库检索）：
""" + "\n".join([
    f"- {t.get('source', t.get('term', ''))} -> {t.get('target', t.get('translation', ''))}\n" +
    (f"  原始上下文: {t.get('context', '无')}\n" if t.get('context') else "  原始上下文: 无\n") +
    f"  上下文匹配度: {'高（适用）' if t.get('context') and len(t.get('context', '')) > 10 else '未知（通用术语）'}"
    for t in translations
]) + f"""

评估要求（核心：基于上下文相似度打分）：

1. **上下文相似度评分（最重要）**：
   - 仔细比较"原始上下文"和"当前句子（需要翻译的句子）"
   - 判断术语在两个上下文中的使用场景是否相似
   
   评分标准：
   - **0.85-1.0**：上下文高度相似，术语使用场景一致
     * 示例：原始"劳动者享有...权利" vs 当前"劳动者享有平等就业的权利" → 0.90
   
   - **0.70-0.84**：上下文相关，但有一定差异
     * 示例：原始"劳动者的权益" vs 当前"劳动者享有...权利" → 0.75
   
   - **0.60-0.69**：上下文略有关联，或为通用术语
     * 示例：原始"招用劳动者" vs 当前"劳动者享有...权利" → 0.65
   
   - **0.50-0.59**：上下文差异较大，但术语本身可能适用
     * 示例：原始"解除劳动合同" vs 当前"订立劳动合同" → 0.55
   
   - **<0.50**：上下文不匹配，或术语翻译有误 → is_valid=false

2. **术语翻译准确性**：
   - 翻译本身是否正确（作为次要因素调整分数±0.05）

3. **法律专业性**：
   - 是否符合法律术语规范（作为次要因素调整分数±0.05）

**关键原则：**
- 置信度 = 上下文相似度（主要因素70%） + 翻译准确性（15%） + 专业规范性（15%）
- 上下文越相似，置信度越高
- 如果原始上下文为空，则基于术语的通用性和准确性评分（通常0.6-0.7）

**同一术语的多个翻译候选的评分逻辑：**
假设从数据库检索到：
- 劳动者 → worker/workers (原始上下文: "用人单位招用劳动者")
- 劳动者 → labourer/labourers (原始上下文: "劳动者享有...权利")  
- 劳动者 → laborer/laborers (原始上下文: "保护劳动者权益")

当前句子: "劳动者享有平等就业的权利"

评分示例：
1. worker/workers: 
   - 上下文相似度: 0.70 (原始"招用"vs 当前"享有权利"，相关但不同场景)
   - 翻译准确性: +0.05 (worker是标准译法)
   - 最终: 0.75

2. labourer/labourers:
   - 上下文相似度: 0.90 (原始"享有...权利"vs 当前"享有...权利"，高度相似！)
   - 翻译准确性: 0 (labourer也正确，但略显英式)
   - 最终: 0.90 ← 因为上下文最相似，所以置信度最高

3. laborer/laborers:
   - 上下文相似度: 0.75 (原始"保护权益"vs 当前"享有权利"，都是权利相关)
   - 翻译准确性: -0.05 (laborer是美式拼写，不太标准)
   - 最终: 0.70

**核心：即使是同一个源术语，不同候选因为原始上下文不同，会得到不同的置信度评分！**

**请严格按照上下文相似度来给出置信度评分！**"""
            }
        ]
        
        try:
            result = await self.call_llm_json(messages)
            evaluations_data = result.get('evaluations', [])
            
            return [
                EvaluateResult(
                    term=item.get('term', ''),
                    translation=item.get('translation', ''),
                    is_valid=item.get('is_valid', False),
                    confidence=item.get('confidence', 0.0),
                    reason=item.get('reason', ''),
                    suggestions=item.get('suggestions', [])
                )
                for item in evaluations_data
            ]
        except Exception as e:
            logger.error(f"Evaluate failed: {e}")
            return []
