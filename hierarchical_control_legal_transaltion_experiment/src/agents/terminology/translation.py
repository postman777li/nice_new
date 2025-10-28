"""
术语翻译Agent - 基于验证的术语表生成初始翻译
支持多候选生成和COMET-Kiwi质量评估选择
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

from ..base import BaseAgent, AgentConfig, AgentRunContext

logger = logging.getLogger(__name__)


@dataclass
class TranslationResult:
    source_text: str
    translated_text: str
    term_table: List[Dict[str, Any]]
    confidence: float
    candidates: Optional[List[str]] = None  # 多个候选翻译（如果生成了）


class TranslationAgent(BaseAgent):
    def __init__(self, locale: str = 'zh', generate_candidates: bool = False, num_candidates: int = 3):
        """
        初始化术语翻译Agent
        
        Args:
            locale: 语言地区
            generate_candidates: 是否生成多个候选翻译（默认False，只生成一个）
            num_candidates: 生成的候选数量（仅当generate_candidates=True时有效）
            generate_candidates: 是否生成多个候选
        """
        super().__init__(AgentConfig(
            name='terminology:translation',
            role='terminology_translator',
            domain='terminology',
            specialty='基于术语表的翻译',
            quality='review',
            locale=locale
        )) 
        
        self.generate_candidates = generate_candidates
        self.num_candidates = num_candidates
        
        if self.generate_candidates:
            logger.info(f"✓ 术语翻译Agent配置为生成{num_candidates}个候选翻译")

    async def execute(self, input_data: Dict[str, Any], ctx: Optional[AgentRunContext] = None) -> TranslationResult:
        """基于验证的术语表生成初始翻译"""
        source_text = input_data.get('source_text', '')
        term_table = input_data.get('term_table', [])
        source_lang = input_data.get('source_lang', 'zh')
        target_lang = input_data.get('target_lang', 'en')
        
        if not source_text:
            return TranslationResult(
                source_text="",
                translated_text="",
                term_table=[],
                confidence=0.0,
                candidates=None
            )
        
        # 根据配置生成单个翻译或多个候选
        if self.generate_candidates:
            candidates = await self._generate_candidates(
                source_text, term_table, source_lang, target_lang
            )
            if not candidates:
                # 如果候选生成失败，fallback到单次翻译
                logger.warning("候选生成失败，fallback到单次翻译")
                return await self._execute_single(
                    source_text, term_table, source_lang, target_lang
                )
            # 返回第一个候选作为translated_text，同时保留所有候选
            return TranslationResult(
                source_text=source_text,
                translated_text=candidates[0],
                term_table=term_table,
                confidence=0.8,  # 默认置信度，实际会由选择器Agent决定
                candidates=candidates
            )
        else:
            return await self._execute_single(
                source_text, term_table, source_lang, target_lang
            )
    
    async def _execute_single(
        self,
        source_text: str,
        term_table: List[Dict[str, Any]],
        source_lang: str,
        target_lang: str
    ) -> TranslationResult:
        """原有的单次翻译逻辑（向后兼容）"""
        messages = [
            {
                "role": "system",
                "content": f"""你是一个专业的法律翻译专家。请基于提供的术语表对法律文本进行翻译。

翻译要求：
1. 严格使用术语表中的翻译
2. 保持法律文本的准确性和专业性
3. 确保术语翻译的一致性
4. 保持原文的结构和逻辑

术语表：
{self._format_term_table(term_table)}

输出要求：请严格以json格式输出，仅输出一个json对象，不要包含任何解释性文字。
输出字段：
- translated_text: 翻译后的文本
- term_table: 使用到的术语表（数组，每项包含 source/target/confidence）
- confidence: 0~1 的置信度
"""
            },
            {
                "role": "user",
                "content": f"请将以下{source_lang}法律文本翻译为{target_lang}：\n\n{source_text}"
            }
        ]
        
        try:
            result = await self.call_llm_json(messages)
            
            return TranslationResult(
                source_text=source_text,
                translated_text=result.get('translated_text', ''),
                term_table=result.get('term_table', term_table),
                confidence=result.get('confidence', 0.8)
            )
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return TranslationResult(
                source_text=source_text,
                translated_text="",
                term_table=term_table,
                confidence=0.0
            )
    
    async def _generate_candidates(
        self,
        source_text: str,
        term_table: List[Dict[str, Any]],
        source_lang: str,
        target_lang: str
    ) -> List[str]:
        """生成多个候选翻译"""
        messages = [
            {
                "role": "system",
                "content": f"""你是一个专业的法律翻译专家。请基于提供的术语表对法律文本进行翻译。

翻译要求：
1. 严格使用术语表中的翻译
2. 保持法律文本的准确性和专业性
3. 确保术语翻译的一致性
4. 保持原文的结构和逻辑

术语表：
{self._format_term_table(term_table)}

**重要**：请生成 {self.num_candidates} 个不同的翻译候选，每个候选应该：
- 严格使用相同的术语表翻译（术语必须一致）
- 在非术语部分尝试不同的表达方式
- 尝试不同的句式结构和连接词
- 保持相同的语义但表达略有差异

输出要求：请严格以json格式输出：
{{
    "candidates": [
        {{"translated_text": "候选1", "confidence": 0.9}},
        {{"translated_text": "候选2", "confidence": 0.85}},
        {{"translated_text": "候选3", "confidence": 0.88}}
    ]
}}
"""
            },
            {
                "role": "user",
                "content": f"请将以下{source_lang}法律文本翻译为{target_lang}（生成{self.num_candidates}个候选）：\n\n{source_text}"
            }
        ]
        
        try:
            # 使用稍高的temperature增加多样性
            result = await self.call_llm_json(messages, temperature=0.4)
            
            candidates_data = result.get('candidates', [])
            candidates = [c.get('translated_text', '') for c in candidates_data if c.get('translated_text', '').strip()]
            
            if len(candidates) >= self.num_candidates:
                logger.info(f"成功生成 {len(candidates)} 个翻译候选")
                return candidates[:self.num_candidates]
            elif candidates:
                logger.warning(f"只生成了 {len(candidates)}/{self.num_candidates} 个候选，补充生成")
                # 补充生成
                additional = await self._generate_candidates_by_multiple_calls(
                    source_text, term_table, source_lang, target_lang,
                    num_needed=self.num_candidates - len(candidates)
                )
                candidates.extend(additional)
                return candidates[:self.num_candidates]
            else:
                logger.warning("LLM未返回有效候选，降级为多次调用")
                return await self._generate_candidates_by_multiple_calls(
                    source_text, term_table, source_lang, target_lang,
                    num_needed=self.num_candidates
                )
                
        except Exception as e:
            logger.error(f"生成候选失败: {e}，降级为多次调用")
            return await self._generate_candidates_by_multiple_calls(
                source_text, term_table, source_lang, target_lang,
                num_needed=self.num_candidates
            )
    
    async def _generate_candidates_by_multiple_calls(
        self,
        source_text: str,
        term_table: List[Dict[str, Any]],
        source_lang: str,
        target_lang: str,
        num_needed: int = None
    ) -> List[str]:
        """通过多次调用LLM生成不同候选（备用方案）"""
        if num_needed is None:
            num_needed = self.num_candidates
            
        candidates = []
        # 使用不同temperature增加多样性
        temperatures = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        for i, temp in enumerate(temperatures[:num_needed]):
            messages = [
                {
                    "role": "system",
                    "content": f"""你是一个专业的法律翻译专家。请基于提供的术语表对法律文本进行翻译。

术语表：
{self._format_term_table(term_table)}

输出JSON格式：{{"translated_text": "翻译内容", "confidence": 0.9}}
"""
                },
                {
                    "role": "user",
                    "content": f"请将以下{source_lang}法律文本翻译为{target_lang}：\n\n{source_text}"
                }
            ]
            
            try:
                result = await self.call_llm_json(messages, temperature=temp)
                translated = result.get('translated_text', '').strip()
                if translated:
                    candidates.append(translated)
                    logger.debug(f"温度{temp}生成候选{i+1}: 成功")
            except Exception as e:
                logger.warning(f"温度{temp}生成候选失败: {e}")
        
        if not candidates:
            # 最后的保底：使用默认参数生成一个
            try:
                result = await self._execute_single(source_text, term_table, source_lang, target_lang)
                if result.translated_text:
                    candidates.append(result.translated_text)
            except:
                pass
        
        logger.info(f"通过多次调用生成了 {len(candidates)} 个候选")
        return candidates
    
    def _format_term_table(self, term_table: List[Dict[str, Any]]) -> str:
        """格式化术语表"""
        if not term_table:
            return "无术语表"
        
        formatted = []
        for item in term_table:
            formatted.append(f"- {item.get('source', '')} -> {item.get('target', '')} (置信度: {item.get('confidence', 0.0)})")
        
        return "\n".join(formatted)
