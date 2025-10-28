"""
术语归一化智能体 - 处理重复术语，正规化术语格式
"""
from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass

from ..base import BaseAgent, AgentConfig, AgentRunContext

logger = logging.getLogger(__name__)


@dataclass
class NormalizedTerm:
    """归一化后的术语"""
    source_term: str
    target_term: str
    normalized_source: str
    normalized_target: str
    confidence: float
    category: str
    source_context: str
    target_context: str
    quality_score: float
    is_valid: bool
    law: str
    domain: str
    year: str
    entry_id: str
    normalization_notes: str = ""


class TermNormalizationAgent(BaseAgent):
    """术语归一化智能体"""
    
    def __init__(self, locale: str = 'zh'):
        super().__init__(AgentConfig(
            name='preprocess:bilingual-term-normalization',
            role='bilingual_terminology_normalizer',
            domain='preprocess',
            specialty='双语术语归一化',
            quality='review',
            locale=locale
        ))

    async def execute(self, input_data: Dict[str, Any], ctx: Optional[AgentRunContext] = None) -> List[NormalizedTerm]:
        """执行术语归一化"""
        return await self.run(input_data, ctx)
    
    async def run(self, input_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> List[NormalizedTerm]:
        """运行术语归一化 - 使用LLM进行智能归一化（包含同术语变体合并）"""
        terms = input_data.get('terms', [])
        src_lang = input_data.get('src_lang', 'zh')
        tgt_lang = input_data.get('tgt_lang', 'en')
        batch_size = input_data.get('batch_size', 50)
        
        logger.info(f"开始术语归一化，输入术语数: {len(terms)}")
        
        if not terms:
            return []
        
        normalized_terms = []
        
        # 批量处理，使用LLM进行智能归一化（LLM会自动处理同术语的变体合并）
        for i in range(0, len(terms), batch_size):
            batch = terms[i:i + batch_size]
            logger.info(f"处理批次 {i//batch_size + 1}/{(len(terms)-1)//batch_size + 1}: {len(batch)} 个术语")
            
            # 使用LLM进行智能归一化
            batch_normalized = await self._normalize_batch_with_llm(batch, src_lang, tgt_lang)
            normalized_terms.extend(batch_normalized)
        
        logger.info(f"归一化完成，输出术语数: {len(normalized_terms)}")
        
        # ✅ 不再做规则去重，LLM已经在归一化时处理了同术语的变体合并
        return normalized_terms
    
    async def _normalize_batch_with_llm(self, batch: List[Dict[str, Any]], src_lang: str, tgt_lang: str) -> List[NormalizedTerm]:
        """使用LLM对批量术语进行智能归一化（根据语言调用不同的子函数）"""
        # 分别归一化源语言和目标语言术语
        source_terms = [term.get('source_term', '') for term in batch]
        target_terms = [term.get('target_term', '') for term in batch]
        
        # 根据语言调用相应的归一化函数
        normalized_sources = await self._normalize_terms_by_language(source_terms, src_lang)
        normalized_targets = await self._normalize_terms_by_language(target_terms, tgt_lang)
        
        # 验证并创建归一化术语对象
        normalized_terms = []
        for i, term in enumerate(batch):
            source_term = term.get('source_term', '')
            target_term = term.get('target_term', '')
            
            normalized_source = normalized_sources[i]
            normalized_target = normalized_targets[i]
            
            # 验证归一化结果的有效性
            if not self._is_valid_normalization(source_term, normalized_source, is_english=(src_lang == 'en')):
                logger.warning(f"⚠️ 无效归一化: '{source_term}' -> '{normalized_source}', 使用原术语")
                normalized_source = source_term
            
            if not self._is_valid_normalization(target_term, normalized_target, is_english=(tgt_lang == 'en')):
                logger.warning(f"⚠️ 无效归一化: '{target_term}' -> '{normalized_target}', 使用原术语")
                normalized_target = target_term
            
            normalized_terms.append(self._create_normalized_term(
                term,
                normalized_source,
                normalized_target,
                notes=""
            ))
        
        return normalized_terms
    
    async def _normalize_terms_by_language(self, terms: List[str], lang: str) -> List[str]:
        """根据语言类型调用相应的归一化函数"""
        if lang == 'zh':
            return await self._normalize_chinese(terms)
        elif lang == 'en':
            return await self._normalize_english(terms)
        elif lang == 'ja':
            return await self._normalize_japanese(terms)
        else:
            logger.warning(f"未知语言类型: {lang}，使用通用归一化")
            return await self._normalize_generic(terms, lang)
    
    async def _normalize_chinese(self, terms: List[str]) -> List[str]:
        """中文术语归一化（处理同术语的变体，不合并不同术语）"""
        if not terms:
            return []
        
        terms_text = "\n".join([f"{i+1}. {term}" for i, term in enumerate(terms)])
        
        messages = [
            {
                "role": "system",
                "content": """你是专业的中文法律术语归一化专家，专注于为法律术语词典提供准确、规范的术语处理。

**中文法律术语归一化规则**：

1. **繁简统一**：将所有术语统一为简体中文。
   - 例如："協議" → "协议"

2. **格式清理**：仅移除术语前后多余空格，保留内部空格和所有标点符号。
   - 例如：" 合 同 " → "合同"

3. **错别字校正**：根据权威法律文本校正常见错别字或异体字。
   - 例如："其它" → "其他"，"帐户" → "账户"

4. **全称简称统一**：对于有全称和简称的术语，统一使用全称。
   - 例如："有限公司" → "有限责任公司"

5. **🔥 结构性标记归一化 - 重要规则**：
   - **将法条编号、章节编号中的具体数字统一替换为XX**：
     - 例如："第36条" → "第XX条"
     - 例如："第三十六条" → "第XX条"
     - 例如："第40条第一项" → "第XX条第XX项"
     - 例如："第87条" → "第XX条"
     - 例如："第二章" → "第XX章"
     - 例如："第五节" → "第XX节"
     - 例如："（一）" → "（XX）"
     - 例如："（二）" → "（XX）"
   - 这样可以将所有相同类型的结构性标记统一归一化，便于去重和管理

6. **🔥 不要合并不同术语 - 关键规则**：
   - **不要合并不同的术语**，即使它们意思相近：
     - ❌ 错误：合并"工会"和"劳工组织" → 这是两个不同的术语！
     - ❌ 错误：合并"合同"和"协议" → 这是两个不同的术语！
     - ❌ 错误：合并"律师"和"法律顾问" → 这是两个不同的术语！

7. **禁止删词**：绝不删除"的"、"之"等助词，以免影响法律语义。

**重要 - 一一对应关系**：
- **每个输入术语必须有且仅有一个输出术语**
- 输出数量必须等于输入数量
- 独立处理每个术语
- 示例：如果输入有"有限公司"和"有限责任公司"，输出应该都是"有限责任公司"（去重在后续阶段处理）

返回JSON格式：
{
    "normalized": ["术语1", "术语2", ...]
}"""
            },
            {
                "role": "user",
                "content": f"""请归一化以下{len(terms)}个中文法律术语：

关键要求：
- 独立处理每个术语
- 输出数量必须等于输入数量（{len(terms)}个术语）
- 转换繁体为简体，统一简称为全称
- 不要合并不同的术语（如"工会"和"劳工组织"是不同术语）

{terms_text}

请严格按照JSON格式返回恰好{len(terms)}个归一化后的术语。"""
            }
        ]
        
        try:
            result = await self.call_llm_json(messages)
            return self._parse_normalized_result(result, terms)
        except Exception as e:
            logger.error(f"中文归一化失败: {e}，使用原术语")
            return terms
    
    async def _normalize_english(self, terms: List[str]) -> List[str]:
        """英文术语归一化（处理同术语的变体，不合并不同术语）"""
        if not terms:
            return []
        
        terms_text = "\n".join([f"{i+1}. {term}" for i, term in enumerate(terms)])
        
        messages = [
            {
                "role": "system",
                "content": """You are a professional legal terminology normalization expert, tasked with preparing terms for a legal dictionary.

**English Legal Terminology Normalization Rules**:

1.  **🔥 Plural/Singular Normalization - CRITICAL RULE**:
    *   **If the input term is in plural form**, normalize it to the singular form and output as "singular/plural" (using a slash to connect both forms):
        *   Example: `"contracts"` → `"contract/contracts"`
        *   Example: `"trade unions"` → `"trade union/trade unions"`
        *   Example: `"attorneys"` → `"attorney/attorneys"`
        *   Example: `"parties"` → `"party/parties"`
        *   Example: `"companies"` → `"company/companies"`
    *   **Handle compound terms** similarly:
        *   Example: `"employment contracts"` → `"employment contract/employment contracts"`
        *   Example: `"criminal defendants"` → `"criminal defendant/criminal defendants"`
    *   **For irregular plurals**, apply the same logic:
        *   Example: `"children"` → `"child/children"`
        *   Example: `"men"` → `"man/men"`
    *   **If the input term is in singular form**, output it as is (do not add plural form):
        *   Example: `"contract"` → `"contract"`
        *   Example: `"attorney"` → `"attorney"`
    *   **Exception**: For fixed legal terms or proper nouns that are inherently plural, keep them as is without adding singular form:
        *   Example: `"United States"` → `"United States"`
        *   Example: `"Securities and Exchange Commission"` → `"Securities and Exchange Commission"`
        *   Example: `"civil rights"` → `"civil rights"`

2.  **🔥 Verb Tense Normalization - CRITICAL RULE**:
    *   **Convert all verbs to their base form (infinitive without "to")**:
        *   Example: `"terminated"` → `"terminate"`
        *   Example: `"terminating"` → `"terminate"`
        *   Example: `"terminates"` → `"terminate"`
        *   Example: `"applied"` → `"apply"`
        *   Example: `"applying"` → `"apply"`
        *   Example: `"executed"` → `"execute"`
        *   Example: `"executing"` → `"execute"`
    *   **For verb phrases**, normalize the verb to base form:
        *   Example: `"being terminated"` → `"be terminated"` (or better: just `"terminate"` if it's a legal term)
        *   Example: `"has been executed"` → `"execute"`
    *   **Note**: If the term is a noun derived from a verb (gerund used as noun), keep the gerund form:
        *   Example: `"termination"` → `"termination"` (this is a noun, not a verb)
        *   Example: `"execution"` → `"execution"` (this is a noun, not a verb)

3.  **🔥 Structural Markers Normalization - IMPORTANT RULE**:
    *   **Replace specific numbers in article/chapter references with XX**:
        *   Example: `"Article 36"` → `"Article XX"`
        *   Example: `"Article 38"` → `"Article XX"`
        *   Example: `"Section 5"` → `"Section XX"`
        *   Example: `"Chapter 3"` → `"Chapter XX"`
        *   Example: `"Paragraph 2"` → `"Paragraph XX"`
        *   Example: `"Item (1)"` → `"Item (XX)"`
        *   Example: `"(a)"` → `"(XX)"`
    *   This unifies all structural markers for easier deduplication

4.  **Case Handling**:
    *   Convert purely generic legal terms to lowercase.
        *   Example: `"Contract"` → `"contract"`, `"Tort"` → `"tort"`
    *   **Preserve the capitalization** of proper nouns, established legal doctrines, and official names.
        *   Example: `"Supreme Court"`, `"Due Process Clause"`, `"Miranda rights"`

4.  **Whitespace & Format Cleaning**: Remove leading/trailing whitespace, normalize internal spacing (e.g., collapse multiple spaces to one).
    *   Example: `"  contract  "` → `"contract"`, `"employment   contract"` → `"employment contract"`

5.  **Spelling Variants**:
    *   Standardize to predominant American English form (unless specified otherwise):
        *   Example: `"judgement"` → `"judgment"`
        *   Example: `"colour"` → `"color"`

6.  **🔥 DO NOT Merge Different Terms - CRITICAL RULE**:
    *   **DO NOT merge different terms**, even if similar or synonymous. Treat each term independently:
        *   ❌ Wrong: Merge `"agreement"` and `"contract"` → Keep both as separate terms!
        *   ❌ Wrong: Merge `"lawyer"` and `"attorney"` → Keep both as separate terms!
        *   ❌ Wrong: Merge `"mediator"` and `"arbitrator"` → Keep both as separate terms!

**IMPORTANT - One-to-One Mapping**:
- **Each input term MUST have exactly one output term**
- Output count MUST equal input count
- Process each term independently (de-duplication happens later in the dictionary process)
- Example: If input has both "contract" and "contracts", output will be "contract" and "contract/contracts" respectively

Return JSON format:
{
    "normalized": ["term1", "term2", ...]
}"""
            },
            {
                "role": "user",
                "content": f"""Please normalize the following {len(terms)} English legal terms:

CRITICAL REQUIREMENTS:
- Process each term independently
- Output count MUST equal input count ({len(terms)} terms)
- Convert plurals to "singular/plural" format
- Convert verb tenses to base form (infinitive)
- Replace numbers in structural markers with XX (e.g., "Article 36" → "Article XX")
- DO NOT merge different terms

{terms_text}

Return strictly in JSON format with exactly {len(terms)} normalized terms."""
            }
        ]
        
        try:
            result = await self.call_llm_json(messages)
            return self._parse_normalized_result(result, terms)
        except Exception as e:
            logger.error(f"英文归一化失败: {e}，使用原术语")
            return terms
    
    async def _normalize_japanese(self, terms: List[str]) -> List[str]:
        """日文术语归一化"""
        if not terms:
            return []
        
        terms_text = "\n".join([f"{i+1}. {term}" for i, term in enumerate(terms)])
        
        messages = [
            {
                "role": "system",
                "content": """あなたは日本語法律用語の正規化専門家です。法律辞典構築を目的として、正確で規範的な用語処理を行います。

**日本語法律用語正規化ルール**：

1. **表記統一**：
   - ひらがな・カタカナ表記は、権威ある法律文献（六法全書、判例集等）に基づき標準漢字表記に統一します。
     - 例：「けいやく」 → 「契約」
   - **ただし**、法律上で確定したカタカナ語はそのまま保持します。
     - 例：「ノンコンテンション条項」は変更しない

2. **送り仮名統一**：
   - 内閣告示「送り仮名の付け方」及び法律分野の慣例に従って統一します。
     - 例：「うけとり」 → 「受領」、「うけわたし」 → 「受渡」

3. **誤字・異体字修正**：
   - 明らかな誤字や異体字を標準形に修正します。
     - 例：「其他」 → 「その他」、「弁償」 → 「弁償」（「賠償」と意味が異なるため注意）

4. **略語と正式名称**：
   - 略語は正式名称に統一しますが、両方が別個の用語として認識される場合は保持します。
     - 例：「民訴」 → 「民事訴訟法」
     - 例：「会社法」と「会社法施行規則」は別々の用語として保持

5. **🔥 構造的マーカーの正規化 - 重要ルール**：
   - **条文番号や章節番号の具体的な数字をXXに統一します**：
     - 例：「第36条」 → 「第XX条」
     - 例：「第三十六条」 → 「第XX条」
     - 例：「第40条第1項」 → 「第XX条第XX項」
     - 例：「第2章」 → 「第XX章」
     - 例：「第5節」 → 「第XX節」
     - 例：「（一）」 → 「（XX）」
     - 例：「（二）」 → 「（XX）」
   - これにより同種の構造的マーカーを統一し、重複排除が容易になります

6. **同義語処理 - 重要ルール**：
   - **原則として同義語は統一しません**。法律上、微妙な意味の違いがあるためです。
     - 例：「弁護士」と「弁護人」は統一しない（「弁護人」は刑事事件に特化）
     - 例：「契約」と「合意」は統一しない（法的効力が異なる）
   - 統一するのは、完全に同義で且つ一方が標準形と明確に判断できる場合のみです。

7. **削除禁止**：
   - 「の」、「こと」などの助詞や、冗長に見える表現も絶対に削除しません。
     - 例：「契約の解除」を「契約解除」に短縮しない

**重要**：正規化に疑義がある場合、特に同義語や表記の判断に迷う場合は、**必ず元の用語を保持してください**。法律用語の完全性と正確性が最優先です。

JSON形式で返答：
{
    "normalized": ["用語1", "用語2", ...]
}"""
            },
            {
                "role": "user",
                "content": f"""以下の{len(terms)}個の日本語法律用語を正規化してください：

{terms_text}

JSON形式で厳密に返答してください。数は入力と一致する必要があります。"""
            }
        ]
        
        try:
            result = await self.call_llm_json(messages)
            return self._parse_normalized_result(result, terms)
        except Exception as e:
            logger.error(f"日文归一化失败: {e}，使用原术语")
            return terms
    
    async def _normalize_generic(self, terms: List[str], lang: str) -> List[str]:
        """通用归一化（用于其他语言）"""
        if not terms:
            return []
        
        terms_text = "\n".join([f"{i+1}. {term}" for i, term in enumerate(terms)])
        
        messages = [
            {
                "role": "system",
                "content": f"""You are a legal terminology normalization expert for {lang}.

**Normalization Rules**:
1. Format cleaning: Remove extra spaces and standardize punctuation
2. Preserve meaning: Normalized term must be highly related to original (at least 80% overlap)
3. Standard form: Use the most standard expression

**Important**: If unable to normalize, return the original term!

Return JSON format:
{{
    "normalized": ["term1", "term2", ...]
}}"""
            },
            {
                "role": "user",
                "content": f"""Please normalize the following {len(terms)} legal terms:

{terms_text}

Return strictly in JSON format, the count must match the input."""
            }
        ]
        
        try:
            result = await self.call_llm_json(messages)
            return self._parse_normalized_result(result, terms)
        except Exception as e:
            logger.error(f"{lang}归一化失败: {e}，使用原术语")
            return terms
    
    def _parse_normalized_result(self, result: Dict[str, Any], original_terms: List[str]) -> List[str]:
        """解析LLM返回的归一化结果（要求一一对应）"""
        if 'error' in result:
            logger.warning(f"LLM返回错误: {result['error']}")
            return original_terms
        
        if 'raw' in result:
            import json as json_lib
            try:
                result = json_lib.loads(result['raw'])
            except json_lib.JSONDecodeError as e:
                logger.warning(f"JSON解析失败: {e}")
                return original_terms
        
        normalized_list = result.get('normalized', [])
        
        if not normalized_list:
            logger.warning(f"返回为空列表")
            return original_terms
        
        # ✅ 要求输入输出数量严格一致
        if len(normalized_list) != len(original_terms):
            logger.warning(f"⚠️ 返回数量不匹配: 期望{len(original_terms)}, 实际{len(normalized_list)}，使用原术语")
            return original_terms
        
        return normalized_list
    
    def _fallback_normalize_batch(self, batch: List[Dict[str, Any]], src_lang: str, tgt_lang: str) -> List[NormalizedTerm]:
        """后备方案：使用规则进行格式归一化"""
        normalized_terms = []
        for term in batch:
            source_term = term.get('source_term', '')
            target_term = term.get('target_term', '')
            
            normalized_source = self._normalize_term_format(source_term, is_english=False)
            normalized_target = self._normalize_term_format(target_term, is_english=(tgt_lang == 'en'))
            
            normalized_terms.append(self._create_normalized_term(
                term,
                normalized_source,
                normalized_target
            ))
        return normalized_terms
    
   
    # 去重由Stage4标准化阶段处理
    
    def _is_valid_normalization(self, original: str, normalized: str, is_english: bool = False) -> bool:
        """验证归一化结果是否有效。

        英文规则增强：
        - 接受复合形式 "singular/plural"（例如：institution/institutions）
        - 接受常见的单复数变化（例如：workers -> worker）
        - 放宽为基于包含、词/字符重叠的保守验证
        中文规则保持字符重叠≥30%。
        """
        if not original or not normalized:
            return False
        
        # 完全相同
        if original == normalized:
            return True
        
        # 🔥 特殊规则：结构性标记归一化（第XX条、第XX章等）
        if not is_english:
            import re
            # 检查是否为结构性标记的归一化（原文有数字，归一化后变成XX）
            structural_pattern = re.compile(r'^第[零〇○一二三四五六七八九十百千两\dXx]+(?:之[零〇○一二三四五六七八九十百千两\dXx]+)?条(?:第[零〇○一二三四五六七八九十百千两\dXx]+(?:项|款))?$')
            chapter_pattern = re.compile(r'^第[零〇○一二三四五六七八九十百千两\dXx]+(?:章|节|目)$')
            enum_pattern = re.compile(r'^[（(][零〇○一二三四五六七八九十\dXx]+[)）]$')
            
            # 如果归一化后是结构性标记且包含XX，则接受
            if 'XX' in normalized or 'xx' in normalized or 'Xx' in normalized:
                if structural_pattern.match(normalized) or chapter_pattern.match(normalized) or enum_pattern.match(normalized):
                    # 检查原文也是同类结构
                    if structural_pattern.match(original) or chapter_pattern.match(original) or enum_pattern.match(original):
                        return True
        
        if is_english:
            orig_lower = original.lower().strip()
            norm_lower = normalized.lower().strip()
            
            # 0) 特殊规则：结构性标记归一化（Article XX, Section XX等）
            import re
            structural_en_pattern = re.compile(r'^(article|section|chapter|paragraph|item|clause)\s+([xX]+|\d+)$', re.IGNORECASE)
            if 'xx' in norm_lower or 'XX' in normalized:
                match_norm = structural_en_pattern.match(norm_lower)
                match_orig = structural_en_pattern.match(orig_lower)
                if match_norm and match_orig:
                    # 检查类型是否相同（都是article/section等）
                    if match_norm.group(1).lower() == match_orig.group(1).lower():
                        return True

            # 1) 复合形式 singular/plural
            if "/" in norm_lower:
                parts = [p.strip() for p in norm_lower.split("/") if p.strip()]
                if self._matches_english_number_variants(orig_lower, parts):
                    return True

            # 2) 括号形式 contract(s)/company(ies) 等（尽量兼容）
            for marker in ["(s)", "(es)", "(ies)"]:
                if norm_lower.replace(marker, "") == orig_lower:
                    return True
                if orig_lower.replace(marker, "") == norm_lower:
                    return True

            # 3) 原词与归一化结果是单复数变体（短包含 + 长度差限制）
            if orig_lower in norm_lower or norm_lower in orig_lower:
                len_diff = abs(len(orig_lower) - len(norm_lower))
                if len_diff <= max(len(orig_lower), len(norm_lower)) * 0.5:
                    return True

            # 4) 直接比较常见的单复数变体（只处理短语最后一个词）
            orig_variants = self._generate_english_phrase_variants(orig_lower)
            norm_variants = self._generate_english_phrase_variants(norm_lower)
            if orig_lower in norm_variants or norm_lower in orig_variants:
                return True

            # 5) 词重叠（按空格与斜杠拆分），降低阈值到20%
            def _tokenize(text: str) -> set:
                text = text.replace("/", " ").replace("-", " ")
                return set(filter(None, text.split()))
            orig_words = _tokenize(orig_lower)
            norm_words = _tokenize(norm_lower)
            if orig_words and norm_words:
                overlap = len(orig_words & norm_words)
                min_words = min(len(orig_words), len(norm_words))
                if overlap >= max(1, int(min_words * 0.2)):
                    return True

            # 6) 字符重叠（去空格与连字符），阈值50%
            oc = set(orig_lower.replace(" ", "").replace("-", ""))
            nc = set(norm_lower.replace(" ", "").replace("-", ""))
            if oc and nc:
                overlap = len(oc & nc)
                min_chars = min(len(oc), len(nc))
                if overlap >= min_chars * 0.5:
                    return True

            return False
        else:
            # 中文：检查字符重叠（≥80%）
            orig_chars = set(original)
            norm_chars = set(normalized)
            overlap = len(orig_chars & norm_chars)
            min_chars = min(len(orig_chars), len(norm_chars))
            return overlap >= min_chars * 0.8

    def _matches_english_number_variants(self, original: str, parts: list) -> bool:
        """判断 original 是否与 parts 中任一项构成单/复数关系或相等。
        仅对短语最后一个词进行形态变化判断。
        """
        if not parts:
            return False
        original_variants = self._generate_english_phrase_variants(original)
        for p in parts:
            p = p.strip()
            if not p:
                continue
            if p in original_variants:
                return True
            # 也尝试对 part 生成变体做交叉匹配
            part_variants = self._generate_english_phrase_variants(p)
            if original in part_variants:
                return True
        return False

    def _generate_english_phrase_variants(self, phrase: str) -> set:
        """生成短语的常见单/复数变体集合（仅改变最后一个词）。"""
        phrase = phrase.strip()
        if not phrase:
            return {phrase}
        words = phrase.split()
        last = words[-1]
        singular = self._singularize_english_word(last)
        plural = self._pluralize_english_word(singular)
        base = " ".join(words[:-1])
        singular_phrase = (base + " " + singular).strip()
        plural_phrase = (base + " " + plural).strip()
        return {phrase, singular_phrase, plural_phrase}

    def _singularize_english_word(self, word: str) -> str:
        """非常轻量的英文词形还原（名词，启发式）。"""
        w = word.strip().lower()
        if not w:
            return w
        irregular = {
            "men": "man",
            "women": "woman",
            "children": "child",
            "people": "person",
            "teeth": "tooth",
            "feet": "foot",
            "geese": "goose",
            "mice": "mouse",
            "indices": "index",
            "appendices": "appendix",
            "matrices": "matrix",
            "vertices": "vertex",
            "data": "datum",
        }
        if w in irregular:
            return irregular[w]
        if w.endswith("ies") and len(w) > 3:
            return w[:-3] + "y"
        # 优先处理以 ch/sh 结尾的 es
        if w.endswith("ches") or w.endswith("shes"):
            return w[:-2]  # 去掉 es
        # 类似 classes/processes -> class/process
        if w.endswith("sses") or w.endswith("xes") or w.endswith("zes"):
            return w[:-2]
        # 常见情况：去掉词尾的单个 s（避免去掉 ss）
        if w.endswith("s") and not w.endswith("ss"):
            return w[:-1]
        return w

    def _pluralize_english_word(self, word: str) -> str:
        """非常轻量的英文名词复数生成（启发式）。"""
        w = word.strip().lower()
        if not w:
            return w
        irregular = {
            "man": "men",
            "woman": "women",
            "child": "children",
            "person": "people",
            "tooth": "teeth",
            "foot": "feet",
            "goose": "geese",
            "mouse": "mice",
            "index": "indices",
            "appendix": "appendices",
            "matrix": "matrices",
            "vertex": "vertices",
            "datum": "data",
        }
        if w in irregular:
            return irregular[w]
        if w.endswith("y") and len(w) > 1 and w[-2] not in "aeiou":
            return w[:-1] + "ies"
        if w.endswith("s") or w.endswith("x") or w.endswith("z") or w.endswith("ch") or w.endswith("sh"):
            return w + "es"
        return w + "s"
    
    def _normalize_term_format(self, term: str, is_english: bool = True) -> str:
        """对单个术语进行形式正规化（不改变语义，只正规化格式）"""
        if not term:
            return term
        
        if is_english:
            # 英文正规化规则（非常保守，只做基本清理）
            normalized = term.strip()
            
            # 1. 统一大小写（转为小写，除非是专有名词）
            # 如果整个词都是大写，转为小写
            if normalized.isupper():
                normalized = normalized.lower()
            # 如果是混合大小写，保持原样（可能是专有名词）
            
            # 2. 移除多余空格
            normalized = ' '.join(normalized.split())
            
            # 3. 正规化引号和特殊字符
            normalized = normalized.replace('"', '"').replace('"', '"')
            normalized = normalized.replace(''', "'").replace(''', "'")
            
            # 注意：不做单复数统一！
            # 原因：mediator, mediators, mediate, mediation 是不同的词
            # 单复数的"选择"应该在正规化阶段完成
            
            return normalized
        else:
            # 中文正规化规则
            normalized = term.strip()
            
            # 1. 移除多余空格
            normalized = ''.join(normalized.split())
            
            # 2. 繁简统一（如果需要）
            # TODO: 可以添加繁简转换
            
            # 3. 移除冗余的助词（保守处理）
            # TODO: 可以添加更多规则
            
            return normalized
    
    def _create_normalized_term(self, original_term: Dict[str, Any], normalized_source: str, normalized_target: str, notes: str = "") -> NormalizedTerm:
        """创建归一化术语对象"""
        return NormalizedTerm(
            source_term=original_term['source_term'],
            target_term=original_term['target_term'],
            normalized_source=normalized_source,
            normalized_target=normalized_target,
            confidence=original_term.get('confidence', 0.0),
            category=original_term.get('category', ''),
            source_context=original_term.get('source_context', ''),
            target_context=original_term.get('target_context', ''),
            quality_score=original_term.get('quality_score', 0.0),
            is_valid=original_term.get('is_valid', True),
            law=original_term.get('law', ''),
            domain=original_term.get('domain', ''),
            year=original_term.get('year', ''),
            entry_id=original_term.get('entry_id', ''),
            normalization_notes=notes
        )
