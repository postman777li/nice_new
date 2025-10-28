"""
Agent基础类 - Python版本
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import asyncio
import json
import os
import logging
from contextlib import suppress

with suppress(ImportError):
    # 优先相对导入（包内调用）
    from ..lib.llm_client import get_llm_client  # type: ignore
with suppress(ImportError):
    # 退回绝对导入（在 backend 目录下运行）
    from lib.llm_client import get_llm_client  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class AgentRunContext:
    """Agent运行上下文"""
    project_id: Optional[str] = None
    user_id: Optional[str] = None
    locale: Optional[str] = None


@dataclass
class AgentConfig:
    """Agent配置"""
    name: str
    role: str = 'professional_assistant'
    domain: str = 'general'
    source_language: str = 'auto'
    target_language: str = 'zh'
    specialty: str = '通用处理'
    quality: str = 'review'  # 'draft' | 'review' | 'final'
    formality: str = 'neutral'  # 'informal' | 'neutral' | 'formal'
    locale: str = 'zh'


class BaseAgent(ABC):
    """Agent基类"""
    
    def __init__(self, config: Union[AgentConfig, str]):
        if isinstance(config, str):
            # 向后兼容
            self.config = AgentConfig(
                name=config,
                role='professional_assistant',
                domain='general',
                source_language='auto',
                target_language='zh',
                specialty='通用处理',
                quality='review',
                formality='neutral',
                locale='zh'
            )
        else:
            self.config = config
        
        self.name = self.config.name
        self.role = self.config.role
        self.domain = self.config.domain
        self.source_language = self.config.source_language
        self.target_language = self.config.target_language
        self.specialty = self.config.specialty
        self.quality = self.config.quality
        self.formality = self.config.formality
        self.locale = self.config.locale
        # 允许在 Agent 层覆盖模型：优先读取环境变量 OPENAI_API_MODEL
        self.model: Optional[str] = os.getenv("OPENAI_API_MODEL") or None
    
    @abstractmethod
    async def execute(self, input_data: Any, ctx: Optional[AgentRunContext] = None) -> Any:
        """执行Agent逻辑"""
        pass
    
    async def run(self, input_data: Any, ctx: Optional[AgentRunContext] = None) -> Any:
        """运行Agent"""
        return await self.execute(input_data, ctx)
    
    # LLM调用方法（子类使用）
    async def call_llm_json(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None, temperature: float = 0.2) -> Any:
        """调用LLM获取JSON响应（支持超时和重试）"""
        logger.info(f"Calling LLM(JSON) with {len(messages)} messages | agent={self.name}")
        try:
            client = get_llm_client()  # type: ignore
            result = await client.chat(
                messages=messages,
                model=self.model,
                temperature=temperature,
                response_format="json_object",
                max_tokens=max_tokens,
            )
            content = result.get("content", "")
            if not content:
                logger.warning("LLM返回空内容")
                return {"error": "empty_response"}
            
            try:
                parsed_json = json.loads(content)
                logger.info(f"LLM JSON解析成功，返回 {len(parsed_json) if isinstance(parsed_json, dict) else 'unknown'} 个字段")
                return parsed_json
            except json.JSONDecodeError as e:
                logger.warning(f"LLM返回内容不是有效JSON: {e}")
                logger.debug(f"原始内容: {content[:200]}...")
                # 返回原始文本，交由上层决定
                return {"raw": content}
                
        except Exception as error:
            logger.error(f"LLM(JSON) call failed: {error}")
            return {"error": str(error)}
    
    async def call_llm_text(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None, temperature: float = 0.2) -> str:
        """调用LLM获取文本响应（支持超时和重试）"""
        logger.info(f"Calling LLM(Text) with {len(messages)} messages | agent={self.name}")
        try:
            client = get_llm_client()  # type: ignore
            result = await client.chat(
                messages=messages,
                model=self.model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = result.get("content", "") or ""
            logger.info(f"LLM文本响应成功，长度: {len(content)} 字符")
            return content
        except Exception as error:
            logger.error(f"LLM(Text) call failed: {error}")
            return ""
    
    # 提示词构建方法
    async def build_prompt(self, output_format: str = 'text', constraints: Optional[List[str]] = None) -> str:
        """构建系统提示词"""
        # 获取国际化文本
        i18n = await self._get_i18n()
        
        # 构建基础提示词
        prompt = i18n.get_system_prompt('base', {'role': self.role})
        
        # 添加领域信息
        if self.domain != 'general':
            prompt += i18n.get_system_prompt('domain', {'domain': self.domain})
        
        # 添加语言对信息
        if self.source_language != 'auto' or self.target_language != 'zh':
            prompt += i18n.get_system_prompt('language_pair', {
                'source_language': self.source_language,
                'target_language': self.target_language
            })
        
        # 添加质量等级
        prompt += i18n.get_system_prompt('quality_requirement', {'quality': self.quality})
        
        # 添加正式度
        prompt += i18n.get_system_prompt('style', {'formality': self.formality})
        
        # 添加输出格式
        if output_format == 'json':
            prompt += i18n.get_system_prompt('json_output')
        else:
            prompt += i18n.get_system_prompt('text_output')
        
        # 添加约束条件
        if constraints:
            prompt += f"\n{i18n.get_system_prompt('requirements')}\n"
            for i, constraint in enumerate(constraints, 1):
                prompt += f"{i}) {constraint}\n"
        
        return prompt
    
    async def build_user_preference(self, preference: Optional[str] = None) -> str:
        """构建用户偏好"""
        if not preference or not preference.strip():
            return ''
        
        i18n = await self._get_i18n()
        return i18n.get_user_prompt('preference', {'preference': preference.strip()})
    
    async def build_glossary(self, glossary: Optional[List[Dict[str, str]]] = None, max_items: int = 200) -> str:
        """构建术语表"""
        if not glossary:
            return ''
        
        glossary_list = glossary[:max_items]
        if not glossary_list:
            return ''
        
        i18n = await self._get_i18n()
        entries = '\n'.join([f"{item['term']} => {item['translation']}" for item in glossary_list])
        return i18n.get_user_prompt('glossary', {'entries': entries})
    
    def build_messages(self, system_content: str, user_content: str) -> List[Dict[str, str]]:
        """构建消息列表"""
        return [
            {'role': 'system', 'content': system_content},
            {'role': 'user', 'content': user_content}
        ]
    
    async def _get_i18n(self):
        """获取国际化对象"""
        # TODO: 实现国际化
        return MockI18n()


class MockI18n:
    """模拟国际化对象"""
    
    def get_system_prompt(self, key: str, params: Dict[str, str] = None) -> str:
        """获取系统提示词"""
        prompts = {
            'base': f"你是一个专业的{params.get('role', '助手')}。",
            'domain': f"你的专业领域是{params.get('domain', '通用')}。",
            'language_pair': f"你需要处理{params.get('source_language', '源语言')}到{params.get('target_language', '目标语言')}的翻译。",
            'quality_requirement': f"请确保输出质量达到{params.get('quality', '标准')}级别。",
            'style': f"请使用{params.get('formality', '正式')}的语调。",
            'json_output': "请以JSON格式输出结果。",
            'text_output': "请以文本格式输出结果。",
            'requirements': "请遵循以下要求："
        }
        return prompts.get(key, "")
    
    def get_user_prompt(self, key: str, params: Dict[str, str] = None) -> str:
        """获取用户提示词"""
        prompts = {
            'preference': f"用户偏好：{params.get('preference', '')}",
            'glossary': f"术语表：\n{params.get('entries', '')}"
        }
        return prompts.get(key, "")
