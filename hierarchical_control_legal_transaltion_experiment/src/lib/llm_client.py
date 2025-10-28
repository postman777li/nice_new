"""
OpenAI LLM 客户端封装 - 使用 AsyncOpenAI 支持高并发

依赖环境变量（由应用层在启动时加载）：
- OPENAI_API_KEY: API 密钥
- OPENAI_BASE_URL: 可选，自定义Base URL（如火山引擎兼容端点）
- OPENAI_API_MODEL: 默认模型名（未在调用中显式指定model时使用）
- LLM_TIMEOUT: 超时时间（秒），默认300
- LLM_MAX_RETRIES: 最大重试次数，默认3
- LLM_MAX_CONCURRENT: 最大并发请求数，默认10

可用方法：
- async chat(messages, model, temperature, response_format)
- async translate(text, src_lang, tgt_lang, model, style, temperature)
"""
import os
import logging
import asyncio
from typing import List, Dict, Optional

try:
    # openai v1.x SDK - 异步版本
    from openai import AsyncOpenAI
    _OPENAI_AVAILABLE = True
except Exception as _e:  # pragma: no cover
    _OPENAI_AVAILABLE = False


logger = logging.getLogger(__name__)


class OpenAILLM:
    """OpenAI LLM 异步封装 - 支持高并发"""

    def __init__(self, api_key: Optional[str] = None):
        if not _OPENAI_AVAILABLE:
            raise RuntimeError("openai 包未安装，请在 requirements.txt 中添加 openai 依赖")

        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY 未设置！\n"
                "请设置环境变量：export OPENAI_API_KEY='your-api-key-here'\n"
                "或在 .env 文件中配置：OPENAI_API_KEY=your-api-key-here"
            )

        # 支持自定义 Base URL 与默认模型
        self.base_url: Optional[str] = os.getenv("OPENAI_BASE_URL") or None
        self.model: str = os.getenv("OPENAI_API_MODEL", "gpt-4o-mini")
        
        # 超时和重试配置
        self.timeout: int = int(os.getenv("LLM_TIMEOUT", "300"))
        self.max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "3"))
        self.retry_delay: float = float(os.getenv("LLM_RETRY_DELAY", "1.0"))
        
        # 并发控制
        max_concurrent = int(os.getenv("LLM_MAX_CONCURRENT", "10"))
        self.semaphore = asyncio.Semaphore(max_concurrent)

        # 创建异步客户端
        if self.base_url:
            self.client = AsyncOpenAI(
                api_key=self.api_key, 
                base_url=self.base_url,
                timeout=float(self.timeout)
            )
        else:
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                timeout=float(self.timeout)
            )

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.2,
        response_format: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict:
        """异步对话接口。messages 形如 [{'role':'system','content':'...'}, {'role':'user','content':'...'}]"""
        params: Dict = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if response_format == "json_object":
            params["response_format"] = {"type": "json_object"}

        # 使用信号量控制并发
        async with self.semaphore:
            # 重试机制
            last_error = None
            for attempt in range(self.max_retries):
                try:
                    logger.info(f"LLM请求尝试 {attempt + 1}/{self.max_retries}")
                    
                    # 使用 asyncio.wait_for 实现超时
                    result = await asyncio.wait_for(
                        self._make_request(params),
                        timeout=self.timeout
                    )
                    
                    choice = result.choices[0]
                    content = choice.message.content or ""
                    return {
                        "id": result.id,
                        "model": result.model,
                        "content": content,
                        "finish_reason": choice.finish_reason,
                        "usage": getattr(result, "usage", None).__dict__ if getattr(result, "usage", None) else None,
                    }
                    
                except asyncio.TimeoutError:
                    last_error = f"请求超时 (尝试 {attempt + 1}/{self.max_retries})"
                    logger.warning(last_error)
                    if attempt < self.max_retries - 1:
                        delay = self.retry_delay * (2 ** attempt)  # 指数退避
                        logger.info(f"等待 {delay} 秒后重试...")
                        await asyncio.sleep(delay)
                    else:
                        raise Exception(last_error)
                        
                except Exception as e:
                    # 处理其他错误（包括API错误、网络错误等）
                    last_error = f"请求失败 (尝试 {attempt + 1}/{self.max_retries}): {str(e)}"
                    logger.warning(last_error)
                    
                    # 对于429（限流）和5xx错误，进行重试
                    should_retry = False
                    if hasattr(e, 'status_code'):
                        if e.status_code in [429, 500, 502, 503, 504]:
                            should_retry = True
                    elif "rate limit" in str(e).lower() or "timeout" in str(e).lower():
                        should_retry = True
                    
                    if should_retry and attempt < self.max_retries - 1:
                        delay = self.retry_delay * (2 ** attempt)  # 指数退避
                        logger.info(f"等待 {delay} 秒后重试...")
                        await asyncio.sleep(delay)
                    else:
                        if attempt >= self.max_retries - 1:
                            raise Exception(last_error)
                        else:
                            raise Exception(last_error)
            
            # 如果所有重试都失败
            raise Exception(f"所有重试都失败，最后错误: {last_error}")
    
    async def _make_request(self, params: Dict):
        """执行实际的API请求（异步）"""
        return await self.client.chat.completions.create(**params)

    async def translate(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str,
        model: Optional[str] = None,
        style: str = "neutral",
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> str:
        """简单翻译封装（异步）"""
        system_prompt = (
            f"You are a professional legal translator. Translate from {src_lang} to {tgt_lang}. "
            f"Keep {style} tone, preserve legal terminology and accuracy."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]
        # 使用显式传入的 model 或默认 self.model
        result = await self.chat(messages, model=(model or self.model), temperature=temperature, max_tokens=max_tokens)
        return result["content"]


# 便捷工厂
_global_client: Optional[OpenAILLM] = None


def get_llm_client() -> OpenAILLM:
    global _global_client
    if _global_client is None:
        _global_client = OpenAILLM()
    return _global_client


