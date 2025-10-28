"""
嵌入向量生成模块
支持 OpenAI 和兼容 API（如豆包 Doubao）
"""
import os
import logging
from typing import List, Optional
from dotenv import load_dotenv # <--- 添加这行

load_dotenv() # <--- 添加这行

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI not installed. Embedding features will be disabled.")

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """嵌入向量生成客户端"""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 model: Optional[str] = None):
        if not OPENAI_AVAILABLE:
            raise RuntimeError("OpenAI package not installed")
        
        # 读取环境变量（优先使用传入的参数）
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        # 优先使用 OPENAI_EMBED_MODEL，回退到 OPENAI_API_MODEL
        # self.model = model or os.getenv("OPENAI_EMBED_MODEL") or os.getenv("OPENAI_API_MODEL", "text-embedding-3-small")
        self.model = model or os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")  # <--- 修改这行

        if not self.api_key:
            logger.warning("OPENAI_API_KEY not set, embedding calls will fail")
        
        # 创建客户端
        if self.base_url:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        else:
            self.client = OpenAI(api_key=self.api_key)
        
        logger.info(f"EmbeddingClient initialized with model: {self.model}")
    
    def get_embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        """获取单个文本的嵌入向量"""
        try:
            response = self.client.embeddings.create(
                input=text,
                model=model or self.model
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            raise
    
    def get_embeddings_batch(self, texts: List[str], model: Optional[str] = None, max_retries: int = 3) -> List[List[float]]:
        """批量获取嵌入向量"""
        import time
        
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    input=texts,
                    model=model or self.model
                )
                return [item.embedding for item in response.data]
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # 2, 4, 6 秒
                    logger.warning(f"Batch embedding failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to get batch embeddings after {max_retries} attempts: {e}")
                    raise


# 全局嵌入客户端实例
_default_embedding_client = None


def get_default_embedding_client() -> EmbeddingClient:
    """获取默认的嵌入客户端"""
    global _default_embedding_client
    if _default_embedding_client is None:
        _default_embedding_client = EmbeddingClient()
    return _default_embedding_client


def get_embedding(text: str) -> List[float]:
    """快捷函数：获取单个文本的嵌入向量"""
    client = get_default_embedding_client()
    return client.get_embedding(text)


def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """快捷函数：批量获取嵌入向量"""
    client = get_default_embedding_client()
    return client.get_embeddings_batch(texts)

