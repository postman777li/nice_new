"""
Milvus 向量数据库模块
用于存储和检索文档、术语的向量表示
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import os  # <--- 添加这行
from dotenv import load_dotenv # <--- 添加这行


load_dotenv() # <--- 添加这行
logger = logging.getLogger(__name__)
try:
    from pymilvus import (
        connections, Collection, CollectionSchema, FieldSchema, DataType,
        utility, MilvusException
    )
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False
    logging.warning("PyMilvus not installed. Vector database features will be disabled.")




@dataclass
class VectorSearchResult:
    """向量搜索结果"""
    id: str
    score: float
    metadata: Dict[str, Any]


class MilvusVectorDB:
    """Milvus 向量数据库管理器"""
    
    def __init__(self, host: str = "localhost", port: str = "19530"):
        self.host = host
        self.port = port
        self.connected = False
        self.collections = {}
        
        if not MILVUS_AVAILABLE:
            logger.error("PyMilvus not available. Vector database will not work.")
            return
        
        self._connect()
    
    def _connect(self):
        """连接到 Milvus"""
        try:
            connections.connect("default", host=self.host, port=self.port)
            self.connected = True
            logger.info(f"Connected to Milvus at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            self.connected = False
    #改
    # def create_collection(self, collection_name: str, dimension: int = 768, description: str = ""):
    #     """创建集合"""
    #     if not self.connected:
    #         logger.error("Not connected to Milvus")
    #         return False
    def create_collection(self, collection_name: str, dimension: Optional[int] = None, description: str = ""):
        """创建集合"""
        if not self.connected:
            logger.error("Not connected to Milvus")
            return False

        # 如果调用者没有传入维度，我们自己从 .env 读取
        if dimension is None:
            try:
                dimension = int(os.getenv("EMBEDDING_DIM"))
                logger.info(f"create_collection: 未传入维度, 自动从 .env 读取: {dimension}")
            except (TypeError, ValueError, AttributeError):
                # 如果 .env 也读不到，才使用 2560 作为最后的保险
                logger.warning("create_collection: 未传入维度且 .env 读取失败, 回退到 2560 (豆包)")
                dimension = 2560
        try:
            # 检查集合是否已存在
            if utility.has_collection(collection_name):
                logger.info(f"Collection {collection_name} already exists")
                self.collections[collection_name] = Collection(collection_name)
                return True
            
            # 定义字段
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=255, is_primary=True),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="source_lang", dtype=DataType.VARCHAR, max_length=10),
                FieldSchema(name="target_lang", dtype=DataType.VARCHAR, max_length=10),
                FieldSchema(name="domain", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="created_at", dtype=DataType.INT64),
            ]
            
            # 创建集合模式
            schema = CollectionSchema(fields, description)
            
            # 创建集合
            collection = Collection(collection_name, schema)
            
            # 创建索引
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            collection.create_index("vector", index_params)
            
            self.collections[collection_name] = collection
            logger.info(f"Created collection {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create collection {collection_name}: {e}")
            return False
    
    def insert_vectors(self, collection_name: str, data: List[Dict[str, Any]]) -> bool:
        """插入向量数据"""
        if not self.connected or collection_name not in self.collections:
            logger.error(f"Collection {collection_name} not available")
            return False
        
        try:
            collection = self.collections[collection_name]
            
            # 准备数据
            entities = [
                [item["id"] for item in data],
                [item["vector"] for item in data],
                [item["text"] for item in data],
                [item.get("source_lang", "") for item in data],
                [item.get("target_lang", "") for item in data],
                [item.get("domain", "") for item in data],
                [item.get("created_at", 0) for item in data],
            ]
            
            # 插入数据
            mr = collection.insert(entities)
            collection.flush()
            
            logger.info(f"Inserted {len(data)} vectors into {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert vectors: {e}")
            return False
    
    def search_vectors(self, collection_name: str, query_vector: List[float], 
                      top_k: int = 10, filters: Optional[str] = None) -> List[VectorSearchResult]:
        """搜索相似向量"""
        if not self.connected or collection_name not in self.collections:
            logger.error(f"Collection {collection_name} not available")
            return []
        
        try:
            collection = self.collections[collection_name]
            collection.load()
            
            # 搜索参数
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
            
            # 执行搜索
            results = collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                expr=filters,
                output_fields=["text", "source_lang", "target_lang", "domain", "created_at"]
            )
            
            # 解析结果
            search_results = []
            for hits in results:
                for hit in hits:
                    result = VectorSearchResult(
                        id=hit.id,
                        score=hit.score,
                        metadata={
                            "text": hit.entity.get("text"),
                            "source_lang": hit.entity.get("source_lang"),
                            "target_lang": hit.entity.get("target_lang"),
                            "domain": hit.entity.get("domain"),
                            "created_at": hit.entity.get("created_at"),
                        }
                    )
                    search_results.append(result)
            
            return search_results
            
        except Exception as e:
            logger.error(f"Failed to search vectors: {e}")
            return []
    
    def delete_vectors(self, collection_name: str, ids: List[str]) -> bool:
        """删除向量"""
        if not self.connected or collection_name not in self.collections:
            logger.error(f"Collection {collection_name} not available")
            return False
        
        try:
            collection = self.collections[collection_name]
            expr = f"id in {ids}"
            collection.delete(expr)
            collection.flush()
            
            logger.info(f"Deleted {len(ids)} vectors from {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete vectors: {e}")
            return False
    
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """获取集合统计信息"""
        if not self.connected:
            return {}
        
        try:
            # 确保 collection 已加载
            if collection_name not in self.collections:
                from pymilvus import Collection
                self.collections[collection_name] = Collection(collection_name)
            
            collection = self.collections[collection_name]
            collection.load()  # 确保加载
            
            return {
                "name": collection_name,
                "num_entities": collection.num_entities,  # 使用 num_entities 属性
                "description": collection.description,
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}
    
    def close(self):
        """关闭连接"""
        if self.connected:
            connections.disconnect("default")
            self.connected = False
            logger.info("Disconnected from Milvus")


# 全局向量数据库实例
vector_db = MilvusVectorDB()


# def init_vector_collections():
#     """初始化向量集合"""
#     import os
#
#     # # 根据嵌入模型自动判断维度
#     # embed_model = os.getenv("OPENAI_EMBED_MODEL", "")
#     # if "doubao" in embed_model.lower():
#     #     dimension = 2560  # 豆包模型
#     # else:
#     #     dimension = 768  # OpenAI 默认
#     #改
#     try:
#         # 优先从 .env 读取 EMBEDDING_DIM
#         dimension = int(os.getenv("EMBEDDING_DIM"))
#         logger.info(f"初始化集合：从 .env 读取维度 (EMBEDDING_DIM): {dimension}")
#     except (TypeError, ValueError, AttributeError):
#         # 如果 .env 里没读到，使用默认值 1536
#         logger.warning("初始化集合：未在 .env 中找到 EMBEDDING_DIM，使用默认值 1536")
#         dimension = 1536
#
#
#
#     collections = [
#         ("legal_terms", "法律术语向量集合"),
#         ("legal_documents", "法律文档向量集合"),
#         ("translation_memory", "翻译记忆向量集合"),
#     ]
#
#     for name, desc in collections:
#         vector_db.create_collection(name, dimension=dimension, description=desc)


def search_similar_terms(query_vector: List[float], source_lang: str = "", 
                        target_lang: str = "", top_k: int = 5) -> List[VectorSearchResult]:
    """搜索相似术语"""
    filters = []
    if source_lang:
        filters.append(f'source_lang == "{source_lang}"')
    if target_lang:
        filters.append(f'target_lang == "{target_lang}"')
    
    filter_expr = " and ".join(filters) if filters else None
    
    return vector_db.search_vectors("legal_terms", query_vector, top_k, filter_expr)


def search_similar_documents(query_vector: List[float], domain: str = "", 
                           top_k: int = 5) -> List[VectorSearchResult]:
    """搜索相似文档"""
    filter_expr = f'domain == "{domain}"' if domain else None
    
    return vector_db.search_vectors("legal_documents", query_vector, top_k, filter_expr)


def search_translation_memory(query_vector: List[float], source_lang: str = "", 
                             target_lang: str = "", top_k: int = 5) -> List[VectorSearchResult]:
    """搜索翻译记忆"""
    filters = []
    if source_lang:
        filters.append(f'source_lang == "{source_lang}"')
    if target_lang:
        filters.append(f'target_lang == "{target_lang}"')
    
    filter_expr = " and ".join(filters) if filters else None
    
    return vector_db.search_vectors("translation_memory", query_vector, top_k, filter_expr)


# # 初始化集合
# if MILVUS_AVAILABLE:
#     init_vector_collections()
