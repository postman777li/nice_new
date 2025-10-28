#!/usr/bin/env python3
"""
重置 Milvus collections（删除旧的，创建新的）
用于修复向量维度不匹配问题
"""
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from pymilvus import connections, utility
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False
    print("❌ PyMilvus not installed")
    sys.exit(1)


def reset_collections(host: str = "localhost", port: str = "19530"):
    """重置所有向量集合"""
    try:
        # 连接到 Milvus
        connections.connect("default", host=host, port=port)
        print(f"✓ 已连接到 Milvus ({host}:{port})")
        
        # 获取当前使用的嵌入模型维度
        embed_model = os.getenv("OPENAI_EMBED_MODEL", "")
        if "doubao" in embed_model.lower():
            dimension = 2560
            print(f"✓ 检测到豆包模型，使用 {dimension} 维向量")
        else:
            #改了
            # dimension = 768
            # print(f"✓ 使用默认 {dimension} 维向量")
            try:
                dimension = int(os.getenv("EMBEDDING_DIM", "1536"))  # 默认 1536
                print(f"✓ 从 .env 读取维度: {dimension}")
            except ValueError:
                dimension = 1536  # 如果转换失败，使用默认值
                print(f"⚠️  无法从 .env 读取维度，使用默认 {dimension}")
        
        # collections = ["legal_terms", "legal_documents", "translation_memory"]
        #改
        # tm_collection_name = os.getenv("TM_COLLECTION", "tm_zh_en")  # 默认 tm_zh_en
        # collections = [tm_collection_name]  # 只处理 TM 集合
        # print(f"✓ 从 .env 读取要重置的集合: {tm_collection_name}")
        # 获取要重置的集合名称 (从 .env 读取)
        tm_collection_name = os.getenv("TM_COLLECTION")
        if not tm_collection_name:
            print("❌ 错误: .env 文件中未找到 TM_COLLECTION 配置")
            return

        collections = [tm_collection_name]  # 只处理 TM 集合
        print(f"✓ 从 .env 读取要重置的集合: {tm_collection_name}")


        for collection_name in collections:
            # 检查并删除旧 collection
            if utility.has_collection(collection_name):
                # 获取旧 collection 的维度
                from pymilvus import Collection
                old_col = Collection(collection_name)
                try:
                    # 尝试获取schema信息
                    schema = old_col.schema
                    for field in schema.fields:
                        if field.name == "vector":
                            old_dim = field.params.get('dim', 0)
                            if old_dim != dimension:
                                print(f"⚠️  {collection_name}: 旧维度={old_dim}, 新维度={dimension}, 需要重建")
                            else:
                                print(f"✓ {collection_name}: 维度匹配 ({dimension}), 无需重建")
                                continue
                except:
                    pass
                
                # 删除旧 collection
                utility.drop_collection(collection_name)
                print(f"✓ 已删除 {collection_name}")
            else:
                print(f"ℹ️  {collection_name} 不存在，将创建新的")
        
        # 断开连接
        connections.disconnect("default")
        print(f"\n✓ 完成！现在可以重新导入数据了")
        print(f"\n建议命令:")
        print(f"python scripts/import_tm_to_db.py \\")
        print(f"  dataset/processed/train_set_zh_en.json \\")
        print(f"  --source-lang zh --target-lang en \\")
        print(f"  --use-embeddings --embedding-batch-size 20")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    load_dotenv()
    import argparse
    parser = argparse.ArgumentParser(description='重置 Milvus collections')
    parser.add_argument('--host', default='localhost', help='Milvus host')
    parser.add_argument('--port', default='19530', help='Milvus port')
    args = parser.parse_args()
    
    reset_collections(args.host, args.port)

