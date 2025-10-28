"""
数据集加载和预处理
"""
import os
import json
import yaml
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class LegalDocument:
    """法律文档"""
    id: str
    title: str
    language: str
    content: str
    articles: List[Dict[str, Any]]  # 条款列表


@dataclass
class TestSample:
    """测试样本"""
    id: str
    source: str
    target: str
    src_lang: str
    tgt_lang: str
    document_id: str
    article_id: str
    metadata: Dict[str, Any]


class LegalDataset:
    """法律数据集"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.documents: Dict[str, LegalDocument] = {}
        self.test_samples: List[TestSample] = []
    
    def load_documents(self, language: str) -> List[LegalDocument]:
        """加载指定语言的法律文档"""
        docs = []
        lang_dir = os.path.join(self.data_dir, language)
        
        if not os.path.exists(lang_dir):
            print(f"Warning: {lang_dir} not found")
            return docs
        
        for filename in os.listdir(lang_dir):
            if filename.endswith('.json'):
                doc_path = os.path.join(lang_dir, filename)
                with open(doc_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    doc = LegalDocument(
                        id=data['id'],
                        title=data['title'],
                        language=language,
                        content=data['content'],
                        articles=data.get('articles', [])
                    )
                    docs.append(doc)
                    self.documents[doc.id] = doc
        
        return docs
    
    def create_test_samples(self, test_documents: List[str], directions: List[Tuple[str, str]]) -> List[TestSample]:
        """创建测试样本"""
        samples = []
        
        for doc_id in test_documents:
            if doc_id not in self.documents:
                print(f"Warning: Document {doc_id} not found")
                continue
            
            doc = self.documents[doc_id]
            
            # 为每个翻译方向创建样本
            for src_lang, tgt_lang in directions:
                # 查找对应的目标语言文档
                tgt_doc_id = f"{doc_id}_{tgt_lang}"
                if tgt_doc_id not in self.documents:
                    print(f"Warning: Target document {tgt_doc_id} not found")
                    continue
                
                tgt_doc = self.documents[tgt_doc_id]
                
                # 创建句子级样本
                for i, (src_article, tgt_article) in enumerate(zip(doc.articles, tgt_doc.articles)):
                    sample = TestSample(
                        id=f"{doc_id}_{i}_{src_lang}_{tgt_lang}",
                        source=src_article['text'],
                        target=tgt_article['text'],
                        src_lang=src_lang,
                        tgt_lang=tgt_lang,
                        document_id=doc_id,
                        article_id=src_article.get('id', str(i)),
                        metadata={
                            'title': doc.title,
                            'article_title': src_article.get('title', ''),
                            'article_number': src_article.get('number', ''),
                            'length': len(src_article['text'])
                        }
                    )
                    samples.append(sample)
        
        self.test_samples = samples
        return samples
    
    def get_samples_by_direction(self, src_lang: str, tgt_lang: str) -> List[TestSample]:
        """获取指定方向的测试样本"""
        return [s for s in self.test_samples if s.src_lang == src_lang and s.tgt_lang == tgt_lang]
    
    def get_samples_by_document(self, doc_id: str) -> List[TestSample]:
        """获取指定文档的测试样本"""
        return [s for s in self.test_samples if s.document_id == doc_id]
    
    def save_samples(self, output_path: str):
        """保存测试样本"""
        data = []
        for sample in self.test_samples:
            data.append({
                'id': sample.id,
                'source': sample.source,
                'target': sample.target,
                'src_lang': sample.src_lang,
                'tgt_lang': sample.tgt_lang,
                'document_id': sample.document_id,
                'article_id': sample.article_id,
                'metadata': sample.metadata
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load_samples(self, input_path: str):
        """加载测试样本"""
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.test_samples = []
        for item in data:
            sample = TestSample(
                id=item['id'],
                source=item['source'],
                target=item['target'],
                src_lang=item['src_lang'],
                tgt_lang=item['tgt_lang'],
                document_id=item['document_id'],
                article_id=item['article_id'],
                metadata=item['metadata']
            )
            self.test_samples.append(sample)


# 使用示例
def create_sample_dataset():
    """创建示例数据集"""
    dataset = LegalDataset()
    
    # 加载中文文档
    zh_docs = dataset.load_documents("zh")
    print(f"Loaded {len(zh_docs)} Chinese documents")
    
    # 加载英文文档
    en_docs = dataset.load_documents("en")
    print(f"Loaded {len(en_docs)} English documents")
    
    # 创建测试样本
    test_docs = ["CompanyLaw", "ContractLaw", "TortLaw"]
    directions = [("zh", "en"), ("zh", "ja")]
    
    samples = dataset.create_test_samples(test_docs, directions)
    print(f"Created {len(samples)} test samples")
    
    # 保存样本
    dataset.save_samples("outputs/test_samples.json")
    
    return dataset


if __name__ == "__main__":
    dataset = create_sample_dataset()
