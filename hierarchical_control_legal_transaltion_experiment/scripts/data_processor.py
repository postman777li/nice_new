#!/usr/bin/env python3
"""
数据处理脚本：将 Excel 文件转换为 JSON 格式
用于处理法律文档数据集
"""
import os
import json
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LegalDataProcessor:
    """法律数据处理器"""
    
    def __init__(self, dataset_dir: str = "dataset", output_dir: str = "dataset/processed", config_file: str = "configs/default.yaml"):
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载配置文件
        self.config = self.load_config(config_file)
        self.test_domains = self.config.get("parameters", {}).get("test_set", [])
        
        # 法律领域映射
        self.domain_mapping = {
            # 劳动法领域 (需要放在合同前面，避免劳动合同法被误分类)
            "劳动合同法": "LaborLaw",
            "劳动争议调解仲裁法": "LaborLaw",
            "劳动法": "LaborLaw",
            "工会法": "LaborLaw",
            
            # 测试领域 (5个)
            "公司法": "CompanyLaw",
            "合同法": "ContractLaw", 
            "侵权责任法": "TortLaw",
            "刑法": "CriminalLaw",
            "刑事诉讼法": "CriminalLaw",
            "行政处罚法": "AdministrativeLaw",
            
            # 知识产权法领域
            "专利法": "IntellectualPropertyLaw",
            "商标法": "IntellectualPropertyLaw",
            "著作权法": "IntellectualPropertyLaw",
            
            # 民商法领域
            "民法典": "CivilLaw",
            "涉外民事关系法律适用法": "CivilLaw",
            
            # 宪法行政法领域
            "宪法": "ConstitutionalLaw",
            
            # 数据保护法领域
            "个人信息保护法": "DataProtectionLaw",
            "数据安全法": "DataProtectionLaw",
            
            # 电子商务法领域
            "电子商务法": "ECommerceLaw",
            
            # 竞争法领域
            "反垄断法": "CompetitionLaw",
            "反不正当竞争法": "CompetitionLaw",
            
            # 外商投资法领域
            "外商投资法": "ForeignInvestmentLaw",
            
            # 安全生产法领域
            "安全生产法": "SafetyLaw",
            
            # 广告法领域
            "广告法": "AdvertisingLaw",
            
            # 循环经济法领域
            "循环经济促进法": "EnvironmentalLaw",
            
            # 可再生能源法领域
            "可再生能源法": "EnvironmentalLaw",
            
            # 社会保险法领域
            "社会保险法": "SocialSecurityLaw",
            
            # 种子法领域
            "种子法": "AgriculturalLaw",
            
            # 科学技术法领域
            "科学技术进步法": "TechnologyLaw",
            "科技成果转化促进法": "TechnologyLaw",
            
            # 反间谍法领域
            "反间谍法": "NationalSecurityLaw",
            
            # 统计法领域
            "统计法": "StatisticsLaw",
            
            # 标准化法领域
            "标准化法": "StandardizationLaw",
            
            # 非物质文化遗产法领域
            "非物质文化遗产法": "CulturalHeritageLaw",
            
            # 海警法领域
            "海警法": "MaritimeLaw",
            
            # 出境入境管理法领域
            "出境入境管理法": "ImmigrationLaw",
        }
    
    def load_config(self, config_file: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded config from {config_file}")
            return config
        except Exception as e:
            logger.warning(f"Failed to load config from {config_file}: {e}")
            return {}
    
    def is_test_law(self, law_name: str) -> bool:
        """判断是否为测试法律"""
        return law_name in self.test_domains
    
    def extract_law_name(self, filename: str) -> tuple[str, str, int]:
        """从文件名提取法律名称、领域和年份"""
        # 移除扩展名
        name = filename.replace('.xlsx', '')
        
        # 提取年份
        year = None
        if '(' in name and ')' in name:
            year_str = name.split('(')[-1].split(')')[0]
            try:
                year = int(year_str)
                name = name.split('(')[0]  # 移除年份部分
            except ValueError:
                year = None
        
        # 移除"中华人民共和国"前缀
        if name.startswith('中华人民共和国'):
            name = name[7:]
        
        # 确定领域
        domain = "GeneralLaw"  # 默认领域
        for key, value in self.domain_mapping.items():
            if key in name:
                domain = value
                break
        
        return name, domain, year
    
    def process_excel_file(self, filepath: Path) -> Dict[str, Any]:
        """处理单个 Excel 文件"""
        logger.info(f"Processing {filepath.name}")
        
        try:
            # 读取 Excel 文件
            df = pd.read_excel(filepath)
            
            # 提取法律信息
            law_name, domain, year = self.extract_law_name(filepath.name)
            
            # 处理数据
            processed_data = {
                "metadata": {
                    "law_name": law_name,
                    "domain": domain,
                    "year": year,
                    "source_file": filepath.name,
                    "total_entries": len(df),
                    "processed_at": datetime.now().isoformat(),
                    "languages": ["zh", "ja", "en"]
                },
                "entries": []
            }
            
            # 处理每一行数据
            for idx, row in df.iterrows():
                # 处理ID，如果是NaN或非数字则使用行索引+1
                row_id = row.get('id', idx + 1)
                if pd.isna(row_id):
                    row_id = idx + 1
                else:
                    # 尝试转换为数字，如果失败则使用行索引+1
                    try:
                        row_id = int(float(row_id))
                    except (ValueError, TypeError):
                        logger.warning(f"Non-numeric ID '{row_id}' in {filepath.name}, using row index {idx + 1}")
                        row_id = idx + 1
                
                entry = {
                    "id": int(row_id),
                    "source_file": row.get('source_file', ''),
                    "original_numbers": {
                        "zh": row.get('original_number_zh', ''),
                        "ja": row.get('original_number_ja', ''),
                        "en": row.get('original_number_en', '')
                    },
                    "texts": {
                        "zh": str(row.get('chinese', '')).strip(),
                        "ja": str(row.get('japanese', '')).strip(),
                        "en": str(row.get('english', '')).strip()
                    }
                }
                
                # 过滤空条目
                if any(entry["texts"].values()):
                    processed_data["entries"].append(entry)
            
            processed_data["metadata"]["valid_entries"] = len(processed_data["entries"])
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing {filepath.name}: {e}")
            return None
    
    def extract_terms_for_termbase(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """从数据中提取术语用于术语库"""
        terms = []
        domain = data["metadata"]["domain"]
        
        for entry in data["entries"]:
            zh_text = entry["texts"]["zh"]
            en_text = entry["texts"]["en"]
            ja_text = entry["texts"]["ja"]
            
            # 简单的术语提取（实际应用中可能需要更复杂的NLP处理）
            if zh_text and en_text:
                # 中英术语对
                terms.append({
                    "source_term": zh_text[:50] + "..." if len(zh_text) > 50 else zh_text,
                    "target_term": en_text[:50] + "..." if len(en_text) > 50 else en_text,
                    "source_lang": "zh",
                    "target_lang": "en",
                    "domain": domain,
                    "confidence": 0.8,
                    "context": f"From {data['metadata']['law_name']}",
                    "metadata": {
                        "law": data["metadata"]["law_name"],
                        "year": data["metadata"]["year"],
                        "entry_id": entry["id"]
                    }
                })
            
            if zh_text and ja_text:
                # 中日术语对
                terms.append({
                    "source_term": zh_text[:50] + "..." if len(zh_text) > 50 else zh_text,
                    "target_term": ja_text[:50] + "..." if len(ja_text) > 50 else ja_text,
                    "source_lang": "zh",
                    "target_lang": "ja",
                    "domain": domain,
                    "confidence": 0.8,
                    "context": f"From {data['metadata']['law_name']}",
                    "metadata": {
                        "law": data["metadata"]["law_name"],
                        "year": data["metadata"]["year"],
                        "entry_id": entry["id"]
                    }
                })
        
        return terms
    
    
    def process_all_files(self) -> Dict[str, Any]:
        """处理所有 Excel 文件"""
        results = {
            "processed_files": [],
            "failed_files": [],
            "total_entries": 0,
            "test_entries": 0,
            "train_entries": 0,
            "all_terms": [],
            "all_test_entries": [],
            "all_train_entries": [],
            "domains": set(),
            "test_domains": set(),
            "train_domains": set(),
            "statistics": {}
        }
        
        excel_files = list(self.dataset_dir.glob("*.xlsx"))
        logger.info(f"Found {len(excel_files)} Excel files")
        
        for filepath in excel_files:
            processed_data = self.process_excel_file(filepath)
            
            if processed_data:
                # 保存单个文件的 JSON
                output_file = self.output_dir / f"{filepath.stem}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(processed_data, f, ensure_ascii=False, indent=2)
                
                results["processed_files"].append(filepath.name)
                results["total_entries"] += processed_data["metadata"]["valid_entries"]
                results["domains"].add(processed_data["metadata"]["domain"])
                
                # 判断是否为测试法律
                law_name = processed_data["metadata"]["law_name"]
                domain = processed_data["metadata"]["domain"]
                is_test = self.is_test_law(law_name)
                
                if is_test:
                    results["test_domains"].add(law_name)
                    results["test_entries"] += processed_data["metadata"]["valid_entries"]
                    logger.info(f"Added {law_name} to TEST set (domain: {domain})")
                else:
                    results["train_domains"].add(law_name)
                    results["train_entries"] += processed_data["metadata"]["valid_entries"]
                    logger.info(f"Added {law_name} to TRAIN set (domain: {domain})")
                
                # 提取术语（仅从训练数据）
                if not is_test:
                    terms = self.extract_terms_for_termbase(processed_data)
                    results["all_terms"].extend(terms)

                # 汇总条目（测试集和训练数据分别处理）
                law_name = processed_data["metadata"]["law_name"]
                year = processed_data["metadata"].get("year")
                entries = [
                    {
                        "law": law_name,
                        "domain": domain,
                        "year": year,
                        "id": entry.get("id"),
                        "zh": entry.get("texts", {}).get("zh", ""),
                        "ja": entry.get("texts", {}).get("ja", ""),
                        "en": entry.get("texts", {}).get("en", "")
                    }
                    for entry in processed_data.get("entries", [])
                ]
                
                if is_test:
                    results["all_test_entries"].extend(entries)
                else:
                    results["all_train_entries"].extend(entries)
                
                # 统计信息
                domain = processed_data["metadata"]["domain"]
                if law_name not in results["statistics"]:
                    results["statistics"][law_name] = {
                        "domain": domain,
                        "files": 0,
                        "entries": 0,
                        "terms": 0,
                        "type": "test" if is_test else "train"
                    }
                
                results["statistics"][law_name]["files"] += 1
                results["statistics"][law_name]["entries"] += processed_data["metadata"]["valid_entries"]
                
                # 只有训练数据才统计术语
                if not is_test:
                    results["statistics"][law_name]["terms"] += len(terms)
                
            else:
                results["failed_files"].append(filepath.name)
        
        results["domains"] = list(results["domains"])
        results["test_domains"] = list(results["test_domains"])
        results["train_domains"] = list(results["train_domains"])
        
        return results
    
    def save_consolidated_data(self, results: Dict[str, Any]):
        """保存合并的数据（按语言对拆分）"""

        # 拆分测试集（仅输出相关语言对字段，仅包含测试领域）
        test_zh_en = []
        test_zh_ja = []
        for item in results["all_test_entries"]:
            zh = item.get("zh", "").strip()
            en = item.get("en", "").strip()
            ja = item.get("ja", "").strip()
            base = {
                "law": item.get("law"),
                "domain": item.get("domain"),
                "year": item.get("year"),
                "id": item.get("id"),
            }
            if zh and en:
                test_zh_en.append({**base, "zh": zh, "en": en})
            if zh and ja:
                test_zh_ja.append({**base, "zh": zh, "ja": ja})
        
        # 拆分训练数据（仅输出相关语言对字段，仅包含训练领域）
        train_zh_en = []
        train_zh_ja = []
        for item in results["all_train_entries"]:
            zh = item.get("zh", "").strip()
            en = item.get("en", "").strip()
            ja = item.get("ja", "").strip()
            base = {
                "law": item.get("law"),
                "domain": item.get("domain"),
                "year": item.get("year"),
                "id": item.get("id"),
            }
            if zh and en:
                train_zh_en.append({**base, "zh": zh, "en": en})
            if zh and ja:
                train_zh_ja.append({**base, "zh": zh, "ja": ja})

        # 保存测试集（中英）
        test_en_file = self.output_dir / "test_set_zh_en.json"
        with open(test_en_file, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": {
                    "pair": "zh-en",
                    "total_entries": len(test_zh_en),
                    "domains": results["test_domains"],
                    "created_at": datetime.now().isoformat()
                },
                "entries": test_zh_en
            }, f, ensure_ascii=False, indent=2)

        # 保存测试集（中日）
        test_ja_file = self.output_dir / "test_set_zh_ja.json"
        with open(test_ja_file, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": {
                    "pair": "zh-ja",
                    "total_entries": len(test_zh_ja),
                    "domains": results["test_domains"],
                    "created_at": datetime.now().isoformat()
                },
                "entries": test_zh_ja
            }, f, ensure_ascii=False, indent=2)
        
        # 保存训练数据（中英）
        train_en_file = self.output_dir / "train_set_zh_en.json"
        with open(train_en_file, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": {
                    "pair": "zh-en",
                    "total_entries": len(train_zh_en),
                    "domains": results["train_domains"],
                    "created_at": datetime.now().isoformat()
                },
                "entries": train_zh_en
            }, f, ensure_ascii=False, indent=2)

        # 保存训练数据（中日）
        train_ja_file = self.output_dir / "train_set_zh_ja.json"
        with open(train_ja_file, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": {
                    "pair": "zh-ja",
                    "total_entries": len(train_zh_ja),
                    "domains": results["train_domains"],
                    "created_at": datetime.now().isoformat()
                },
                "entries": train_zh_ja
            }, f, ensure_ascii=False, indent=2)
        
        # 保存处理统计
        stats_file = self.output_dir / "processing_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump({
                "summary": {
                    "processed_files": len(results["processed_files"]),
                    "failed_files": len(results["failed_files"]),
                    "total_entries": results["total_entries"],
                    "test_entries": results["test_entries"],
                    "train_entries": results["train_entries"],
                    "total_terms": len(results["all_terms"]),
                    "total_test_entries": len(results["all_test_entries"]),
                    "total_test_entries_zh_en": len(test_zh_en),
                    "total_test_entries_zh_ja": len(test_zh_ja),
                    "total_train_entries": len(results["all_train_entries"]),
                    "total_train_entries_zh_en": len(train_zh_en),
                    "total_train_entries_zh_ja": len(train_zh_ja),
                    "all_domains": results["domains"],
                    "test_domains": results["test_domains"],
                    "train_domains": results["train_domains"]
                },
                "details": results["statistics"],
                "processed_files": results["processed_files"],
                "failed_files": results["failed_files"]
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved consolidated data:")
        logger.info(f"  - Test Set zh-en: {len(test_zh_en)} entries (test domains only)")
        logger.info(f"  - Test Set zh-ja: {len(test_zh_ja)} entries (test domains only)")
        logger.info(f"  - Train Set zh-en: {len(train_zh_en)} entries (train domains only)")
        logger.info(f"  - Train Set zh-ja: {len(train_zh_ja)} entries (train domains only)")
        logger.info(f"  - Terms extracted: {len(results['all_terms'])} entries (from train data)")
        logger.info(f"  - Test domains: {results['test_domains']}")
        logger.info(f"  - Train domains: {results['train_domains']}")
        logger.info(f"  - Statistics saved to processing_stats.json")


def main():
    parser = argparse.ArgumentParser(description="Process legal dataset Excel files to JSON")
    parser.add_argument("--dataset-dir", default="dataset", help="Dataset directory")
    parser.add_argument("--output-dir", default="dataset/processed", help="Output directory")
    parser.add_argument("--config", default="configs/default.yaml", help="Configuration file")
    parser.add_argument("--single-file", help="Process single file only")
    
    args = parser.parse_args()
    
    processor = LegalDataProcessor(args.dataset_dir, args.output_dir, args.config)
    
    if args.single_file:
        # 处理单个文件
        filepath = Path(args.dataset_dir) / args.single_file
        if filepath.exists():
            result = processor.process_excel_file(filepath)
            if result:
                output_file = Path(args.output_dir) / f"{filepath.stem}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                logger.info(f"Processed {args.single_file} -> {output_file}")
            else:
                logger.error(f"Failed to process {args.single_file}")
        else:
            logger.error(f"File not found: {filepath}")
    else:
        # 处理所有文件
        results = processor.process_all_files()
        processor.save_consolidated_data(results)
        
        logger.info("Processing completed!")
        logger.info(f"Processed: {len(results['processed_files'])} files")
        logger.info(f"Failed: {len(results['failed_files'])} files")
        logger.info(f"Total entries: {results['total_entries']}")
        logger.info(f"Test entries: {results['test_entries']}")
        logger.info(f"Train entries: {results['train_entries']}")
        logger.info(f"All domains: {', '.join(results['domains'])}")
        logger.info(f"Test domains: {', '.join(results['test_domains'])}")
        logger.info(f"Train domains: {', '.join(results['train_domains'])}")


if __name__ == "__main__":
    main()
