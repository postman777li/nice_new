"""
工具函数
"""
import json
import yaml
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import asyncio
import aiofiles


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            return yaml.safe_load(f)
        else:
            return json.load(f)


def save_json(data: Any, file_path: str, indent: int = 2):
    """保存JSON文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_json(file_path: str) -> Any:
    """加载JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


async def save_json_async(data: Any, file_path: str, indent: int = 2):
    """异步保存JSON文件"""
    async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
        await f.write(json.dumps(data, ensure_ascii=False, indent=indent))


async def load_json_async(file_path: str) -> Any:
    """异步加载JSON文件"""
    async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
        content = await f.read()
        return json.loads(content)


def create_experiment_id(prefix: str = "exp") -> str:
    """创建实验ID"""
    timestamp = int(time.time())
    random_suffix = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
    return f"{prefix}_{timestamp}_{random_suffix}"


def ensure_dir(path: str) -> Path:
    """确保目录存在"""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def get_file_hash(file_path: str) -> str:
    """计算文件哈希值"""
    with open(file_path, 'rb') as f:
        content = f.read()
        return hashlib.md5(content).hexdigest()


def format_duration(seconds: float) -> str:
    """格式化持续时间"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_metrics(metrics: Dict[str, float]) -> str:
    """格式化指标显示"""
    lines = []
    for metric, value in metrics.items():
        lines.append(f"{metric}: {value:.3f}")
    return "\n".join(lines)


def calculate_confidence_interval(values: List[float], confidence: float = 0.95) -> tuple:
    """计算置信区间"""
    import numpy as np
    from scipy import stats
    
    if len(values) < 2:
        return (values[0], values[0]) if values else (0, 0)
    
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    n = len(values)
    
    # 计算t分布的临界值
    alpha = 1 - confidence
    t_critical = stats.t.ppf(1 - alpha/2, n - 1)
    
    # 计算置信区间
    margin_error = t_critical * (std / np.sqrt(n))
    ci_lower = mean - margin_error
    ci_upper = mean + margin_error
    
    return (ci_lower, ci_upper)


def create_progress_bar(total: int, desc: str = "Processing"):
    """创建进度条"""
    from tqdm import tqdm
    return tqdm(total=total, desc=desc, unit="item")


def log_experiment_info(experiment_id: str, config: Dict[str, Any], start_time: float):
    """记录实验信息"""
    log_entry = {
        "experiment_id": experiment_id,
        "start_time": start_time,
        "config": config,
        "status": "started"
    }
    
    log_file = Path("logs") / f"experiment_{experiment_id}.json"
    ensure_dir("logs")
    save_json(log_entry, str(log_file))


def log_experiment_completion(experiment_id: str, end_time: float, status: str = "completed", error: str = None):
    """记录实验完成信息"""
    log_file = Path("logs") / f"experiment_{experiment_id}.json"
    
    if log_file.exists():
        log_entry = load_json(str(log_file))
        log_entry.update({
            "end_time": end_time,
            "duration": end_time - log_entry.get("start_time", end_time),
            "status": status,
            "error": error
        })
        save_json(log_entry, str(log_file))


class ExperimentLogger:
    """实验日志记录器"""
    
    def __init__(self, experiment_id: str):
        self.experiment_id = experiment_id
        self.log_file = Path("logs") / f"experiment_{experiment_id}.json"
        ensure_dir("logs")
        
        # 初始化日志
        self.log_data = {
            "experiment_id": experiment_id,
            "start_time": time.time(),
            "events": [],
            "status": "running"
        }
        self._save_log()
    
    def log_event(self, event_type: str, message: str, data: Dict[str, Any] = None):
        """记录事件"""
        event = {
            "timestamp": time.time(),
            "type": event_type,
            "message": message,
            "data": data or {}
        }
        self.log_data["events"].append(event)
        self._save_log()
    
    def log_error(self, error: str, traceback: str = None):
        """记录错误"""
        self.log_event("error", error, {"traceback": traceback})
        self.log_data["status"] = "failed"
        self._save_log()
    
    def log_completion(self):
        """记录完成"""
        self.log_data["status"] = "completed"
        self.log_data["end_time"] = time.time()
        self.log_data["duration"] = self.log_data["end_time"] - self.log_data["start_time"]
        self._save_log()
    
    def _save_log(self):
        """保存日志"""
        save_json(self.log_data, str(self.log_file))


def validate_config(config: Dict[str, Any]) -> List[str]:
    """验证配置文件"""
    errors = []
    
    # 检查必要字段
    required_fields = ['agent', 'parameters', 'ablations']
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")
    
    # 检查agent配置
    if 'agent' in config:
        agent_required = ['base_url']
        for field in agent_required:
            if field not in config['agent']:
                errors.append(f"Missing agent.{field}")
    
    # 检查参数配置
    if 'parameters' in config:
        if 'directions' not in config['parameters']:
            errors.append("Missing parameters.directions")
        if 'test_set' not in config['parameters']:
            errors.append("Missing parameters.test_set")
    
    return errors


def setup_experiment_environment():
    """设置实验环境"""
    # 创建必要的目录
    directories = ["data", "outputs", "results", "logs", "cache"]
    for directory in directories:
        ensure_dir(directory)
    
    print("Experiment environment setup completed!")


def cleanup_old_experiments(days: int = 7):
    """清理旧的实验文件"""
    import os
    from datetime import datetime, timedelta
    
    cutoff_time = time.time() - (days * 24 * 3600)
    
    # 清理日志文件
    logs_dir = Path("logs")
    if logs_dir.exists():
        for log_file in logs_dir.glob("experiment_*.json"):
            if os.path.getmtime(log_file) < cutoff_time:
                log_file.unlink()
    
    # 清理缓存文件
    cache_dir = Path("outputs/cache")
    if cache_dir.exists():
        for cache_file in cache_dir.glob("*.json"):
            if os.path.getmtime(cache_file) < cutoff_time:
                cache_file.unlink()
    
    print(f"Cleaned up experiment files older than {days} days")


if __name__ == "__main__":
    # 测试工具函数
    print("Testing utility functions...")
    
    # 测试配置验证
    test_config = {
        "agent": {"base_url": "http://localhost:3000"},
        "parameters": {"directions": ["zh-en"], "test_set": ["test"]},
        "ablations": {"test": {}}
    }
    
    errors = validate_config(test_config)
    print(f"Config validation errors: {errors}")
    
    # 设置环境
    setup_experiment_environment()
    
    print("Utility functions test completed!")
