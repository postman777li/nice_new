"""
数据模型定义
"""
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class TranslationStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TranslationConfig:
    source: str
    src_lang: str
    tgt_lang: str
    options: Dict[str, Any]


@dataclass
class TranslationStatusInfo:
    job_id: str
    status: TranslationStatus
    progress: int  # 0-100
    current_stage: str
    message: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None


@dataclass
class TranslationResult:
    job_id: str
    success: bool
    trace: Dict[str, Any]
    final: str
    duration: float
    error: Optional[str] = None
