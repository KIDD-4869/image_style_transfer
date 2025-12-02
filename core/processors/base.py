#!/usr/bin/env python3
"""
处理器基类 - 定义统一接口
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Dict, Any
from PIL import Image
from dataclasses import dataclass


class ProcessingStrategy(Enum):
    """处理策略枚举"""
    FAST = "fast"           # 快速处理（5-10秒）
    BALANCED = "balanced"   # 平衡模式（15-30秒）
    QUALITY = "quality"     # 高质量（30-60秒）


@dataclass
class ProcessingResult:
    """处理结果"""
    success: bool
    image: Optional[Image.Image] = None
    error_message: Optional[str] = None
    processing_time: float = 0.0
    metadata: Optional[Dict[str, Any]] = None


class BaseProcessor(ABC):
    """处理器基类"""
    
    def __init__(self, name: str, description: str):
        """
        初始化处理器
        
        Args:
            name: 处理器名称
            description: 处理器描述
        """
        self.name = name
        self.description = description
        self.progress_callback = None
        self.task_id = None
    
    @abstractmethod
    def process(
        self, 
        image: Image.Image,
        strategy: ProcessingStrategy = ProcessingStrategy.BALANCED,
        **kwargs
    ) -> ProcessingResult:
        """
        处理图像
        
        Args:
            image: 输入图像
            strategy: 处理策略
            **kwargs: 额外参数
            
        Returns:
            ProcessingResult: 处理结果
        """
        pass
    
    def set_progress_callback(self, callback, task_id: str):
        """
        设置进度回调
        
        Args:
            callback: 回调函数
            task_id: 任务ID
        """
        self.progress_callback = callback
        self.task_id = task_id
    
    def update_progress(self, progress: int, step: int, total_steps: int, loss: float = 0):
        """
        更新进度
        
        Args:
            progress: 进度百分比
            step: 当前步骤
            total_steps: 总步骤数
            loss: 损失值
        """
        if self.progress_callback and self.task_id:
            self.progress_callback(self.task_id, progress, step, total_steps, loss)
    
    def get_info(self) -> Dict[str, Any]:
        """
        获取处理器信息
        
        Returns:
            处理器信息字典
        """
        return {
            'name': self.name,
            'description': self.description,
            'strategies': [s.value for s in ProcessingStrategy]
        }
