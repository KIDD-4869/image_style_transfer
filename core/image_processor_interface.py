#!/usr/bin/env python3
"""
图像处理器统一接口
定义所有图像处理模块需要实现的标准接口
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional
from PIL import Image


class ProcessingStyle(Enum):
    """处理风格类型"""
    GHIBLI_CLASSIC = "ghibli_classic"
    GHIBLI_ENHANCED = "ghibli_enhanced"
    GHIBLI_NEURAL = "ghibli_neural"
    ANIME_GAN = "anime_gan"


class ProcessingStatus(Enum):
    """处理状态"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessingResult:
    """处理结果封装"""
    def __init__(self, 
                 success: bool, 
                 image: Optional[Image.Image] = None, 
                 error_message: str = "",
                 processing_time: float = 0.0,
                 metadata: Optional[Dict[str, Any]] = None):
        self.success = success
        self.image = image
        self.error_message = error_message
        self.processing_time = processing_time
        self.metadata = metadata or {}


class ImageProcessorInterface(ABC):
    """图像处理器统一接口"""
    
    def __init__(self, style_type: ProcessingStyle):
        self.style_type = style_type
        self.progress_callback = None
        self.task_id = None
    
    @abstractmethod
    def process(self, image: Image.Image, **kwargs) -> ProcessingResult:
        """
        处理图像的核心方法
        
        Args:
            image: 输入图像
            **kwargs: 其他处理参数
            
        Returns:
            ProcessingResult: 处理结果
        """
        pass
    
    def set_progress_callback(self, callback, task_id):
        """
        设置进度回调函数
        
        Args:
            callback: 进度回调函数
            task_id: 任务ID
        """
        self.progress_callback = callback
        self.task_id = task_id
    
    def _update_progress(self, progress: int, current_step: int, total_steps: int, loss: float = 0.0):
        """
        更新处理进度
        
        Args:
            progress: 进度百分比 (0-100)
            current_step: 当前步骤
            total_steps: 总步骤数
            loss: 损失值（如适用）
        """
        if self.progress_callback and self.task_id:
            self.progress_callback(self.task_id, progress, current_step, total_steps, loss)
    
    @abstractmethod
    def get_processing_info(self) -> Dict[str, Any]:
        """
        获取处理器信息
        
        Returns:
            Dict: 处理器信息字典
        """
        pass


# 工厂类用于创建不同的处理器实例
class ImageProcessorFactory:
    """图像处理器工厂类"""
    
    _processors = {}
    
    @classmethod
    def register_processor(cls, style_type: ProcessingStyle, processor_class):
        """
        注册处理器
        
        Args:
            style_type: 处理风格类型
            processor_class: 处理器类
        """
        cls._processors[style_type] = processor_class
    
    @classmethod
    def create_processor(cls, style_type: ProcessingStyle, **kwargs) -> ImageProcessorInterface:
        """
        创建处理器实例
        
        Args:
            style_type: 处理风格类型
            **kwargs: 处理器初始化参数
            
        Returns:
            ImageProcessorInterface: 处理器实例
            
        Raises:
            ValueError: 不支持的处理风格类型
        """
        if style_type not in cls._processors:
            raise ValueError(f"Unsupported processing style: {style_type}")
        
        return cls._processors[style_type](**kwargs)
    
    @classmethod
    def get_supported_styles(cls) -> list:
        """
        获取支持的处理风格列表
        
        Returns:
            list: 支持的处理风格列表
        """
        return list(cls._processors.keys())