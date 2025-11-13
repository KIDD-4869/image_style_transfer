#!/usr/bin/env python3
"""
模型管理器 - 优化模型加载和内存管理
"""

import torch
import threading
import gc
import psutil
import os
from typing import Optional, Dict, Any
from torchvision import models
import logging

logger = logging.getLogger(__name__)

class ModelManager:
    """单例模式的模型管理器，优化内存使用和模型加载"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._models: Dict[str, torch.nn.Module] = {}
        self._model_lock = threading.RLock()
        
        # 内存管理配置
        self.max_memory_usage = 0.8  # 最大内存使用率
        self.cleanup_threshold = 0.7  # 内存清理阈值
        
        logger.info(f"ModelManager initialized with device: {self.device}")
    
    def get_vgg19(self) -> torch.nn.Module:
        """获取VGG19模型，使用延迟加载和内存优化"""
        with self._model_lock:
            if 'vgg19' not in self._models:
                self._check_memory_and_cleanup()
                
                try:
                    # 使用新的权重API
                    vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
                    
                    # 冻结参数减少内存占用
                    for param in vgg.parameters():
                        param.requires_grad = False
                    
                    # 设置为评估模式
                    vgg.eval()
                    vgg = vgg.to(self.device)
                    
                    self._models['vgg19'] = vgg
                    logger.info("VGG19 model loaded successfully")
                    
                except Exception as e:
                    logger.error(f"Failed to load VGG19 model: {e}")
                    raise
            
            return self._models['vgg19']
    
    def get_segmentation_model(self) -> torch.nn.Module:
        """获取语义分割模型"""
        with self._model_lock:
            if 'deeplabv3' not in self._models:
                self._check_memory_and_cleanup()
                
                try:
                    model = models.segmentation.deeplabv3_resnet50(
                        weights=models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
                    )
                    model.eval()
                    model = model.to(self.device)
                    
                    self._models['deeplabv3'] = model
                    logger.info("DeepLabV3 model loaded successfully")
                    
                except Exception as e:
                    logger.error(f"Failed to load DeepLabV3 model: {e}")
                    raise
            
            return self._models['deeplabv3']
    
    def _check_memory_and_cleanup(self):
        """检查内存使用并在必要时清理"""
        memory_percent = psutil.virtual_memory().percent / 100.0
        
        if memory_percent > self.cleanup_threshold:
            logger.warning(f"Memory usage high: {memory_percent:.1%}, performing cleanup")
            self._cleanup_memory()
    
    def _cleanup_memory(self):
        """清理内存"""
        # 清理PyTorch缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 强制垃圾回收
        gc.collect()
        
        logger.info("Memory cleanup completed")
    
    def unload_model(self, model_name: str):
        """卸载指定模型释放内存"""
        with self._model_lock:
            if model_name in self._models:
                del self._models[model_name]
                self._cleanup_memory()
                logger.info(f"Model {model_name} unloaded")
    
    def get_memory_info(self) -> Dict[str, Any]:
        """获取内存使用信息"""
        memory = psutil.virtual_memory()
        gpu_memory = {}
        
        if torch.cuda.is_available():
            gpu_memory = {
                'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
                'cached': torch.cuda.memory_reserved() / 1024**3,  # GB
            }
        
        return {
            'system_memory': {
                'total': memory.total / 1024**3,  # GB
                'available': memory.available / 1024**3,  # GB
                'percent': memory.percent
            },
            'gpu_memory': gpu_memory,
            'loaded_models': list(self._models.keys())
        }

# 全局单例实例
model_manager = ModelManager()