#!/usr/bin/env python3
"""
优化的图像处理管道
"""

import torch
import numpy as np
from PIL import Image
import cv2
from typing import Tuple, Optional, Callable
from torchvision import transforms
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class OptimizedProcessingPipeline:
    """优化的图像处理管道，减少内存分配和提高处理速度"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self._transform_cache = {}
        self._setup_transforms()
    
    def _setup_transforms(self):
        """预设置常用的变换，避免重复创建"""
        self._transform_cache = {
            'normalize': transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
            'to_tensor': transforms.ToTensor(),
            'to_pil': transforms.ToPILImage()
        }
    
    @contextmanager
    def memory_efficient_processing(self):
        """内存高效处理的上下文管理器"""
        try:
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            yield
        finally:
            # 处理完成后清理
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def preprocess_image(self, image: Image.Image, target_size: Optional[int] = None) -> torch.Tensor:
        """优化的图像预处理"""
        with self.memory_efficient_processing():
            # 尺寸优化
            if target_size and max(image.size) > target_size:
                image = self._resize_keep_aspect_ratio(image, target_size)
            
            # 转换为张量
            tensor = self._transform_cache['to_tensor'](image).unsqueeze(0)
            tensor = self._transform_cache['normalize'](tensor)
            
            return tensor.to(self.device, non_blocking=True)
    
    def postprocess_tensor(self, tensor: torch.Tensor, original_size: Tuple[int, int]) -> Image.Image:
        """优化的张量后处理"""
        with self.memory_efficient_processing():
            # 移到CPU并反归一化
            tensor = tensor.squeeze(0).cpu()
            
            # 反归一化
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            tensor = tensor * std + mean
            tensor = torch.clamp(tensor, 0, 1)
            
            # 转换为PIL图像
            image = self._transform_cache['to_pil'](tensor)
            
            # 恢复原始尺寸
            if image.size != original_size:
                image = image.resize(original_size, Image.LANCZOS)
            
            return image
    
    def _resize_keep_aspect_ratio(self, image: Image.Image, max_size: int) -> Image.Image:
        """保持宽高比的尺寸调整"""
        w, h = image.size
        if max(w, h) <= max_size:
            return image
        
        scale = max_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        return image.resize((new_w, new_h), Image.LANCZOS)
    
    def batch_process_features(self, tensors: list, model: torch.nn.Module, 
                             layers: list) -> dict:
        """批量特征提取优化"""
        features = {}
        
        with torch.no_grad():
            for tensor in tensors:
                x = tensor
                for name, layer in model._modules.items():
                    x = layer(x)
                    if name in layers:
                        if name not in features:
                            features[name] = []
                        features[name].append(x)
        
        return features
    
    def optimize_cv_processing(self, image_bgr: np.ndarray) -> np.ndarray:
        """优化的OpenCV处理管道"""
        # 使用内存映射减少复制
        if not image_bgr.flags.writeable:
            image_bgr = image_bgr.copy()
        
        # 批量处理多个操作
        height, width = image_bgr.shape[:2]
        
        # 预分配结果数组
        result = np.empty_like(image_bgr)
        
        # 使用OpenCV的优化函数
        cv2.bilateralFilter(image_bgr, 9, 75, 75, dst=result)
        
        return result
    
    def memory_efficient_kmeans(self, image: np.ndarray, k: int = 16) -> np.ndarray:
        """内存高效的K-means颜色量化"""
        # 重塑数据
        data = image.reshape((-1, 3)).astype(np.float32)
        
        # 使用OpenCV的K-means，内存效率更高
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(
            data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
        
        # 直接在原数据上修改，减少内存分配
        centers = np.uint8(centers)
        quantized_data = centers[labels.flatten()]
        
        return quantized_data.reshape(image.shape)