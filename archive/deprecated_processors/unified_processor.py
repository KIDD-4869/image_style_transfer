#!/usr/bin/env python3
"""
统一处理器 - 整合所有宫崎骏风格转换方法
"""

from PIL import Image
import os
from typing import Optional

class UnifiedGhibliProcessor:
    """统一的宫崎骏风格处理器"""
    
    def __init__(self):
        self.processors = {}
        self._initialize_processors()
    
    def _initialize_processors(self):
        """初始化所有可用的处理器"""
        
        # 1. 学习型处理器（推荐）
        try:
            from .learning_ghibli_processor import create_learning_processor
            self.processors['learning'] = create_learning_processor()
            print("✅ 学习型处理器已加载")
        except Exception as e:
            print(f"⚠️ 学习型处理器加载失败: {e}")
        
        # 2. 修复版GAN处理器
        try:
            from .fixed_ghibli_gan import process_with_fixed_gan
            self.processors['gan'] = process_with_fixed_gan
            print("✅ 修复版GAN处理器已加载")
        except Exception as e:
            print(f"⚠️ GAN处理器加载失败: {e}")
        
        # 3. 传统CV处理器（备用）
        try:
            from .true_ghibli_style import TrueGhibliStyleTransfer
            self.processors['traditional'] = TrueGhibliStyleTransfer()
            print("✅ 传统处理器已加载")
        except Exception as e:
            print(f"⚠️ 传统处理器加载失败: {e}")
    
    def process_image(self, image_path: str, style_strength: float = 1.0, 
                     method: str = 'auto') -> Image.Image:
        """
        处理图像生成宫崎骏风格
        
        Args:
            image_path: 图像路径
            style_strength: 风格强度
            method: 处理方法 ('auto', 'learning', 'gan', 'traditional')
        """
        
        if method == 'auto':
            # 自动选择最佳处理器
            if 'learning' in self.processors:
                method = 'learning'
            elif 'gan' in self.processors:
                method = 'gan'
            elif 'traditional' in self.processors:
                method = 'traditional'
            else:
                raise RuntimeError("没有可用的处理器")
        
        processor = self.processors.get(method)
        if not processor:
            raise ValueError(f"处理器 '{method}' 不可用")
        
        try:
            if method == 'learning':
                return processor.process_image(image_path, style_strength)
            elif method == 'gan':
                return processor(image_path)
            elif method == 'traditional':
                image = Image.open(image_path)
                return processor.apply_ghibli_style(image)
            
        except Exception as e:
            print(f"⚠️ {method}处理器失败: {e}")
            # 自动降级到下一个可用处理器
            if method != 'traditional' and 'traditional' in self.processors:
                print("🔄 降级到传统处理器")
                return self.process_image(image_path, style_strength, 'traditional')
            raise
    
    def get_available_methods(self):
        """获取可用的处理方法"""
        return list(self.processors.keys())

# 全局实例
unified_processor = UnifiedGhibliProcessor()