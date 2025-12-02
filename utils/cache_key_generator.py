#!/usr/bin/env python3
"""
缓存键生成器 - 为图像处理结果生成唯一的缓存键
"""

import hashlib
import io
from typing import Optional
from PIL import Image
import logging

from core.processors.base import ProcessingStrategy

logger = logging.getLogger(__name__)


class CacheKeyGenerator:
    """缓存键生成器"""
    
    @staticmethod
    def generate(
        image: Image.Image,
        strategy: ProcessingStrategy
    ) -> str:
        """
        生成缓存键
        
        基于图像内容和处理策略生成唯一的缓存键。
        相同的图像和策略将生成相同的键（确定性）。
        
        Args:
            image: 输入图像
            strategy: 处理策略
            
        Returns:
            缓存键字符串，格式为 "image_hash:strategy_value"
            
        Raises:
            ValueError: 如果图像为None或无效
        """
        if image is None:
            raise ValueError("Image cannot be None")
        
        if not isinstance(image, Image.Image):
            raise ValueError("Image must be a PIL Image object")
        
        try:
            # 计算图像哈希
            image_hash = CacheKeyGenerator.hash_image(image)
            
            # 组合图像哈希和策略
            cache_key = f"{image_hash}:{strategy.value}"
            
            return cache_key
            
        except Exception as e:
            logger.error(f"Failed to generate cache key: {e}")
            raise
    
    @staticmethod
    def hash_image(image: Image.Image) -> str:
        """
        计算图像哈希
        
        使用MD5算法计算图像内容的哈希值。
        MD5足够快速且碰撞概率极低，适合缓存场景。
        
        Args:
            image: 输入图像
            
        Returns:
            图像哈希字符串（32字符的十六进制）
            
        Raises:
            ValueError: 如果图像为None或无效
        """
        if image is None:
            raise ValueError("Image cannot be None")
        
        if not isinstance(image, Image.Image):
            raise ValueError("Image must be a PIL Image object")
        
        try:
            # 将图像转换为字节流
            img_bytes = io.BytesIO()
            
            # 使用PNG格式保存以确保无损
            # 这样相同的图像内容总是产生相同的字节流
            image.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            # 计算MD5哈希
            md5_hash = hashlib.md5()
            md5_hash.update(img_bytes.read())
            
            return md5_hash.hexdigest()
            
        except Exception as e:
            logger.error(f"Failed to hash image: {e}")
            raise
    
    @staticmethod
    def validate_key(key: str) -> bool:
        """
        验证缓存键格式
        
        Args:
            key: 缓存键
            
        Returns:
            是否为有效的缓存键格式
        """
        if not key or not isinstance(key, str):
            return False
        
        # 检查格式: hash:strategy
        parts = key.split(':')
        if len(parts) != 2:
            return False
        
        image_hash, strategy = parts
        
        # 检查哈希长度（MD5是32个字符）
        if len(image_hash) != 32:
            return False
        
        # 检查策略是否有效
        valid_strategies = [s.value for s in ProcessingStrategy]
        if strategy not in valid_strategies:
            return False
        
        return True
