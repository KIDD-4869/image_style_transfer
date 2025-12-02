#!/usr/bin/env python3
"""
缓存管理器 - 统一管理内存和磁盘缓存
"""

import sys
from typing import Optional
from dataclasses import dataclass
from PIL import Image
import logging

from core.processors.base import ProcessingStrategy, ProcessingResult
from utils.cache_key_generator import CacheKeyGenerator
from utils.memory_cache import MemoryCache
from utils.disk_cache import DiskCache

logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """缓存统计"""
    hits: int
    misses: int
    hit_rate: float
    memory_items: int
    memory_size_mb: float
    disk_items: int
    disk_size_mb: float
    total_requests: int


class CacheManager:
    """缓存管理器 - 两层缓存架构"""
    
    def __init__(
        self,
        memory_size_mb: int = 100,
        disk_size_mb: int = 1000,
        cache_dir: str = "./cache",
        ttl_hours: int = 24,
        enable_disk_cache: bool = True
    ):
        """
        初始化缓存管理器
        
        Args:
            memory_size_mb: 内存缓存大小（MB）
            disk_size_mb: 磁盘缓存大小（MB）
            cache_dir: 缓存目录路径
            ttl_hours: 缓存项生存时间（小时）
            enable_disk_cache: 是否启用磁盘缓存
        """
        self.memory_cache = MemoryCache(
            max_size_mb=memory_size_mb,
            ttl_hours=ttl_hours
        )
        
        self.enable_disk_cache = enable_disk_cache
        if enable_disk_cache:
            self.disk_cache = DiskCache(
                cache_dir=cache_dir,
                max_size_mb=disk_size_mb
            )
        else:
            self.disk_cache = None
        
        logger.info(
            f"Cache manager initialized: "
            f"memory={memory_size_mb}MB, "
            f"disk={disk_size_mb}MB (enabled={enable_disk_cache}), "
            f"ttl={ttl_hours}h"
        )
    
    def get(
        self,
        image: Image.Image,
        strategy: ProcessingStrategy
    ) -> Optional[ProcessingResult]:
        """
        获取缓存的处理结果
        
        查询顺序：内存缓存 -> 磁盘缓存
        
        Args:
            image: 输入图像
            strategy: 处理策略
            
        Returns:
            处理结果或None（未命中）
        """
        try:
            # 生成缓存键
            cache_key = CacheKeyGenerator.generate(image, strategy)
            
            # 1. 检查内存缓存
            result = self.memory_cache.get(cache_key)
            if result is not None:
                logger.debug(f"Memory cache hit: {cache_key}")
                return result
            
            # 2. 检查磁盘缓存
            if self.enable_disk_cache and self.disk_cache:
                serializable_result = self.disk_cache.get(cache_key)
                if serializable_result is not None:
                    logger.debug(f"Disk cache hit: {cache_key}")
                    
                    # 反序列化为ProcessingResult
                    result = self._deserialize_result(serializable_result)
                    
                    # 加载到内存缓存
                    result_size = self._estimate_result_size(result)
                    self.memory_cache.set(cache_key, result, result_size)
                    
                    return result
            
            # 缓存未命中
            logger.debug(f"Cache miss: {cache_key}")
            return None
            
        except Exception as e:
            logger.error(f"Cache get failed: {e}")
            return None
    
    def set(
        self,
        image: Image.Image,
        strategy: ProcessingStrategy,
        result: ProcessingResult
    ) -> bool:
        """
        保存处理结果到缓存
        
        同时保存到内存和磁盘缓存
        
        Args:
            image: 输入图像
            strategy: 处理策略
            result: 处理结果
            
        Returns:
            是否成功保存
        """
        try:
            # 生成缓存键
            cache_key = CacheKeyGenerator.generate(image, strategy)
            
            # 估算结果大小
            result_size = self._estimate_result_size(result)
            
            # 保存到内存缓存（直接保存对象）
            memory_success = self.memory_cache.set(cache_key, result, result_size)
            
            # 保存到磁盘缓存（需要序列化友好的格式）
            disk_success = True
            if self.enable_disk_cache and self.disk_cache:
                # 将ProcessingResult转换为可序列化的格式
                serializable_result = self._make_serializable(result)
                disk_success = self.disk_cache.set(cache_key, serializable_result)
            
            success = memory_success or disk_success
            if success:
                logger.debug(f"Cache set: {cache_key}")
            
            return success
            
        except Exception as e:
            logger.error(f"Cache set failed: {e}")
            return False
    
    def clear(self) -> None:
        """清空所有缓存"""
        try:
            self.memory_cache.clear()
            
            if self.enable_disk_cache and self.disk_cache:
                self.disk_cache.clear()
            
            logger.info("All caches cleared")
            
        except Exception as e:
            logger.error(f"Cache clear failed: {e}")
    
    def get_stats(self) -> CacheStats:
        """
        获取缓存统计信息
        
        Returns:
            缓存统计对象
        """
        try:
            # 获取内存缓存统计
            memory_stats = self.memory_cache.get_stats()
            
            # 获取磁盘缓存统计
            disk_stats = {'items': 0, 'size_mb': 0.0}
            if self.enable_disk_cache and self.disk_cache:
                disk_stats = self.disk_cache.get_stats()
            
            # 合并统计
            total_requests = memory_stats['total_requests']
            hits = memory_stats['hits']
            misses = memory_stats['misses']
            hit_rate = hits / total_requests if total_requests > 0 else 0.0
            
            return CacheStats(
                hits=hits,
                misses=misses,
                hit_rate=hit_rate,
                memory_items=memory_stats['items'],
                memory_size_mb=memory_stats['size_mb'],
                disk_items=disk_stats['items'],
                disk_size_mb=disk_stats['size_mb'],
                total_requests=total_requests
            )
            
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return CacheStats(
                hits=0,
                misses=0,
                hit_rate=0.0,
                memory_items=0,
                memory_size_mb=0.0,
                disk_items=0,
                disk_size_mb=0.0,
                total_requests=0
            )
    
    def cleanup(self) -> None:
        """清理过期和超容量的缓存"""
        try:
            # 清理内存缓存中的过期项
            expired = self.memory_cache.cleanup_expired()
            if expired > 0:
                logger.info(f"Cleaned up {expired} expired memory cache entries")
            
            # 清理磁盘缓存中的旧文件
            if self.enable_disk_cache and self.disk_cache:
                cleaned = self.disk_cache.cleanup_old_files()
                if cleaned > 0:
                    logger.info(f"Cleaned up {cleaned} old disk cache files")
                    
        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")
    
    def _estimate_result_size(self, result: ProcessingResult) -> int:
        """
        估算处理结果的大小
        
        Args:
            result: 处理结果
            
        Returns:
            估算的大小（字节）
        """
        try:
            # 基础大小
            size = sys.getsizeof(result)
            
            # 如果有图像，估算图像大小
            if result.image is not None:
                width, height = result.image.size
                # RGB图像：width * height * 3 bytes
                size += width * height * 3
            
            return size
            
        except Exception as e:
            logger.warning(f"Failed to estimate result size: {e}")
            # 返回一个保守的估计值
            return 1024 * 1024  # 1MB
    
    def _make_serializable(self, result: ProcessingResult) -> dict:
        """
        将ProcessingResult转换为可序列化的字典
        
        Args:
            result: 处理结果
            
        Returns:
            可序列化的字典
        """
        import io
        
        serializable = {
            'success': result.success,
            'error_message': result.error_message,
            'processing_time': result.processing_time,
            'metadata': result.metadata,
            'image_data': None
        }
        
        # 将PIL Image转换为字节
        if result.image is not None:
            img_bytes = io.BytesIO()
            result.image.save(img_bytes, format='PNG')
            serializable['image_data'] = img_bytes.getvalue()
        
        return serializable
    
    def _deserialize_result(self, data: dict) -> ProcessingResult:
        """
        从字典反序列化ProcessingResult
        
        Args:
            data: 序列化的字典
            
        Returns:
            ProcessingResult对象
        """
        import io
        
        # 恢复PIL Image
        image = None
        if data.get('image_data') is not None:
            img_bytes = io.BytesIO(data['image_data'])
            image = Image.open(img_bytes)
        
        return ProcessingResult(
            success=data['success'],
            image=image,
            error_message=data.get('error_message'),
            processing_time=data.get('processing_time', 0.0),
            metadata=data.get('metadata')
        )


# 全局缓存管理器实例
_global_cache_manager: Optional[CacheManager] = None


def get_cache_manager(
    memory_size_mb: int = 100,
    disk_size_mb: int = 1000,
    cache_dir: str = "./cache",
    ttl_hours: int = 24,
    enable_disk_cache: bool = True
) -> CacheManager:
    """
    获取全局缓存管理器实例（单例模式）
    
    Args:
        memory_size_mb: 内存缓存大小（MB）
        disk_size_mb: 磁盘缓存大小（MB）
        cache_dir: 缓存目录路径
        ttl_hours: 缓存项生存时间（小时）
        enable_disk_cache: 是否启用磁盘缓存
        
    Returns:
        缓存管理器实例
    """
    global _global_cache_manager
    
    if _global_cache_manager is None:
        _global_cache_manager = CacheManager(
            memory_size_mb=memory_size_mb,
            disk_size_mb=disk_size_mb,
            cache_dir=cache_dir,
            ttl_hours=ttl_hours,
            enable_disk_cache=enable_disk_cache
        )
    
    return _global_cache_manager
