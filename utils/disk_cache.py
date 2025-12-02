#!/usr/bin/env python3
"""
磁盘缓存 - 持久化缓存实现
"""

import os
import pickle
import threading
from pathlib import Path
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)


class DiskCache:
    """磁盘缓存"""
    
    def __init__(self, cache_dir: str = "./cache", max_size_mb: int = 1000):
        """
        初始化磁盘缓存
        
        Args:
            cache_dir: 缓存目录路径
            max_size_mb: 最大缓存大小（MB）
        """
        self.cache_dir = Path(cache_dir)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self._lock = threading.RLock()
        
        # 创建缓存目录
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Disk cache initialized: dir={cache_dir}, max_size={max_size_mb}MB")
    
    def get(self, key: str) -> Optional[Any]:
        """
        从磁盘加载缓存
        
        Args:
            key: 缓存键
            
        Returns:
            缓存的值，如果不存在或损坏则返回None
        """
        with self._lock:
            try:
                cache_file = self._get_cache_file(key)
                
                if not cache_file.exists():
                    return None
                
                # 读取缓存文件
                with open(cache_file, 'rb') as f:
                    value = pickle.load(f)
                
                # 更新访问时间（通过touch）
                cache_file.touch()
                
                logger.debug(f"Disk cache hit: {key}")
                return value
                
            except (pickle.PickleError, EOFError, IOError) as e:
                # 文件损坏，删除它
                logger.warning(f"Corrupted cache file: {key}, error: {e}")
                self._remove_file(key)
                return None
            except Exception as e:
                logger.error(f"Failed to load from disk cache: {e}")
                return None
    
    def set(self, key: str, value: Any) -> bool:
        """
        保存到磁盘
        
        Args:
            key: 缓存键
            value: 缓存值
            
        Returns:
            是否成功保存
        """
        with self._lock:
            try:
                # 确保有足够空间
                self._ensure_space()
                
                cache_file = self._get_cache_file(key)
                
                # 序列化并保存
                with open(cache_file, 'wb') as f:
                    pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                logger.debug(f"Disk cache set: {key}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to save to disk cache: {e}")
                return False
    
    def remove(self, key: str) -> bool:
        """
        删除缓存文件
        
        Args:
            key: 缓存键
            
        Returns:
            是否成功删除
        """
        with self._lock:
            return self._remove_file(key)
    
    def clear(self) -> None:
        """清空磁盘缓存"""
        with self._lock:
            try:
                # 删除所有缓存文件
                for cache_file in self.cache_dir.glob('*.cache'):
                    cache_file.unlink()
                
                logger.info("Disk cache cleared")
                
            except Exception as e:
                logger.error(f"Failed to clear disk cache: {e}")
    
    def cleanup_old_files(self) -> int:
        """
        清理旧文件以释放空间
        
        Returns:
            清理的文件数量
        """
        with self._lock:
            try:
                # 获取当前大小
                current_size = self._get_total_size()
                
                if current_size <= self.max_size_bytes:
                    return 0
                
                # 按修改时间排序（最旧的在前）
                files = sorted(
                    self.cache_dir.glob('*.cache'),
                    key=lambda f: f.stat().st_mtime
                )
                
                cleaned = 0
                for cache_file in files:
                    if current_size <= self.max_size_bytes * 0.9:  # 保留10%余量
                        break
                    
                    file_size = cache_file.stat().st_size
                    cache_file.unlink()
                    current_size -= file_size
                    cleaned += 1
                
                if cleaned > 0:
                    logger.info(f"Cleaned up {cleaned} old cache files")
                
                return cleaned
                
            except Exception as e:
                logger.error(f"Failed to cleanup old files: {e}")
                return 0
    
    def get_stats(self) -> dict:
        """
        获取磁盘缓存统计
        
        Returns:
            统计信息字典
        """
        with self._lock:
            try:
                files = list(self.cache_dir.glob('*.cache'))
                total_size = sum(f.stat().st_size for f in files)
                
                return {
                    'items': len(files),
                    'size_bytes': total_size,
                    'size_mb': total_size / (1024 * 1024),
                    'max_size_mb': self.max_size_bytes / (1024 * 1024)
                }
                
            except Exception as e:
                logger.error(f"Failed to get disk cache stats: {e}")
                return {
                    'items': 0,
                    'size_bytes': 0,
                    'size_mb': 0.0,
                    'max_size_mb': self.max_size_bytes / (1024 * 1024)
                }
    
    def _get_cache_file(self, key: str) -> Path:
        """
        获取缓存文件路径
        
        Args:
            key: 缓存键
            
        Returns:
            缓存文件路径
        """
        # 使用键的哈希作为文件名，避免特殊字符问题
        import hashlib
        filename = hashlib.md5(key.encode()).hexdigest() + '.cache'
        return self.cache_dir / filename
    
    def _remove_file(self, key: str) -> bool:
        """
        删除缓存文件
        
        Args:
            key: 缓存键
            
        Returns:
            是否成功删除
        """
        try:
            cache_file = self._get_cache_file(key)
            if cache_file.exists():
                cache_file.unlink()
                logger.debug(f"Disk cache file removed: {key}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to remove cache file: {e}")
            return False
    
    def _get_total_size(self) -> int:
        """
        获取缓存目录总大小
        
        Returns:
            总大小（字节）
        """
        try:
            return sum(
                f.stat().st_size 
                for f in self.cache_dir.glob('*.cache')
            )
        except Exception as e:
            logger.error(f"Failed to get total size: {e}")
            return 0
    
    def _ensure_space(self) -> None:
        """确保有足够的磁盘空间"""
        current_size = self._get_total_size()
        
        # 如果超过90%容量，清理旧文件
        if current_size > self.max_size_bytes * 0.9:
            self.cleanup_old_files()
    
    def __len__(self) -> int:
        """返回缓存文件数量"""
        with self._lock:
            return len(list(self.cache_dir.glob('*.cache')))
    
    def __bool__(self) -> bool:
        """对象总是为True，即使缓存为空"""
        return True
