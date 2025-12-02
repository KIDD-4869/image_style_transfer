#!/usr/bin/env python3
"""
内存缓存 - 使用LRU策略的内存缓存实现
"""

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional, Dict
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """缓存项"""
    key: str
    value: Any
    size_bytes: int
    created_at: datetime
    accessed_at: datetime
    access_count: int = 0


class MemoryCache:
    """内存缓存（LRU策略）"""
    
    def __init__(self, max_size_mb: int = 100, ttl_hours: int = 24):
        """
        初始化内存缓存
        
        Args:
            max_size_mb: 最大缓存大小（MB）
            ttl_hours: 缓存项生存时间（小时）
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.ttl = timedelta(hours=ttl_hours)
        
        # 使用OrderedDict实现LRU
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        
        # 统计信息
        self._current_size_bytes = 0
        self._hits = 0
        self._misses = 0
        
        logger.info(f"Memory cache initialized: max_size={max_size_mb}MB, ttl={ttl_hours}h")
    
    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存项
        
        Args:
            key: 缓存键
            
        Returns:
            缓存的值，如果不存在或已过期则返回None
        """
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            entry = self._cache[key]
            
            # 检查是否过期
            if self._is_expired(entry):
                logger.debug(f"Cache entry expired: {key}")
                self._remove_entry(key)
                self._misses += 1
                return None
            
            # 更新访问信息
            entry.accessed_at = datetime.now()
            entry.access_count += 1
            
            # 移到末尾（最近使用）
            self._cache.move_to_end(key)
            
            self._hits += 1
            logger.debug(f"Cache hit: {key}")
            return entry.value
    
    def set(self, key: str, value: Any, size_bytes: int) -> bool:
        """
        设置缓存项
        
        Args:
            key: 缓存键
            value: 缓存值
            size_bytes: 值的大小（字节）
            
        Returns:
            是否成功设置
        """
        with self._lock:
            try:
                # 如果键已存在，先删除旧的
                if key in self._cache:
                    self._remove_entry(key)
                
                # 确保有足够空间
                while self._current_size_bytes + size_bytes > self.max_size_bytes:
                    if not self._evict_lru():
                        logger.warning("Cannot evict more entries, cache full")
                        return False
                
                # 创建新条目
                now = datetime.now()
                entry = CacheEntry(
                    key=key,
                    value=value,
                    size_bytes=size_bytes,
                    created_at=now,
                    accessed_at=now,
                    access_count=0
                )
                
                # 添加到缓存
                self._cache[key] = entry
                self._current_size_bytes += size_bytes
                
                logger.debug(f"Cache set: {key}, size={size_bytes}B")
                return True
                
            except Exception as e:
                logger.error(f"Failed to set cache entry: {e}")
                return False
    
    def remove(self, key: str) -> bool:
        """
        删除缓存项
        
        Args:
            key: 缓存键
            
        Returns:
            是否成功删除
        """
        with self._lock:
            return self._remove_entry(key)
    
    def clear(self) -> None:
        """清空所有缓存"""
        with self._lock:
            self._cache.clear()
            self._current_size_bytes = 0
            logger.info("Memory cache cleared")
    
    def cleanup_expired(self) -> int:
        """
        清理过期项
        
        Returns:
            清理的项数量
        """
        with self._lock:
            expired_keys = []
            
            for key, entry in self._cache.items():
                if self._is_expired(entry):
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._remove_entry(key)
            
            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired entries")
            
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            统计信息字典
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
            
            return {
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate,
                'total_requests': total_requests,
                'items': len(self._cache),
                'size_bytes': self._current_size_bytes,
                'size_mb': self._current_size_bytes / (1024 * 1024),
                'max_size_mb': self.max_size_bytes / (1024 * 1024)
            }
    
    def reset_stats(self) -> None:
        """重置统计计数器"""
        with self._lock:
            self._hits = 0
            self._misses = 0
            logger.info("Cache statistics reset")
    
    def _remove_entry(self, key: str) -> bool:
        """
        内部方法：删除缓存项
        
        Args:
            key: 缓存键
            
        Returns:
            是否成功删除
        """
        if key in self._cache:
            entry = self._cache.pop(key)
            self._current_size_bytes -= entry.size_bytes
            logger.debug(f"Cache entry removed: {key}")
            return True
        return False
    
    def _evict_lru(self) -> bool:
        """
        内部方法：淘汰最久未使用的项
        
        Returns:
            是否成功淘汰
        """
        if not self._cache:
            return False
        
        # OrderedDict的第一项是最久未使用的
        lru_key = next(iter(self._cache))
        self._remove_entry(lru_key)
        logger.debug(f"LRU evicted: {lru_key}")
        return True
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """
        内部方法：检查缓存项是否过期
        
        Args:
            entry: 缓存项
            
        Returns:
            是否过期
        """
        age = datetime.now() - entry.created_at
        return age > self.ttl
    
    def __len__(self) -> int:
        """返回缓存项数量"""
        with self._lock:
            return len(self._cache)
    
    def __contains__(self, key: str) -> bool:
        """检查键是否存在"""
        with self._lock:
            return key in self._cache
    
    def __bool__(self) -> bool:
        """对象总是为True，即使缓存为空"""
        return True
