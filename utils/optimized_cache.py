#!/usr/bin/env python3
"""
优化的缓存管理器 - 高性能缓存实现
"""

import os
import json
import hashlib
import time
import threading
import sqlite3
import pickle
import lz4.frame
from typing import Dict, Any, Optional, Tuple
from PIL import Image
import numpy as np
import logging
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime, timedelta
import io

logger = logging.getLogger(__name__)

class OptimizedCacheManager:
    """高性能缓存管理器"""
    
    def __init__(self, cache_dir: str = "cache", max_cache_size_mb: int = 1024, 
                 max_age_hours: int = 24, compression_level: int = 3):
        """
        初始化优化缓存管理器
        
        Args:
            cache_dir: 缓存目录
            max_cache_size_mb: 最大缓存大小(MB)
            max_age_hours: 缓存最大保存时间(小时)
            compression_level: 压缩级别(1-12)
        """
        self.cache_dir = Path(cache_dir)
        self.max_cache_size_bytes = max_cache_size_mb * 1024 * 1024
        self.max_age_hours = max_age_hours
        self.compression_level = compression_level
        
        # 线程安全
        self._lock = threading.RLock()
        
        # 创建缓存目录
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # SQLite数据库用于元数据管理
        self.db_path = self.cache_dir / "cache.db"
        self._init_database()
        
        # 内存缓存用于热数据
        self._memory_cache: Dict[str, Tuple[Image.Image, float]] = {}
        self._memory_cache_size = 0
        self._max_memory_cache_mb = 256
        
        logger.info(f"OptimizedCacheManager initialized: {cache_dir}, "
                   f"max_size={max_cache_size_mb}MB, max_age={max_age_hours}h")
    
    def _init_database(self):
        """初始化SQLite数据库"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS cache_entries (
                    cache_key TEXT PRIMARY KEY,
                    image_hash TEXT NOT NULL,
                    processor_type TEXT NOT NULL,
                    params_hash TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    last_accessed TIMESTAMP NOT NULL,
                    access_count INTEGER DEFAULT 1
                )
            ''')
            
            # 创建索引
            conn.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON cache_entries(created_at)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_last_accessed ON cache_entries(last_accessed)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_image_hash ON cache_entries(image_hash)')
            
            conn.commit()
    
    @contextmanager
    def _db_connection(self):
        """数据库连接上下文管理器"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def _get_image_hash(self, image: Image.Image) -> str:
        """计算图片哈希值（优化版本）"""
        # 使用图片的关键属性计算哈希，避免完整序列化
        img_info = f"{image.size}_{image.mode}"
        
        # 对图片数据进行采样哈希，减少计算量
        if hasattr(image, 'tobytes'):
            # 采样部分像素数据
            img_array = np.array(image)
            if img_array.size > 10000:  # 大图片采样
                step = max(1, img_array.size // 10000)
                sample_data = img_array.flatten()[::step]
            else:
                sample_data = img_array.flatten()
            
            hash_input = img_info + str(hash(sample_data.tobytes()))
        else:
            hash_input = img_info
        
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def _get_cache_key(self, image_hash: str, processor_type: str, params: Dict[str, Any]) -> str:
        """生成缓存键"""
        # 优化参数哈希计算
        params_str = json.dumps(params, sort_keys=True, separators=(',', ':'))
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:16]  # 截短哈希
        
        return f"{image_hash[:16]}_{processor_type}_{params_hash}"
    
    def _compress_image(self, image: Image.Image) -> bytes:
        """压缩图片数据"""
        # 将图片转换为字节流
        img_bytes = io.BytesIO()
        
        # 根据图片大小选择压缩策略
        if max(image.size) > 1024:
            # 大图片使用JPEG压缩
            image.save(img_bytes, format='JPEG', quality=85, optimize=True)
        else:
            # 小图片使用PNG
            image.save(img_bytes, format='PNG', optimize=True)
        
        img_data = img_bytes.getvalue()
        
        # 使用LZ4压缩
        compressed_data = lz4.frame.compress(img_data, compression_level=self.compression_level)
        
        return compressed_data
    
    def _decompress_image(self, compressed_data: bytes) -> Image.Image:
        """解压缩图片数据"""
        # LZ4解压缩
        img_data = lz4.frame.decompress(compressed_data)
        
        # 从字节流创建图片
        img_bytes = io.BytesIO(img_data)
        image = Image.open(img_bytes)
        
        return image.copy()  # 返回副本避免文件句柄问题
    
    def get_cached_result(self, image: Image.Image, processor_type: str, 
                         params: Dict[str, Any]) -> Optional[Image.Image]:
        """获取缓存结果（优化版本）"""
        with self._lock:
            try:
                # 计算缓存键
                image_hash = self._get_image_hash(image)
                cache_key = self._get_cache_key(image_hash, processor_type, params)
                
                # 首先检查内存缓存
                if cache_key in self._memory_cache:
                    cached_image, cache_time = self._memory_cache[cache_key]
                    if time.time() - cache_time < self.max_age_hours * 3600:
                        logger.debug(f"Memory cache hit: {cache_key}")
                        return cached_image.copy()
                    else:
                        # 内存缓存过期
                        del self._memory_cache[cache_key]
                
                # 检查磁盘缓存
                with self._db_connection() as conn:
                    cursor = conn.execute(
                        'SELECT file_path, created_at FROM cache_entries WHERE cache_key = ?',
                        (cache_key,)
                    )
                    row = cursor.fetchone()
                    
                    if not row:
                        return None
                    
                    file_path, created_at = row['file_path'], row['created_at']
                    
                    # 检查是否过期
                    created_time = datetime.fromisoformat(created_at)
                    if datetime.now() - created_time > timedelta(hours=self.max_age_hours):
                        self._remove_cache_entry(cache_key, conn)
                        return None
                    
                    # 检查文件是否存在
                    cache_file = self.cache_dir / file_path
                    if not cache_file.exists():
                        self._remove_cache_entry(cache_key, conn)
                        return None
                    
                    # 读取并解压缩图片
                    with open(cache_file, 'rb') as f:
                        compressed_data = f.read()
                    
                    cached_image = self._decompress_image(compressed_data)
                    
                    # 更新访问记录
                    conn.execute(
                        'UPDATE cache_entries SET last_accessed = ?, access_count = access_count + 1 WHERE cache_key = ?',
                        (datetime.now().isoformat(), cache_key)
                    )
                    conn.commit()
                    
                    # 添加到内存缓存
                    self._add_to_memory_cache(cache_key, cached_image)
                    
                    logger.debug(f"Disk cache hit: {cache_key}")
                    return cached_image
                    
            except Exception as e:
                logger.error(f"Cache read error: {e}")
                return None
    
    def save_result(self, image: Image.Image, result_image: Image.Image, 
                   processor_type: str, params: Dict[str, Any]):
        """保存结果到缓存（优化版本）"""
        with self._lock:
            try:
                # 计算缓存键
                image_hash = self._get_image_hash(image)
                cache_key = self._get_cache_key(image_hash, processor_type, params)
                
                # 压缩图片数据
                compressed_data = self._compress_image(result_image)
                
                # 生成文件路径
                file_name = f"{cache_key}.lz4"
                cache_file = self.cache_dir / file_name
                
                # 写入文件
                with open(cache_file, 'wb') as f:
                    f.write(compressed_data)
                
                file_size = len(compressed_data)
                
                # 更新数据库
                with self._db_connection() as conn:
                    params_hash = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()
                    
                    conn.execute('''
                        INSERT OR REPLACE INTO cache_entries 
                        (cache_key, image_hash, processor_type, params_hash, file_path, 
                         file_size, created_at, last_accessed)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        cache_key, image_hash, processor_type, params_hash,
                        file_name, file_size, datetime.now().isoformat(), datetime.now().isoformat()
                    ))
                    conn.commit()
                
                # 添加到内存缓存
                self._add_to_memory_cache(cache_key, result_image)
                
                # 清理过期和超量缓存
                self._cleanup_cache()
                
                logger.debug(f"Cached result: {cache_key}, size: {file_size} bytes")
                
            except Exception as e:
                logger.error(f"Cache save error: {e}")
    
    def _add_to_memory_cache(self, cache_key: str, image: Image.Image):
        """添加到内存缓存"""
        # 估算图片内存大小
        img_size = image.size[0] * image.size[1] * len(image.getbands()) * 4  # 假设32位
        
        # 检查内存缓存大小限制
        max_memory_bytes = self._max_memory_cache_mb * 1024 * 1024
        
        if self._memory_cache_size + img_size > max_memory_bytes:
            # 清理最旧的内存缓存项
            self._cleanup_memory_cache()
        
        if img_size < max_memory_bytes:  # 只缓存不太大的图片
            self._memory_cache[cache_key] = (image.copy(), time.time())
            self._memory_cache_size += img_size
    
    def _cleanup_memory_cache(self):
        """清理内存缓存"""
        # 按时间排序，删除最旧的一半
        if not self._memory_cache:
            return
        
        sorted_items = sorted(self._memory_cache.items(), key=lambda x: x[1][1])
        items_to_remove = len(sorted_items) // 2
        
        for i in range(items_to_remove):
            cache_key = sorted_items[i][0]
            del self._memory_cache[cache_key]
        
        # 重新计算内存使用
        self._memory_cache_size = sum(
            img.size[0] * img.size[1] * len(img.getbands()) * 4
            for img, _ in self._memory_cache.values()
        )
    
    def _remove_cache_entry(self, cache_key: str, conn: sqlite3.Connection):
        """删除缓存项"""
        # 获取文件路径
        cursor = conn.execute('SELECT file_path FROM cache_entries WHERE cache_key = ?', (cache_key,))
        row = cursor.fetchone()
        
        if row:
            file_path = row['file_path']
            cache_file = self.cache_dir / file_path
            
            # 删除文件
            if cache_file.exists():
                cache_file.unlink()
        
        # 从数据库删除
        conn.execute('DELETE FROM cache_entries WHERE cache_key = ?', (cache_key,))
        
        # 从内存缓存删除
        if cache_key in self._memory_cache:
            del self._memory_cache[cache_key]
    
    def _cleanup_cache(self):
        """清理缓存"""
        with self._db_connection() as conn:
            # 清理过期项
            cutoff_time = datetime.now() - timedelta(hours=self.max_age_hours)
            
            cursor = conn.execute(
                'SELECT cache_key FROM cache_entries WHERE created_at < ?',
                (cutoff_time.isoformat(),)
            )
            expired_keys = [row['cache_key'] for row in cursor.fetchall()]
            
            for cache_key in expired_keys:
                self._remove_cache_entry(cache_key, conn)
            
            # 检查总大小限制
            cursor = conn.execute('SELECT SUM(file_size) as total_size FROM cache_entries')
            total_size = cursor.fetchone()['total_size'] or 0
            
            if total_size > self.max_cache_size_bytes:
                # 删除最少访问的项
                cursor = conn.execute('''
                    SELECT cache_key FROM cache_entries 
                    ORDER BY access_count ASC, last_accessed ASC
                ''')
                
                for row in cursor.fetchall():
                    if total_size <= self.max_cache_size_bytes * 0.8:  # 清理到80%
                        break
                    
                    cache_key = row['cache_key']
                    
                    # 获取文件大小
                    size_cursor = conn.execute('SELECT file_size FROM cache_entries WHERE cache_key = ?', (cache_key,))
                    size_row = size_cursor.fetchone()
                    if size_row:
                        total_size -= size_row['file_size']
                    
                    self._remove_cache_entry(cache_key, conn)
            
            conn.commit()
            
            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self._lock:
            with self._db_connection() as conn:
                cursor = conn.execute('''
                    SELECT 
                        COUNT(*) as entry_count,
                        SUM(file_size) as total_size,
                        AVG(access_count) as avg_access_count
                    FROM cache_entries
                ''')
                stats = cursor.fetchone()
                
                return {
                    'disk_cache_entries': stats['entry_count'] or 0,
                    'disk_cache_size_bytes': stats['total_size'] or 0,
                    'disk_cache_size_mb': (stats['total_size'] or 0) / (1024 * 1024),
                    'memory_cache_entries': len(self._memory_cache),
                    'memory_cache_size_mb': self._memory_cache_size / (1024 * 1024),
                    'avg_access_count': stats['avg_access_count'] or 0,
                    'max_cache_size_mb': self.max_cache_size_bytes / (1024 * 1024),
                    'max_age_hours': self.max_age_hours
                }
    
    def clear_cache(self):
        """清空所有缓存"""
        with self._lock:
            try:
                # 清空内存缓存
                self._memory_cache.clear()
                self._memory_cache_size = 0
                
                # 删除所有缓存文件
                for cache_file in self.cache_dir.glob("*.lz4"):
                    cache_file.unlink()
                
                # 清空数据库
                with self._db_connection() as conn:
                    conn.execute('DELETE FROM cache_entries')
                    conn.commit()
                
                logger.info("Cache cleared successfully")
                
            except Exception as e:
                logger.error(f"Cache clear error: {e}")

# 全局优化缓存管理器实例
optimized_cache_manager = OptimizedCacheManager()