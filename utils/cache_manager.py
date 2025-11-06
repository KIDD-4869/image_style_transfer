#!/usr/bin/env python3
"""
处理结果缓存管理器
用于缓存已处理的图片结果，避免重复处理
"""

import hashlib
import os
import json
import time
from typing import Optional, Dict, Any
from PIL import Image
import io
import threading
from datetime import datetime, timedelta


class CacheManager:
    """处理结果缓存管理器"""
    
    def __init__(self, cache_dir: str = "cache", max_cache_size: int = 1000, max_age_hours: int = 24):
        """
        初始化缓存管理器
        
        Args:
            cache_dir: 缓存目录路径
            max_cache_size: 最大缓存项目数
            max_age_hours: 缓存最大保存时间（小时）
        """
        self.cache_dir = cache_dir
        self.max_cache_size = max_cache_size
        self.max_age_hours = max_age_hours
        self.lock = threading.Lock()
        
        # 创建缓存目录
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(os.path.join(self.cache_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.cache_dir, "metadata"), exist_ok=True)
        
        # 缓存索引文件
        self.index_file = os.path.join(self.cache_dir, "cache_index.json")
        self._load_index()
    
    def _load_index(self):
        """加载缓存索引"""
        try:
            if os.path.exists(self.index_file):
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    self.cache_index = json.load(f)
            else:
                self.cache_index = {}
        except Exception as e:
            print(f"⚠️ 缓存索引加载失败: {e}")
            self.cache_index = {}
    
    def _save_index(self):
        """保存缓存索引"""
        try:
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache_index, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ 缓存索引保存失败: {e}")
    
    def _get_image_hash(self, image: Image.Image) -> str:
        """
        计算图片哈希值
        
        Args:
            image: PIL图片对象
            
        Returns:
            str: 图片哈希值
        """
        # 将图片转换为字节流计算哈希
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        return hashlib.md5(img_bytes.getvalue()).hexdigest()
    
    def _get_cache_key(self, image_hash: str, processor_type: str, params: Dict[str, Any]) -> str:
        """
        生成缓存键
        
        Args:
            image_hash: 图片哈希值
            processor_type: 处理器类型
            params: 处理参数
            
        Returns:
            str: 缓存键
        """
        # 将参数排序后计算哈希
        sorted_params = json.dumps(params, sort_keys=True)
        params_hash = hashlib.md5(sorted_params.encode('utf-8')).hexdigest()
        return f"{image_hash}_{processor_type}_{params_hash}"
    
    def get_cached_result(self, image: Image.Image, processor_type: str, params: Dict[str, Any]) -> Optional[Image.Image]:
        """
        获取缓存的处理结果
        
        Args:
            image: 输入图片
            processor_type: 处理器类型
            params: 处理参数
            
        Returns:
            Image.Image or None: 缓存的处理结果，如果不存在则返回None
        """
        with self.lock:
            try:
                # 计算图片哈希
                image_hash = self._get_image_hash(image)
                
                # 生成缓存键
                cache_key = self._get_cache_key(image_hash, processor_type, params)
                
                # 检查是否存在缓存
                if cache_key not in self.cache_index:
                    return None
                
                # 检查缓存是否过期
                cache_entry = self.cache_index[cache_key]
                cached_time = datetime.fromisoformat(cache_entry['timestamp'])
                if datetime.now() - cached_time > timedelta(hours=self.max_age_hours):
                    # 缓存过期，删除缓存项
                    self._remove_cache_entry(cache_key)
                    return None
                
                # 检查缓存文件是否存在
                image_file = os.path.join(self.cache_dir, "images", f"{cache_key}.jpg")
                if not os.path.exists(image_file):
                    # 缓存文件不存在，删除缓存项
                    self._remove_cache_entry(cache_key)
                    return None
                
                # 加载缓存的图片
                cached_image = Image.open(image_file)
                print(f"✅ 从缓存加载处理结果: {cache_key}")
                return cached_image.copy()
                
            except Exception as e:
                print(f"⚠️ 缓存读取失败: {e}")
                return None
    
    def save_result(self, image: Image.Image, result_image: Image.Image, processor_type: str, params: Dict[str, Any]):
        """
        保存处理结果到缓存
        
        Args:
            image: 输入图片
            result_image: 处理结果图片
            processor_type: 处理器类型
            params: 处理参数
        """
        with self.lock:
            try:
                # 计算图片哈希
                image_hash = self._get_image_hash(image)
                
                # 生成缓存键
                cache_key = self._get_cache_key(image_hash, processor_type, params)
                
                # 保存图片文件
                image_file = os.path.join(self.cache_dir, "images", f"{cache_key}.jpg")
                result_image.save(image_file, format='JPEG', quality=95)
                
                # 更新索引
                self.cache_index[cache_key] = {
                    'image_hash': image_hash,
                    'processor_type': processor_type,
                    'params': params,
                    'timestamp': datetime.now().isoformat()
                }
                
                # 清理过期缓存
                self._cleanup_expired()
                
                # 控制缓存大小
                self._limit_cache_size()
                
                # 保存索引
                self._save_index()
                
                print(f"✅ 处理结果已缓存: {cache_key}")
                
            except Exception as e:
                print(f"⚠️ 缓存保存失败: {e}")
    
    def _remove_cache_entry(self, cache_key: str):
        """
        删除缓存项
        
        Args:
            cache_key: 缓存键
        """
        if cache_key in self.cache_index:
            # 删除图片文件
            image_file = os.path.join(self.cache_dir, "images", f"{cache_key}.jpg")
            if os.path.exists(image_file):
                os.remove(image_file)
            
            # 删除元数据文件（如果存在）
            metadata_file = os.path.join(self.cache_dir, "metadata", f"{cache_key}.json")
            if os.path.exists(metadata_file):
                os.remove(metadata_file)
            
            # 从索引中删除
            del self.cache_index[cache_key]
    
    def _cleanup_expired(self):
        """清理过期缓存"""
        current_time = datetime.now()
        expired_keys = []
        
        for cache_key, entry in self.cache_index.items():
            cached_time = datetime.fromisoformat(entry['timestamp'])
            if current_time - cached_time > timedelta(hours=self.max_age_hours):
                expired_keys.append(cache_key)
        
        for cache_key in expired_keys:
            self._remove_cache_entry(cache_key)
            print(f"🗑️ 清理过期缓存: {cache_key}")
    
    def _limit_cache_size(self):
        """限制缓存大小"""
        if len(self.cache_index) > self.max_cache_size:
            # 按时间排序，删除最旧的缓存项
            sorted_entries = sorted(self.cache_index.items(), 
                                  key=lambda x: datetime.fromisoformat(x[1]['timestamp']))
            
            # 删除超出限制的项
            excess_count = len(self.cache_index) - self.max_cache_size
            for i in range(excess_count):
                cache_key = sorted_entries[i][0]
                self._remove_cache_entry(cache_key)
                print(f"🗑️ 清理超量缓存: {cache_key}")
    
    def clear_cache(self):
        """清空所有缓存"""
        with self.lock:
            try:
                # 删除所有缓存文件
                import shutil
                if os.path.exists(self.cache_dir):
                    shutil.rmtree(self.cache_dir)
                
                # 重新创建目录
                os.makedirs(self.cache_dir, exist_ok=True)
                os.makedirs(os.path.join(self.cache_dir, "images"), exist_ok=True)
                os.makedirs(os.path.join(self.cache_dir, "metadata"), exist_ok=True)
                
                # 清空索引
                self.cache_index = {}
                self._save_index()
                
                print("🗑️ 缓存已清空")
                
            except Exception as e:
                print(f"⚠️ 缓存清空失败: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            Dict: 缓存统计信息
        """
        with self.lock:
            total_size = 0
            image_dir = os.path.join(self.cache_dir, "images")
            if os.path.exists(image_dir):
                for file in os.listdir(image_dir):
                    file_path = os.path.join(image_dir, file)
                    if os.path.isfile(file_path):
                        total_size += os.path.getsize(file_path)
            
            return {
                'cache_items': len(self.cache_index),
                'total_size_bytes': total_size,
                'max_cache_size': self.max_cache_size,
                'max_age_hours': self.max_age_hours
            }


# 创建全局缓存管理器实例
cache_manager = CacheManager()


def get_cache_stats():
    """获取缓存统计信息的便捷函数"""
    return cache_manager.get_cache_stats()


def clear_all_cache():
    """清空所有缓存的便捷函数"""
    cache_manager.clear_cache()