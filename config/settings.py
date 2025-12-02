#!/usr/bin/env python3
"""
应用配置管理 - 统一配置接口
"""

import os
from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class AppConfig:
    """应用配置"""
    
    # 应用基本信息
    app_name: str = "Ghibli Style Transfer"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # 服务器配置
    host: str = "0.0.0.0"
    port: int = 5003
    
    # 上传配置
    upload_folder: str = "static/uploads"
    max_content_length: int = 20 * 1024 * 1024  # 20MB
    max_image_size: int = 2048
    allowed_extensions: set = field(default_factory=lambda: {'jpg', 'jpeg', 'png', 'bmp', 'gif'})
    
    # 处理配置
    max_concurrent_tasks: int = 3
    default_strategy: str = "balanced"
    
    # 任务管理配置
    task_max_tasks: int = 100
    task_ttl_hours: int = 24
    task_cleanup_interval: int = 300  # 秒
    
    # 缓存配置
    cache_enabled: bool = True
    cache_memory_size_mb: int = 100
    cache_disk_size_mb: int = 1000
    cache_dir: str = "./cache"
    cache_ttl_hours: int = 24
    cache_enable_disk: bool = True
    
    def __post_init__(self):
        """初始化后验证配置"""
        self._load_from_env()
        self._validate()
    
    def _load_from_env(self):
        """从环境变量加载配置"""
        # 应用配置
        self.debug = os.getenv('DEBUG', str(self.debug)).lower() in ('true', '1', 'yes')
        self.host = os.getenv('HOST', self.host)
        self.port = int(os.getenv('PORT', str(self.port)))
        
        # 上传配置
        self.upload_folder = os.getenv('UPLOAD_FOLDER', self.upload_folder)
        self.max_content_length = int(os.getenv('MAX_CONTENT_LENGTH', str(self.max_content_length)))
        self.max_image_size = int(os.getenv('MAX_IMAGE_SIZE', str(self.max_image_size)))
        
        # 处理配置
        self.max_concurrent_tasks = int(os.getenv('MAX_CONCURRENT_TASKS', str(self.max_concurrent_tasks)))
        self.default_strategy = os.getenv('DEFAULT_STRATEGY', self.default_strategy)
        
        # 任务管理配置
        self.task_max_tasks = int(os.getenv('TASK_MAX_TASKS', str(self.task_max_tasks)))
        self.task_ttl_hours = int(os.getenv('TASK_TTL_HOURS', str(self.task_ttl_hours)))
        
        # 缓存配置
        self.cache_enabled = os.getenv('CACHE_ENABLED', str(self.cache_enabled)).lower() in ('true', '1', 'yes')
        self.cache_memory_size_mb = int(os.getenv('CACHE_MEMORY_SIZE', str(self.cache_memory_size_mb)))
        self.cache_disk_size_mb = int(os.getenv('CACHE_DISK_SIZE', str(self.cache_disk_size_mb)))
        self.cache_dir = os.getenv('CACHE_DIR', self.cache_dir)
        self.cache_ttl_hours = int(os.getenv('CACHE_TTL_HOURS', str(self.cache_ttl_hours)))
        self.cache_enable_disk = os.getenv('CACHE_ENABLE_DISK', str(self.cache_enable_disk)).lower() in ('true', '1', 'yes')
    
    def _validate(self):
        """验证配置参数"""
        # 验证端口
        if not (1024 <= self.port <= 65535):
            logger.warning(f"端口 {self.port} 不在推荐范围内 (1024-65535)")
        
        # 验证图像大小
        if self.max_image_size < 256:
            raise ValueError("max_image_size 必须 >= 256")
        if self.max_image_size > 4096:
            logger.warning(f"max_image_size {self.max_image_size} 可能导致性能问题")
        
        # 验证并发任务数
        if self.max_concurrent_tasks < 1:
            raise ValueError("max_concurrent_tasks 必须 >= 1")
        if self.max_concurrent_tasks > 10:
            logger.warning(f"max_concurrent_tasks {self.max_concurrent_tasks} 可能导致资源耗尽")
        
        # 验证缓存配置
        if self.cache_memory_size_mb < 10:
            logger.warning("cache_memory_size_mb < 10MB 可能导致缓存效果不佳")
        if self.cache_disk_size_mb < 100:
            logger.warning("cache_disk_size_mb < 100MB 可能导致缓存效果不佳")
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'app_name': self.app_name,
            'app_version': self.app_version,
            'debug': self.debug,
            'host': self.host,
            'port': self.port,
            'upload_folder': self.upload_folder,
            'max_content_length': self.max_content_length,
            'max_image_size': self.max_image_size,
            'max_concurrent_tasks': self.max_concurrent_tasks,
            'default_strategy': self.default_strategy,
            'task_max_tasks': self.task_max_tasks,
            'task_ttl_hours': self.task_ttl_hours,
            'cache_enabled': self.cache_enabled,
            'cache_memory_size_mb': self.cache_memory_size_mb,
            'cache_disk_size_mb': self.cache_disk_size_mb,
            'cache_dir': self.cache_dir,
            'cache_ttl_hours': self.cache_ttl_hours,
            'cache_enable_disk': self.cache_enable_disk
        }
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"AppConfig(app={self.app_name} v{self.app_version}, port={self.port}, cache={self.cache_enabled})"


# 全局配置实例
_global_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """获取全局配置实例（单例模式）"""
    global _global_config
    if _global_config is None:
        _global_config = AppConfig()
        logger.info(f"配置已加载: {_global_config}")
    return _global_config


def reload_config() -> AppConfig:
    """重新加载配置"""
    global _global_config
    _global_config = AppConfig()
    logger.info(f"配置已重新加载: {_global_config}")
    return _global_config


# 为了向后兼容，保留旧的config字典
config = {
    'default': type('Config', (), {
        'UPLOAD_FOLDER': 'static/uploads',
        'MAX_CONTENT_LENGTH': 20 * 1024 * 1024,
        'MAX_IMAGE_SIZE': 2048
    })()
}
