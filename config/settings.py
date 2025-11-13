"""
项目配置文件
"""

class Config:
    # Flask配置
    SECRET_KEY = 'ghibli-style-transfer-secret-key'
    MAX_CONTENT_LENGTH = 20 * 1024 * 1024  # 20MB max file size
    UPLOAD_FOLDER = 'static/uploads'
    
    # 图像处理参数
    MAX_IMAGE_SIZE = 0  # 0表示不限制图片尺寸，支持任意尺寸
    DEFAULT_QUALITY = 95
    
    # 宫崎骏风格参数
    GHIBLI_PARAMETERS = {
        'color_saturation': 1.3,
        'edge_detection': 'adaptive',
        'lighting_effect': 'dreamy',
        'color_enhancement': 1.2,
        'edge_softness': 0.8,
        'warm_tone': 1.1
    }
    
    # 处理队列配置
    MAX_QUEUE_SIZE = 100
    PROCESS_TIMEOUT = 300  # 5分钟超时
    
    # 缓存配置
    CACHE_TYPE = 'optimized'
    CACHE_DEFAULT_TIMEOUT = 300
    CACHE_MAX_SIZE_MB = 1024
    CACHE_MAX_AGE_HOURS = 24
    CACHE_COMPRESSION_LEVEL = 3
    
    # 性能配置
    PERFORMANCE = {
        'max_workers': None,  # 自动检测
        'use_process_pool': False,
        'memory_cleanup_threshold': 0.7,
        'max_memory_cache_mb': 256,
        'enable_gpu_acceleration': True
    }
    
    # 监控配置
    MONITORING = {
        'enable_metrics': True,
        'metrics_interval': 60,  # 秒
        'log_level': 'INFO',
        'enable_health_check': True
    }
    
class DevelopmentConfig(Config):
    DEBUG = True
    
class ProductionConfig(Config):
    DEBUG = False
    
# 配置字典
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}