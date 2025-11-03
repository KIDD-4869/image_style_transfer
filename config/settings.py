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
    CACHE_TYPE = 'simple'
    CACHE_DEFAULT_TIMEOUT = 300
    
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