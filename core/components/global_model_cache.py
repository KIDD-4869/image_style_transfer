"""
Global Model Cache - 全局模型缓存
避免每次处理都重新加载模型
"""
import logging
from typing import Optional
from core.components.model_manager import ModelManager
from core.models import ProcessorConfig

logger = logging.getLogger(__name__)

# 全局模型缓存
_global_model_manager: Optional[ModelManager] = None
_global_config_hash: Optional[str] = None


def get_config_hash(config: ProcessorConfig) -> str:
    """
    生成配置的哈希值，用于判断配置是否改变
    
    Args:
        config: 处理器配置
    
    Returns:
        配置哈希字符串
    """
    import hashlib
    config_str = f"{config.base_model}_{config.device}_{config.dtype}_{config.use_controlnet}"
    return hashlib.md5(config_str.encode()).hexdigest()


def get_global_model_manager(
    config: ProcessorConfig,
    logger_instance: logging.Logger = None,
    progress_callback=None,
    force_reload: bool = False
) -> ModelManager:
    """
    获取全局模型管理器（单例模式）
    
    Args:
        config: 处理器配置
        logger_instance: 日志实例
        progress_callback: 进度回调
        force_reload: 是否强制重新加载
    
    Returns:
        ModelManager 实例
    """
    global _global_model_manager, _global_config_hash
    
    current_hash = get_config_hash(config)
    
    # 如果配置改变或强制重新加载，清除缓存
    if force_reload or (_global_config_hash and _global_config_hash != current_hash):
        logger.info("配置已改变或强制重新加载，清除模型缓存")
        _global_model_manager = None
        _global_config_hash = None
    
    # 如果已有缓存且配置未改变，直接返回
    if _global_model_manager is not None:
        logger.info("✅ 使用缓存的模型（无需重新加载）")
        if progress_callback:
            progress_callback(15, "使用缓存的模型（秒级加载）")
        return _global_model_manager
    
    # 首次加载或配置改变，创建新的模型管理器
    logger.info("🔄 首次加载模型或配置已改变，开始加载...")
    if progress_callback:
        progress_callback(0, "首次加载模型，请稍候...")
    
    _global_model_manager = ModelManager(config, logger_instance, progress_callback)
    _global_config_hash = current_hash
    
    logger.info("✅ 模型已加载并缓存到内存")
    
    return _global_model_manager


def clear_global_model_cache():
    """清除全局模型缓存"""
    global _global_model_manager, _global_config_hash
    
    logger.info("清除全局模型缓存")
    _global_model_manager = None
    _global_config_hash = None


def is_model_cached() -> bool:
    """检查模型是否已缓存"""
    return _global_model_manager is not None
