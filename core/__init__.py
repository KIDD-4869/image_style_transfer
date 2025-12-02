"""
核心处理模块初始化文件 - 重构后的简化版本
"""

from enum import Enum

# 导入新的处理器架构
try:
    from .processors import GhibliProcessor, ProcessingStrategy, ProcessingResult
    print("✅ 新处理器架构加载成功")
except ImportError as e:
    print(f"⚠️ 处理器加载失败: {e}")
    GhibliProcessor = None
    ProcessingStrategy = None
    ProcessingResult = None

class ProcessorType(Enum):
    """处理器类型枚举"""
    GHIBLI = "ghibli"
    FAST = "fast"
    BALANCED = "balanced"
    QUALITY = "quality"

__all__ = [
    'GhibliProcessor',
    'ProcessingStrategy', 
    'ProcessingResult',
    'ProcessorType'
]