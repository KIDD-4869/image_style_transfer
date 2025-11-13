"""
核心处理模块初始化文件
"""

from enum import Enum
from .image_processor_interface import ImageProcessorInterface, ImageProcessorFactory, ProcessingStyle
from .real_ghibli_transfer import RealGhibliStyleTransfer
from .ghibli_enhanced import GhibliEnhancedTransfer
from .true_ghibli_processor import TrueGhibliProcessor
from .anime_style_processor import AnimeStyleProcessor
from .perfect_ghibli_processor import PerfectGhibliProcessor

# 注册处理器
ImageProcessorFactory.register_processor(ProcessingStyle.GHIBLI_CLASSIC, RealGhibliStyleTransfer)
ImageProcessorFactory.register_processor(ProcessingStyle.GHIBLI_ENHANCED, PerfectGhibliProcessor)
ImageProcessorFactory.register_processor(ProcessingStyle.GHIBLI_NEURAL, GhibliEnhancedTransfer)

class ProcessorType(Enum):
    """处理器类型枚举"""
    CLASSIC = "classic"
    ENHANCED = "enhanced"
    NEURAL = "neural"