"""
核心处理器模块
"""

from .base import BaseProcessor, ProcessingStrategy, ProcessingResult
from .ghibli import GhibliProcessor

__all__ = [
    'BaseProcessor',
    'ProcessingStrategy',
    'ProcessingResult',
    'GhibliProcessor'
]
