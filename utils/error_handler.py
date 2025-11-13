#!/usr/bin/env python3
"""
标准化错误处理系统
"""

import logging
import traceback
import functools
from typing import Any, Callable, Dict, Optional, Type, Union
from enum import Enum
import time

logger = logging.getLogger(__name__)

class ErrorCode(Enum):
    """错误代码枚举"""
    # 系统错误
    SYSTEM_ERROR = "SYSTEM_ERROR"
    MEMORY_ERROR = "MEMORY_ERROR"
    TIMEOUT_ERROR = "TIMEOUT_ERROR"
    
    # 输入错误
    INVALID_INPUT = "INVALID_INPUT"
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    INVALID_FORMAT = "INVALID_FORMAT"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    
    # 处理错误
    PROCESSING_ERROR = "PROCESSING_ERROR"
    MODEL_LOAD_ERROR = "MODEL_LOAD_ERROR"
    CACHE_ERROR = "CACHE_ERROR"
    
    # 网络错误
    NETWORK_ERROR = "NETWORK_ERROR"
    API_ERROR = "API_ERROR"

class ProcessingError(Exception):
    """处理错误基类"""
    
    def __init__(self, message: str, error_code: ErrorCode, 
                 details: Optional[Dict[str, Any]] = None, 
                 original_exception: Optional[Exception] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.original_exception = original_exception
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'error_code': self.error_code.value,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp,
            'original_error': str(self.original_exception) if self.original_exception else None
        }

class ErrorHandler:
    """错误处理器"""
    
    def __init__(self):
        self.error_stats: Dict[str, int] = {}
        self.error_history: list = []
        self.max_history = 1000
    
    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> ProcessingError:
        """处理错误并转换为标准格式"""
        context = context or {}
        
        # 根据异常类型确定错误代码
        if isinstance(error, ProcessingError):
            processing_error = error
        elif isinstance(error, MemoryError):
            processing_error = ProcessingError(
                "内存不足，请尝试处理较小的图片",
                ErrorCode.MEMORY_ERROR,
                context,
                error
            )
        elif isinstance(error, TimeoutError):
            processing_error = ProcessingError(
                "处理超时，请稍后重试",
                ErrorCode.TIMEOUT_ERROR,
                context,
                error
            )
        elif isinstance(error, FileNotFoundError):
            processing_error = ProcessingError(
                "文件未找到",
                ErrorCode.FILE_NOT_FOUND,
                context,
                error
            )
        elif isinstance(error, ValueError):
            processing_error = ProcessingError(
                "输入参数无效",
                ErrorCode.INVALID_INPUT,
                context,
                error
            )
        else:
            processing_error = ProcessingError(
                f"系统错误: {str(error)}",
                ErrorCode.SYSTEM_ERROR,
                context,
                error
            )
        
        # 记录错误统计
        error_code = processing_error.error_code.value
        self.error_stats[error_code] = self.error_stats.get(error_code, 0) + 1
        
        # 记录错误历史
        self.error_history.append({
            'error': processing_error.to_dict(),
            'context': context,
            'traceback': traceback.format_exc()
        })
        
        # 限制历史记录大小
        if len(self.error_history) > self.max_history:
            self.error_history = self.error_history[-self.max_history:]
        
        # 记录日志
        logger.error(f"Error handled: {processing_error.error_code.value} - {processing_error.message}")
        if processing_error.original_exception:
            logger.error(f"Original exception: {processing_error.original_exception}")
        
        return processing_error
    
    def get_error_stats(self) -> Dict[str, Any]:
        """获取错误统计信息"""
        return {
            'error_counts': self.error_stats.copy(),
            'total_errors': sum(self.error_stats.values()),
            'recent_errors': len([
                e for e in self.error_history 
                if time.time() - e['error']['timestamp'] < 3600  # 最近1小时
            ])
        }

# 全局错误处理器
error_handler = ErrorHandler()

def handle_errors(error_context: Optional[Dict[str, Any]] = None):
    """错误处理装饰器"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = error_context or {}
                context.update({
                    'function': func.__name__,
                    'args_count': len(args),
                    'kwargs_keys': list(kwargs.keys())
                })
                
                processing_error = error_handler.handle_error(e, context)
                raise processing_error
        
        return wrapper
    return decorator

def safe_execute(func: Callable, *args, default_return=None, 
                error_context: Optional[Dict[str, Any]] = None, **kwargs):
    """安全执行函数，捕获异常并返回默认值"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        context = error_context or {}
        context.update({
            'function': func.__name__ if hasattr(func, '__name__') else str(func),
            'safe_execution': True
        })
        
        processing_error = error_handler.handle_error(e, context)
        logger.warning(f"Safe execution failed, returning default: {processing_error.message}")
        return default_return

class RetryHandler:
    """重试处理器"""
    
    @staticmethod
    def retry_on_error(max_retries: int = 3, delay: float = 1.0, 
                      backoff_factor: float = 2.0,
                      retry_on: tuple = (Exception,)):
        """重试装饰器"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                last_exception = None
                current_delay = delay
                
                for attempt in range(max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except retry_on as e:
                        last_exception = e
                        
                        if attempt < max_retries:
                            logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
                            logger.warning(f"Retrying in {current_delay} seconds...")
                            time.sleep(current_delay)
                            current_delay *= backoff_factor
                        else:
                            logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}")
                            break
                
                # 所有重试都失败了，抛出最后的异常
                if last_exception:
                    raise last_exception
            
            return wrapper
        return decorator