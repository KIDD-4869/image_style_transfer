#!/usr/bin/env python3
"""
优化的日志配置系统
"""

import logging
import logging.handlers
import os
import json
import time
from typing import Dict, Any, Optional
from datetime import datetime
import threading
from pathlib import Path

class StructuredFormatter(logging.Formatter):
    """结构化日志格式器"""
    
    def format(self, record: logging.LogRecord) -> str:
        # 创建结构化日志数据
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread_id': record.thread,
            'thread_name': record.threadName
        }
        
        # 添加异常信息
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # 添加额外字段
        if hasattr(record, 'extra_fields'):
            log_data.update(record.extra_fields)
        
        return json.dumps(log_data, ensure_ascii=False)

class PerformanceFilter(logging.Filter):
    """性能日志过滤器"""
    
    def filter(self, record: logging.LogRecord) -> bool:
        # 过滤掉过于频繁的日志
        if hasattr(record, 'rate_limit'):
            current_time = time.time()
            if not hasattr(self, '_last_log_time'):
                self._last_log_time = {}
            
            key = f"{record.name}:{record.funcName}:{record.lineno}"
            last_time = self._last_log_time.get(key, 0)
            
            if current_time - last_time < record.rate_limit:
                return False
            
            self._last_log_time[key] = current_time
        
        return True

class LoggingManager:
    """日志管理器"""
    
    def __init__(self, log_dir: str = "logs", max_file_size: int = 10*1024*1024, 
                 backup_count: int = 5):
        self.log_dir = Path(log_dir)
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self._setup_directories()
        self._configured = False
        self._lock = threading.Lock()
    
    def _setup_directories(self):
        """创建日志目录"""
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def configure_logging(self, level: str = "INFO", 
                         enable_console: bool = True,
                         enable_file: bool = True,
                         enable_structured: bool = True) -> None:
        """配置日志系统"""
        with self._lock:
            if self._configured:
                return
            
            # 获取根日志器
            root_logger = logging.getLogger()
            root_logger.setLevel(getattr(logging, level.upper()))
            
            # 清除现有处理器
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)
            
            # 控制台处理器
            if enable_console:
                console_handler = logging.StreamHandler()
                if enable_structured:
                    console_handler.setFormatter(StructuredFormatter())
                else:
                    console_handler.setFormatter(logging.Formatter(
                        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                    ))
                console_handler.addFilter(PerformanceFilter())
                root_logger.addHandler(console_handler)
            
            # 文件处理器
            if enable_file:
                # 应用日志
                app_log_file = self.log_dir / "app.log"
                app_handler = logging.handlers.RotatingFileHandler(
                    app_log_file, maxBytes=self.max_file_size, 
                    backupCount=self.backup_count, encoding='utf-8'
                )
                
                if enable_structured:
                    app_handler.setFormatter(StructuredFormatter())
                else:
                    app_handler.setFormatter(logging.Formatter(
                        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                    ))
                
                root_logger.addHandler(app_handler)
                
                # 错误日志
                error_log_file = self.log_dir / "error.log"
                error_handler = logging.handlers.RotatingFileHandler(
                    error_log_file, maxBytes=self.max_file_size,
                    backupCount=self.backup_count, encoding='utf-8'
                )
                error_handler.setLevel(logging.ERROR)
                error_handler.setFormatter(StructuredFormatter() if enable_structured 
                                         else logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(pathname)s:%(lineno)d'
                ))
                root_logger.addHandler(error_handler)
                
                # 性能日志
                perf_log_file = self.log_dir / "performance.log"
                perf_handler = logging.handlers.RotatingFileHandler(
                    perf_log_file, maxBytes=self.max_file_size,
                    backupCount=self.backup_count, encoding='utf-8'
                )
                perf_handler.addFilter(lambda record: hasattr(record, 'performance'))
                perf_handler.setFormatter(StructuredFormatter() if enable_structured 
                                        else logging.Formatter(
                    '%(asctime)s - %(message)s'
                ))
                root_logger.addHandler(perf_handler)
            
            self._configured = True
            logging.info("Logging system configured successfully")
    
    def get_logger(self, name: str) -> logging.Logger:
        """获取日志器"""
        return logging.getLogger(name)
    
    def log_performance(self, operation: str, duration: float, 
                       extra_data: Optional[Dict[str, Any]] = None):
        """记录性能日志"""
        logger = logging.getLogger('performance')
        
        perf_data = {
            'operation': operation,
            'duration_seconds': duration,
            'timestamp': datetime.now().isoformat()
        }
        
        if extra_data:
            perf_data.update(extra_data)
        
        # 创建日志记录
        record = logger.makeRecord(
            logger.name, logging.INFO, __file__, 0,
            f"Performance: {operation} took {duration:.3f}s",
            (), None
        )
        record.performance = True
        record.extra_fields = perf_data
        
        logger.handle(record)
    
    def get_log_stats(self) -> Dict[str, Any]:
        """获取日志统计信息"""
        stats = {
            'log_directory': str(self.log_dir),
            'log_files': [],
            'total_size_bytes': 0
        }
        
        if self.log_dir.exists():
            for log_file in self.log_dir.glob("*.log*"):
                file_size = log_file.stat().st_size
                stats['log_files'].append({
                    'name': log_file.name,
                    'size_bytes': file_size,
                    'size_mb': file_size / (1024 * 1024),
                    'modified': datetime.fromtimestamp(log_file.stat().st_mtime).isoformat()
                })
                stats['total_size_bytes'] += file_size
        
        stats['total_size_mb'] = stats['total_size_bytes'] / (1024 * 1024)
        return stats

class PerformanceLogger:
    """性能日志记录器"""
    
    def __init__(self, logger_name: str = 'performance'):
        self.logger = logging.getLogger(logger_name)
        self.logging_manager = logging_manager
    
    def __call__(self, operation_name: Optional[str] = None):
        """装饰器用法"""
        def decorator(func):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    self.logging_manager.log_performance(
                        op_name, duration, 
                        {'status': 'success', 'args_count': len(args)}
                    )
                    
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    
                    self.logging_manager.log_performance(
                        op_name, duration,
                        {'status': 'error', 'error': str(e), 'args_count': len(args)}
                    )
                    
                    raise
            
            return wrapper
        return decorator
    
    def log_timing(self, operation: str, duration: float, **extra):
        """直接记录时间"""
        self.logging_manager.log_performance(operation, duration, extra)

# 全局日志管理器
logging_manager = LoggingManager()

# 性能日志记录器
perf_logger = PerformanceLogger()

def setup_logging(level: str = "INFO", enable_console: bool = True,
                 enable_file: bool = True, enable_structured: bool = True):
    """设置日志系统的便捷函数"""
    logging_manager.configure_logging(level, enable_console, enable_file, enable_structured)