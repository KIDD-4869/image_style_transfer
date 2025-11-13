#!/usr/bin/env python3
"""
系统健康检查模块
"""

import time
import psutil
import torch
import threading
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """健康状态枚举"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

@dataclass
class HealthCheckResult:
    """健康检查结果"""
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any]
    timestamp: float
    duration_ms: float

class HealthChecker:
    """健康检查器基类"""
    
    def __init__(self, name: str, timeout: float = 5.0):
        self.name = name
        self.timeout = timeout
    
    def check(self) -> HealthCheckResult:
        """执行健康检查"""
        start_time = time.time()
        
        try:
            status, message, details = self._perform_check()
            duration_ms = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                name=self.name,
                status=status,
                message=message,
                details=details,
                timestamp=time.time(),
                duration_ms=duration_ms
            )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Health check {self.name} failed: {e}")
            
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.CRITICAL,
                message=f"Health check failed: {str(e)}",
                details={'error': str(e)},
                timestamp=time.time(),
                duration_ms=duration_ms
            )
    
    def _perform_check(self) -> tuple[HealthStatus, str, Dict[str, Any]]:
        """子类需要实现的检查逻辑"""
        raise NotImplementedError

class SystemResourceChecker(HealthChecker):
    """系统资源检查器"""
    
    def __init__(self, cpu_threshold: float = 80.0, memory_threshold: float = 85.0,
                 disk_threshold: float = 90.0):
        super().__init__("system_resources")
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.disk_threshold = disk_threshold
    
    def _perform_check(self) -> tuple[HealthStatus, str, Dict[str, Any]]:
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # 内存使用率
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # 磁盘使用率
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        
        details = {
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'memory_available_gb': memory.available / (1024**3),
            'disk_percent': disk_percent,
            'disk_free_gb': disk.free / (1024**3)
        }
        
        # 判断状态
        if (cpu_percent > self.cpu_threshold or 
            memory_percent > self.memory_threshold or 
            disk_percent > self.disk_threshold):
            
            status = HealthStatus.CRITICAL
            message = "System resources critical"
        elif (cpu_percent > self.cpu_threshold * 0.8 or 
              memory_percent > self.memory_threshold * 0.8 or 
              disk_percent > self.disk_threshold * 0.8):
            
            status = HealthStatus.WARNING
            message = "System resources warning"
        else:
            status = HealthStatus.HEALTHY
            message = "System resources normal"
        
        return status, message, details

class GPUChecker(HealthChecker):
    """GPU检查器"""
    
    def __init__(self):
        super().__init__("gpu")
    
    def _perform_check(self) -> tuple[HealthStatus, str, Dict[str, Any]]:
        details = {
            'cuda_available': torch.cuda.is_available(),
            'device_count': 0,
            'devices': []
        }
        
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            details['device_count'] = device_count
            
            for i in range(device_count):
                device_props = torch.cuda.get_device_properties(i)
                memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)
                memory_total = device_props.total_memory / (1024**3)
                
                details['devices'].append({
                    'id': i,
                    'name': device_props.name,
                    'memory_allocated_gb': memory_allocated,
                    'memory_reserved_gb': memory_reserved,
                    'memory_total_gb': memory_total,
                    'memory_usage_percent': (memory_reserved / memory_total) * 100
                })
            
            status = HealthStatus.HEALTHY
            message = f"GPU available with {device_count} device(s)"
        else:
            status = HealthStatus.WARNING
            message = "GPU not available, using CPU"
        
        return status, message, details

class ModelChecker(HealthChecker):
    """模型检查器"""
    
    def __init__(self, model_manager):
        super().__init__("models")
        self.model_manager = model_manager
    
    def _perform_check(self) -> tuple[HealthStatus, str, Dict[str, Any]]:
        try:
            # 检查模型管理器状态
            memory_info = self.model_manager.get_memory_info()
            
            details = {
                'loaded_models': memory_info['loaded_models'],
                'system_memory': memory_info['system_memory'],
                'gpu_memory': memory_info['gpu_memory']
            }
            
            # 检查内存使用情况
            memory_percent = memory_info['system_memory']['percent']
            
            if memory_percent > 90:
                status = HealthStatus.CRITICAL
                message = "High memory usage"
            elif memory_percent > 75:
                status = HealthStatus.WARNING
                message = "Moderate memory usage"
            else:
                status = HealthStatus.HEALTHY
                message = "Models loaded successfully"
            
            return status, message, details
            
        except Exception as e:
            return HealthStatus.CRITICAL, f"Model check failed: {str(e)}", {'error': str(e)}

class CacheChecker(HealthChecker):
    """缓存检查器"""
    
    def __init__(self, cache_manager):
        super().__init__("cache")
        self.cache_manager = cache_manager
    
    def _perform_check(self) -> tuple[HealthStatus, str, Dict[str, Any]]:
        try:
            stats = self.cache_manager.get_cache_stats()
            
            details = {
                'cache_entries': stats.get('disk_cache_entries', 0),
                'cache_size_mb': stats.get('disk_cache_size_mb', 0),
                'memory_cache_entries': stats.get('memory_cache_entries', 0),
                'memory_cache_size_mb': stats.get('memory_cache_size_mb', 0)
            }
            
            # 检查缓存大小
            cache_size_mb = stats.get('disk_cache_size_mb', 0)
            max_size_mb = stats.get('max_cache_size_mb', 1024)
            
            usage_percent = (cache_size_mb / max_size_mb) * 100 if max_size_mb > 0 else 0
            
            if usage_percent > 95:
                status = HealthStatus.WARNING
                message = "Cache nearly full"
            else:
                status = HealthStatus.HEALTHY
                message = "Cache operating normally"
            
            details['usage_percent'] = usage_percent
            
            return status, message, details
            
        except Exception as e:
            return HealthStatus.CRITICAL, f"Cache check failed: {str(e)}", {'error': str(e)}

class HealthMonitor:
    """健康监控器"""
    
    def __init__(self):
        self.checkers: List[HealthChecker] = []
        self.check_history: List[Dict[str, Any]] = []
        self.max_history = 100
        self._lock = threading.Lock()
    
    def add_checker(self, checker: HealthChecker):
        """添加健康检查器"""
        with self._lock:
            self.checkers.append(checker)
            logger.info(f"Added health checker: {checker.name}")
    
    def run_checks(self) -> Dict[str, Any]:
        """运行所有健康检查"""
        start_time = time.time()
        results = []
        overall_status = HealthStatus.HEALTHY
        
        for checker in self.checkers:
            result = checker.check()
            results.append({
                'name': result.name,
                'status': result.status.value,
                'message': result.message,
                'details': result.details,
                'duration_ms': result.duration_ms
            })
            
            # 更新整体状态
            if result.status == HealthStatus.CRITICAL:
                overall_status = HealthStatus.CRITICAL
            elif result.status == HealthStatus.WARNING and overall_status == HealthStatus.HEALTHY:
                overall_status = HealthStatus.WARNING
        
        total_duration = (time.time() - start_time) * 1000
        
        health_report = {
            'overall_status': overall_status.value,
            'timestamp': datetime.now().isoformat(),
            'total_duration_ms': total_duration,
            'checks': results
        }
        
        # 保存到历史记录
        with self._lock:
            self.check_history.append(health_report)
            if len(self.check_history) > self.max_history:
                self.check_history = self.check_history[-self.max_history:]
        
        return health_report
    
    def get_health_summary(self) -> Dict[str, Any]:
        """获取健康状态摘要"""
        if not self.check_history:
            return {'status': 'no_data', 'message': 'No health checks performed yet'}
        
        latest = self.check_history[-1]
        
        # 统计最近的检查结果
        recent_checks = self.check_history[-10:] if len(self.check_history) >= 10 else self.check_history
        
        status_counts = {}
        for check in recent_checks:
            status = check['overall_status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            'current_status': latest['overall_status'],
            'last_check': latest['timestamp'],
            'recent_status_distribution': status_counts,
            'total_checks_performed': len(self.check_history),
            'checkers_count': len(self.checkers)
        }

# 全局健康监控器
health_monitor = HealthMonitor()

def setup_health_monitoring(model_manager=None, cache_manager=None):
    """设置健康监控"""
    # 添加系统资源检查器
    health_monitor.add_checker(SystemResourceChecker())
    
    # 添加GPU检查器
    health_monitor.add_checker(GPUChecker())
    
    # 添加模型检查器
    if model_manager:
        health_monitor.add_checker(ModelChecker(model_manager))
    
    # 添加缓存检查器
    if cache_manager:
        health_monitor.add_checker(CacheChecker(cache_manager))
    
    logger.info("Health monitoring setup completed")