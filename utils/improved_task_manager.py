#!/usr/bin/env python3
"""
改进的任务管理器 - 带自动清理和内存管理
"""

import time
import threading
from collections import OrderedDict
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskInfo:
    """任务信息类"""
    
    def __init__(self, task_id: str, task_type: str, metadata: Optional[Dict[str, Any]] = None):
        self.task_id = task_id
        self.task_type = task_type
        self.status = TaskStatus.PENDING
        self.progress = 0
        self.current_step = 0
        self.total_steps = 10
        self.result = None
        self.error_message = None
        self.created_at = datetime.now()
        self.started_at = None
        self.completed_at = None
        self.metadata = metadata or {}
        
        # 新增：缓存和优化相关字段
        self.cache_hit = False
        self.optimization_status = None  # None, 'pending', 'running', 'completed', 'failed'
        self.optimization_progress = 0
        self.has_optimized_version = False
    
    def mark_cache_hit(self):
        """标记任务为缓存命中"""
        self.cache_hit = True
        self.metadata['cache_hit'] = True
        self.metadata['cache_hit_at'] = datetime.now().isoformat()
        logger.info(f"任务 {self.task_id} 标记为缓存命中")
    
    def start_optimization(self):
        """开始后台优化"""
        self.optimization_status = 'running'
        self.metadata['optimization_started_at'] = datetime.now().isoformat()
        logger.info(f"任务 {self.task_id} 开始后台优化")
    
    def update_optimization_progress(self, progress: int):
        """更新优化进度"""
        self.optimization_progress = progress
        self.metadata['optimization_progress'] = progress
    
    def complete_optimization(self):
        """完成后台优化"""
        self.optimization_status = 'completed'
        self.has_optimized_version = True
        self.optimization_progress = 100
        self.metadata['optimization_completed_at'] = datetime.now().isoformat()
        logger.info(f"任务 {self.task_id} 后台优化完成")
    
    def fail_optimization(self, error_message: str):
        """优化失败"""
        self.optimization_status = 'failed'
        self.metadata['optimization_error'] = error_message
        self.metadata['optimization_failed_at'] = datetime.now().isoformat()
        logger.warning(f"任务 {self.task_id} 优化失败: {error_message}")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'task_id': self.task_id,
            'task_type': self.task_type,
            'status': self.status.value,
            'progress': self.progress,
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'error_message': self.error_message,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'metadata': self.metadata,
            # 新增字段
            'cache_hit': self.cache_hit,
            'optimization_status': self.optimization_status,
            'optimization_progress': self.optimization_progress,
            'has_optimized_version': self.has_optimized_version
        }


class ImprovedTaskManager:
    """改进的任务管理器 - 带自动清理功能"""
    
    def __init__(self, max_tasks: int = 100, ttl_hours: int = 24, cleanup_interval: int = 300):
        """
        初始化任务管理器
        
        Args:
            max_tasks: 最大任务数量
            ttl_hours: 任务生存时间（小时）
            cleanup_interval: 清理间隔（秒）
        """
        self.tasks = OrderedDict()
        self.max_tasks = max_tasks
        self.ttl = timedelta(hours=ttl_hours)
        self.cleanup_interval = cleanup_interval
        self._lock = threading.Lock()
        self._cleanup_thread = None
        self._stop_cleanup = threading.Event()
        
        # 启动清理线程
        self._start_cleanup_thread()
        
        logger.info(f"任务管理器初始化: max_tasks={max_tasks}, ttl={ttl_hours}h, cleanup_interval={cleanup_interval}s")
    
    def _start_cleanup_thread(self):
        """启动清理线程"""
        if self._cleanup_thread is None or not self._cleanup_thread.is_alive():
            self._stop_cleanup.clear()
            self._cleanup_thread = threading.Thread(
                target=self._cleanup_loop,
                daemon=True,
                name="TaskCleanupThread"
            )
            self._cleanup_thread.start()
            logger.info("清理线程已启动")
    
    def _cleanup_loop(self):
        """清理循环"""
        while not self._stop_cleanup.is_set():
            try:
                time.sleep(self.cleanup_interval)
                cleaned = self.cleanup_old_tasks()
                if cleaned > 0:
                    logger.info(f"清理了 {cleaned} 个过期任务")
            except Exception as e:
                logger.error(f"清理任务时出错: {e}")
    
    def cleanup_old_tasks(self) -> int:
        """
        清理过期任务
        
        Returns:
            清理的任务数量
        """
        with self._lock:
            now = datetime.now()
            expired_ids = [
                task_id for task_id, task in self.tasks.items()
                if now - task.created_at > self.ttl
            ]
            
            for task_id in expired_ids:
                del self.tasks[task_id]
            
            return len(expired_ids)
    
    def create_task(self, task_id: str, task_type: str, metadata: Optional[Dict[str, Any]] = None):
        """
        创建新任务
        
        Args:
            task_id: 任务ID
            task_type: 任务类型
            metadata: 任务元数据
        """
        with self._lock:
            # 如果达到最大任务数，删除最旧的任务
            if len(self.tasks) >= self.max_tasks:
                oldest_id = next(iter(self.tasks))
                del self.tasks[oldest_id]
                logger.warning(f"达到最大任务数，删除最旧任务: {oldest_id}")
            
            task = TaskInfo(task_id, task_type, metadata)
            self.tasks[task_id] = task
            logger.info(f"创建任务: {task_id}, 类型: {task_type}")
    
    def get_task(self, task_id: str) -> Optional[TaskInfo]:
        """
        获取任务信息
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务信息，如果不存在返回None
        """
        with self._lock:
            return self.tasks.get(task_id)
    
    def set_task_status(self, task_id: str, status: TaskStatus):
        """
        设置任务状态
        
        Args:
            task_id: 任务ID
            status: 新状态
        """
        with self._lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.status = status
                
                if status == TaskStatus.PROCESSING and task.started_at is None:
                    task.started_at = datetime.now()
                elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                    task.completed_at = datetime.now()
                
                logger.debug(f"任务 {task_id} 状态更新: {status.value}")
    
    def update_task_progress(self, task_id: str, progress: int, current_step: int, total_steps: int, loss: float = 0):
        """
        更新任务进度
        
        Args:
            task_id: 任务ID
            progress: 进度百分比
            current_step: 当前步骤
            total_steps: 总步骤数
            loss: 损失值（可选）
        """
        with self._lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.progress = progress
                task.current_step = current_step
                task.total_steps = total_steps
                
                if loss > 0:
                    task.metadata['loss'] = loss
    
    def set_task_result(self, task_id: str, result: Any):
        """
        设置任务结果
        
        Args:
            task_id: 任务ID
            result: 任务结果
        """
        with self._lock:
            if task_id in self.tasks:
                self.tasks[task_id].result = result
                logger.info(f"任务 {task_id} 结果已设置")
    
    def set_task_error(self, task_id: str, error_message: str):
        """
        设置任务错误
        
        Args:
            task_id: 任务ID
            error_message: 错误信息
        """
        with self._lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.error_message = error_message
                task.status = TaskStatus.FAILED
                logger.error(f"任务 {task_id} 失败: {error_message}")
    
    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """
        获取所有任务
        
        Returns:
            任务列表
        """
        with self._lock:
            return [task.to_dict() for task in self.tasks.values()]
    
    def get_active_tasks(self) -> List[Dict[str, Any]]:
        """
        获取活跃任务（待处理和处理中）
        
        Returns:
            活跃任务列表
        """
        with self._lock:
            return [
                task.to_dict() for task in self.tasks.values()
                if task.status in [TaskStatus.PENDING, TaskStatus.PROCESSING]
            ]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        with self._lock:
            total = len(self.tasks)
            status_counts = {}
            
            for task in self.tasks.values():
                status = task.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
            
            return {
                'total_tasks': total,
                'max_tasks': self.max_tasks,
                'status_counts': status_counts,
                'ttl_hours': self.ttl.total_seconds() / 3600
            }
    
    def stop(self):
        """停止任务管理器"""
        logger.info("停止任务管理器...")
        self._stop_cleanup.set()
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5)
        logger.info("任务管理器已停止")


# 全局任务管理器实例
task_manager = ImprovedTaskManager(max_tasks=100, ttl_hours=24, cleanup_interval=300)
