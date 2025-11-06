#!/usr/bin/env python3
"""
任务管理器
用于管理异步处理任务的状态和生命周期
"""

import threading
import time
from typing import Dict, Any, Optional
from enum import Enum
from datetime import datetime, timedelta
import logging

# 配置日志
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
    
    def __init__(self, task_id: str, task_type: str, data: Dict[str, Any] = None):
        self.task_id = task_id
        self.task_type = task_type
        self.status = TaskStatus.PENDING
        self.data = data or {}
        self.progress = 0
        self.current_step = 0
        self.total_steps = 100
        self.loss = 0.0
        self.result = None
        self.error_message = ""
        self.created_at = datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.lock = threading.Lock()
    
    def update_progress(self, progress: int, current_step: int = None, total_steps: int = None, loss: float = None):
        """
        更新任务进度
        
        Args:
            progress: 进度百分比 (0-100)
            current_step: 当前步骤
            total_steps: 总步骤数
            loss: 损失值
        """
        with self.lock:
            self.progress = max(0, min(100, progress))
            if current_step is not None:
                self.current_step = current_step
            if total_steps is not None:
                self.total_steps = total_steps
            if loss is not None:
                self.loss = loss
    
    def set_status(self, status: TaskStatus):
        """
        设置任务状态
        
        Args:
            status: 任务状态
        """
        with self.lock:
            # 记录时间戳
            if status == TaskStatus.PROCESSING and self.started_at is None:
                self.started_at = datetime.now()
            elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED] and self.completed_at is None:
                self.completed_at = datetime.now()
            
            self.status = status
    
    def set_result(self, result: Any):
        """
        设置任务结果
        
        Args:
            result: 任务结果
        """
        with self.lock:
            self.result = result
    
    def set_error(self, error_message: str):
        """
        设置错误信息
        
        Args:
            error_message: 错误信息
        """
        with self.lock:
            self.error_message = error_message
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式
        
        Returns:
            Dict: 任务信息字典
        """
        with self.lock:
            return {
                'task_id': self.task_id,
                'task_type': self.task_type,
                'status': self.status.value,
                'progress': self.progress,
                'current_step': self.current_step,
                'total_steps': self.total_steps,
                'loss': self.loss,
                'error_message': self.error_message,
                'created_at': self.created_at.isoformat(),
                'started_at': self.started_at.isoformat() if self.started_at else None,
                'completed_at': self.completed_at.isoformat() if self.completed_at else None,
                'duration': (self.completed_at - self.started_at).total_seconds() if self.started_at and self.completed_at else None
            }


class TaskManager:
    """任务管理器"""
    
    def __init__(self, max_tasks: int = 1000, task_timeout_hours: int = 24):
        """
        初始化任务管理器
        
        Args:
            max_tasks: 最大任务数
            task_timeout_hours: 任务超时时间（小时）
        """
        self.max_tasks = max_tasks
        self.task_timeout_hours = task_timeout_hours
        self.tasks: Dict[str, TaskInfo] = {}
        self.lock = threading.Lock()
        self.cleanup_timer = None
        self._start_cleanup_timer()
    
    def _start_cleanup_timer(self):
        """启动清理定时器"""
        if self.cleanup_timer:
            self.cleanup_timer.cancel()
        
        self.cleanup_timer = threading.Timer(3600, self._cleanup_expired_tasks)  # 每小时清理一次
        self.cleanup_timer.daemon = True
        self.cleanup_timer.start()
    
    def create_task(self, task_id: str, task_type: str, data: Dict[str, Any] = None) -> TaskInfo:
        """
        创建任务
        
        Args:
            task_id: 任务ID
            task_type: 任务类型
            data: 任务数据
            
        Returns:
            TaskInfo: 任务信息
        """
        with self.lock:
            # 检查任务是否已存在
            if task_id in self.tasks:
                return self.tasks[task_id]
            
            # 控制任务数量
            if len(self.tasks) >= self.max_tasks:
                self._remove_oldest_task()
            
            # 创建新任务
            task_info = TaskInfo(task_id, task_type, data)
            self.tasks[task_id] = task_info
            
            logger.info(f"创建任务: {task_id}, 类型: {task_type}")
            return task_info
    
    def get_task(self, task_id: str) -> Optional[TaskInfo]:
        """
        获取任务信息
        
        Args:
            task_id: 任务ID
            
        Returns:
            TaskInfo or None: 任务信息
        """
        with self.lock:
            return self.tasks.get(task_id)
    
    def update_task_progress(self, task_id: str, progress: int, current_step: int = None, 
                           total_steps: int = None, loss: float = None):
        """
        更新任务进度
        
        Args:
            task_id: 任务ID
            progress: 进度百分比 (0-100)
            current_step: 当前步骤
            total_steps: 总步骤数
            loss: 损失值
        """
        task_info = self.get_task(task_id)
        if task_info:
            task_info.update_progress(progress, current_step, total_steps, loss)
            logger.debug(f"更新任务 {task_id} 进度: {progress}%")
    
    def set_task_status(self, task_id: str, status: TaskStatus):
        """
        设置任务状态
        
        Args:
            task_id: 任务ID
            status: 任务状态
        """
        task_info = self.get_task(task_id)
        if task_info:
            task_info.set_status(status)
            logger.info(f"设置任务 {task_id} 状态为: {status.value}")
    
    def set_task_result(self, task_id: str, result: Any):
        """
        设置任务结果
        
        Args:
            task_id: 任务ID
            result: 任务结果
        """
        task_info = self.get_task(task_id)
        if task_info:
            task_info.set_result(result)
            logger.info(f"设置任务 {task_id} 结果")
    
    def set_task_error(self, task_id: str, error_message: str):
        """
        设置任务错误信息
        
        Args:
            task_id: 任务ID
            error_message: 错误信息
        """
        task_info = self.get_task(task_id)
        if task_info:
            task_info.set_error(error_message)
            task_info.set_status(TaskStatus.FAILED)
            logger.error(f"任务 {task_id} 发生错误: {error_message}")
    
    def _remove_oldest_task(self):
        """删除最旧的任务"""
        if not self.tasks:
            return
        
        # 找到最旧的任务
        oldest_task_id = None
        oldest_time = datetime.now()
        
        for task_id, task_info in self.tasks.items():
            if task_info.created_at < oldest_time:
                oldest_time = task_info.created_at
                oldest_task_id = task_id
        
        # 删除最旧的任务
        if oldest_task_id:
            del self.tasks[oldest_task_id]
            logger.info(f"删除最旧任务: {oldest_task_id}")
    
    def _cleanup_expired_tasks(self):
        """清理过期任务"""
        try:
            current_time = datetime.now()
            expired_tasks = []
            
            with self.lock:
                for task_id, task_info in self.tasks.items():
                    # 检查任务是否超时
                    if current_time - task_info.created_at > timedelta(hours=self.task_timeout_hours):
                        expired_tasks.append(task_id)
            
            # 删除过期任务
            for task_id in expired_tasks:
                with self.lock:
                    if task_id in self.tasks:
                        del self.tasks[task_id]
                logger.info(f"清理过期任务: {task_id}")
        
        finally:
            # 重新启动清理定时器
            self._start_cleanup_timer()
    
    def get_all_tasks(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有任务信息
        
        Returns:
            Dict: 所有任务信息
        """
        with self.lock:
            return {task_id: task_info.to_dict() for task_id, task_info in self.tasks.items()}
    
    def get_active_tasks(self) -> Dict[str, Dict[str, Any]]:
        """
        获取活跃任务信息（处理中和待处理的任务）
        
        Returns:
            Dict: 活跃任务信息
        """
        with self.lock:
            active_tasks = {}
            for task_id, task_info in self.tasks.items():
                if task_info.status in [TaskStatus.PENDING, TaskStatus.PROCESSING]:
                    active_tasks[task_id] = task_info.to_dict()
            return active_tasks
    
    def cancel_task(self, task_id: str) -> bool:
        """
        取消任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            bool: 是否成功取消
        """
        task_info = self.get_task(task_id)
        if task_info and task_info.status in [TaskStatus.PENDING, TaskStatus.PROCESSING]:
            task_info.set_status(TaskStatus.CANCELLED)
            logger.info(f"取消任务: {task_id}")
            return True
        return False


# 创建全局任务管理器实例
task_manager = TaskManager()


def get_task_manager():
    """获取任务管理器实例"""
    return task_manager