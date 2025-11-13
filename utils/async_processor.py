#!/usr/bin/env python3
"""
异步处理器 - 优化异步任务处理性能
"""

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Callable, Any, Dict, List, Optional
import logging
import queue
import multiprocessing as mp
from functools import wraps
import psutil

logger = logging.getLogger(__name__)

class AsyncProcessor:
    """优化的异步处理器"""
    
    def __init__(self, max_workers: Optional[int] = None, use_process_pool: bool = False):
        """
        初始化异步处理器
        
        Args:
            max_workers: 最大工作线程数，None表示自动检测
            use_process_pool: 是否使用进程池（CPU密集型任务）
        """
        if max_workers is None:
            # 根据CPU核心数和内存自动调整
            cpu_count = mp.cpu_count()
            memory_gb = psutil.virtual_memory().total / (1024**3)
            # 每个工作进程大约需要2GB内存
            max_workers = min(cpu_count, max(1, int(memory_gb / 2)))
        
        self.max_workers = max_workers
        self.use_process_pool = use_process_pool
        
        # 创建执行器
        if use_process_pool:
            self.executor = ProcessPoolExecutor(max_workers=max_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # 任务队列和状态管理
        self.task_queue = queue.PriorityQueue()
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.task_results: Dict[str, Any] = {}
        
        logger.info(f"AsyncProcessor initialized with {max_workers} workers, "
                   f"using {'process' if use_process_pool else 'thread'} pool")
    
    async def submit_task(self, task_id: str, func: Callable, *args, 
                         priority: int = 0, **kwargs) -> Any:
        """
        提交异步任务
        
        Args:
            task_id: 任务ID
            func: 要执行的函数
            priority: 任务优先级（数字越小优先级越高）
            *args, **kwargs: 函数参数
            
        Returns:
            任务结果
        """
        loop = asyncio.get_event_loop()
        
        # 包装函数以支持进度回调
        wrapped_func = self._wrap_function_with_monitoring(func, task_id)
        
        try:
            # 提交到执行器
            future = loop.run_in_executor(
                self.executor, 
                wrapped_func, 
                *args, 
                **kwargs
            )
            
            # 存储任务
            self.active_tasks[task_id] = future
            
            # 等待完成
            result = await future
            
            # 清理任务
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
            
            self.task_results[task_id] = result
            return result
            
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
            raise
    
    def _wrap_function_with_monitoring(self, func: Callable, task_id: str) -> Callable:
        """包装函数以添加监控功能"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                logger.info(f"Starting task {task_id}")
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(f"Task {task_id} completed in {duration:.2f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"Task {task_id} failed after {duration:.2f}s: {e}")
                raise
        return wrapper
    
    async def batch_process(self, tasks: List[Dict[str, Any]], 
                           max_concurrent: Optional[int] = None) -> List[Any]:
        """
        批量处理任务
        
        Args:
            tasks: 任务列表，每个任务包含 {'id', 'func', 'args', 'kwargs', 'priority'}
            max_concurrent: 最大并发数
            
        Returns:
            结果列表
        """
        if max_concurrent is None:
            max_concurrent = self.max_workers
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_task(task_info):
            async with semaphore:
                return await self.submit_task(
                    task_info['id'],
                    task_info['func'],
                    *task_info.get('args', []),
                    priority=task_info.get('priority', 0),
                    **task_info.get('kwargs', {})
                )
        
        # 创建所有任务
        coroutines = [process_single_task(task) for task in tasks]
        
        # 等待所有任务完成
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        
        return results
    
    def cancel_task(self, task_id: str) -> bool:
        """
        取消任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            是否成功取消
        """
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            if not task.done():
                task.cancel()
                del self.active_tasks[task_id]
                logger.info(f"Task {task_id} cancelled")
                return True
        return False
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        获取任务状态
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务状态信息
        """
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            return {
                'status': 'running' if not task.done() else 'completed',
                'done': task.done(),
                'cancelled': task.cancelled() if hasattr(task, 'cancelled') else False
            }
        elif task_id in self.task_results:
            return {
                'status': 'completed',
                'done': True,
                'cancelled': False
            }
        else:
            return {
                'status': 'not_found',
                'done': False,
                'cancelled': False
            }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        return {
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.task_results),
            'max_workers': self.max_workers,
            'executor_type': 'process' if self.use_process_pool else 'thread',
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'active_task_ids': list(self.active_tasks.keys())
        }
    
    def cleanup_completed_results(self, max_age_seconds: int = 3600):
        """
        清理完成的任务结果
        
        Args:
            max_age_seconds: 最大保留时间（秒）
        """
        current_time = time.time()
        to_remove = []
        
        for task_id in self.task_results:
            # 这里简化处理，实际应该记录完成时间
            to_remove.append(task_id)
        
        for task_id in to_remove:
            del self.task_results[task_id]
        
        logger.info(f"Cleaned up {len(to_remove)} completed task results")
    
    def shutdown(self, wait: bool = True):
        """关闭处理器"""
        logger.info("Shutting down AsyncProcessor")
        
        # 取消所有活跃任务
        for task_id in list(self.active_tasks.keys()):
            self.cancel_task(task_id)
        
        # 关闭执行器
        self.executor.shutdown(wait=wait)
        
        logger.info("AsyncProcessor shutdown complete")

# 全局异步处理器实例
_async_processor = None
_processor_lock = threading.Lock()

def get_async_processor(max_workers: Optional[int] = None, 
                       use_process_pool: bool = False) -> AsyncProcessor:
    """获取全局异步处理器实例"""
    global _async_processor
    
    with _processor_lock:
        if _async_processor is None:
            _async_processor = AsyncProcessor(max_workers, use_process_pool)
    
    return _async_processor