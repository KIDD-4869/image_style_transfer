#!/usr/bin/env python3
"""
异步图像处理器 - 使用ProcessPoolExecutor实现真正的并发处理
"""

import asyncio
from concurrent.futures import ProcessPoolExecutor, Future
from typing import Optional, Callable, Dict, Any
from PIL import Image
import logging
import multiprocessing as mp

from .processors import GhibliProcessor, ProcessingStrategy, ProcessingResult

logger = logging.getLogger(__name__)


class AsyncImageProcessor:
    """异步图像处理器"""
    
    def __init__(self, max_workers: Optional[int] = None):
        """
        初始化异步处理器
        
        Args:
            max_workers: 最大工作进程数，None表示使用CPU核心数
        """
        if max_workers is None:
            max_workers = min(4, mp.cpu_count())  # 最多4个进程
        
        self.max_workers = max_workers
        self.executor = ProcessPoolExecutor(max_workers=max_workers)
        self._active_tasks: Dict[str, Future] = {}
        
        logger.info(f"异步处理器初始化: max_workers={max_workers}")
    
    async def process_async(
        self,
        task_id: str,
        image: Image.Image,
        strategy: ProcessingStrategy = ProcessingStrategy.BALANCED,
        progress_callback: Optional[Callable] = None
    ) -> ProcessingResult:
        """
        异步处理图像
        
        Args:
            task_id: 任务ID
            image: 输入图像
            strategy: 处理策略
            progress_callback: 进度回调函数
            
        Returns:
            ProcessingResult: 处理结果
        """
        loop = asyncio.get_event_loop()
        
        # 在进程池中执行处理
        future = loop.run_in_executor(
            self.executor,
            _process_image_worker,
            image,
            strategy,
            task_id
        )
        
        # 保存活跃任务
        self._active_tasks[task_id] = future
        
        try:
            # 等待处理完成
            result = await future
            return result
        finally:
            # 清理任务
            if task_id in self._active_tasks:
                del self._active_tasks[task_id]
    
    def cancel_task(self, task_id: str) -> bool:
        """
        取消任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            是否成功取消
        """
        if task_id in self._active_tasks:
            future = self._active_tasks[task_id]
            cancelled = future.cancel()
            if cancelled:
                del self._active_tasks[task_id]
                logger.info(f"任务 {task_id} 已取消")
            return cancelled
        return False
    
    def get_active_task_count(self) -> int:
        """获取活跃任务数量"""
        return len(self._active_tasks)
    
    def shutdown(self, wait: bool = True):
        """
        关闭处理器
        
        Args:
            wait: 是否等待所有任务完成
        """
        logger.info("关闭异步处理器...")
        self.executor.shutdown(wait=wait)
        self._active_tasks.clear()
        logger.info("异步处理器已关闭")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


def _process_image_worker(
    image: Image.Image,
    strategy: ProcessingStrategy,
    task_id: str
) -> ProcessingResult:
    """
    工作进程中的图像处理函数
    
    Args:
        image: 输入图像
        strategy: 处理策略
        task_id: 任务ID
        
    Returns:
        ProcessingResult: 处理结果
    """
    try:
        # 创建处理器（在工作进程中）
        processor = GhibliProcessor()
        
        # 处理图像
        result = processor.process(image, strategy=strategy)
        
        logger.info(f"任务 {task_id} 处理完成")
        return result
        
    except Exception as e:
        logger.error(f"任务 {task_id} 处理失败: {e}")
        return ProcessingResult(
            success=False,
            error_message=str(e)
        )


# 全局异步处理器实例
_global_async_processor: Optional[AsyncImageProcessor] = None


def get_async_processor(max_workers: Optional[int] = None) -> AsyncImageProcessor:
    """
    获取全局异步处理器实例
    
    Args:
        max_workers: 最大工作进程数
        
    Returns:
        AsyncImageProcessor: 异步处理器实例
    """
    global _global_async_processor
    
    if _global_async_processor is None:
        _global_async_processor = AsyncImageProcessor(max_workers=max_workers)
    
    return _global_async_processor


def shutdown_async_processor():
    """关闭全局异步处理器"""
    global _global_async_processor
    
    if _global_async_processor is not None:
        _global_async_processor.shutdown()
        _global_async_processor = None
