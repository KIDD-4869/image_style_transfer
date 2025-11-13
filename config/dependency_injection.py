#!/usr/bin/env python3
"""
依赖注入容器 - 管理组件依赖关系
"""

import threading
from typing import Dict, Any, Type, TypeVar, Callable, Optional
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')

class DIContainer:
    """依赖注入容器"""
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
        self._singletons: Dict[str, Any] = {}
        self._lock = threading.RLock()
    
    def register_singleton(self, interface: Type[T], implementation: Type[T], 
                          name: Optional[str] = None) -> 'DIContainer':
        """注册单例服务"""
        service_name = name or interface.__name__
        
        with self._lock:
            self._factories[service_name] = lambda: implementation()
            logger.debug(f"Registered singleton: {service_name}")
        
        return self
    
    def register_instance(self, interface: Type[T], instance: T, 
                         name: Optional[str] = None) -> 'DIContainer':
        """注册实例"""
        service_name = name or interface.__name__
        
        with self._lock:
            self._singletons[service_name] = instance
            logger.debug(f"Registered instance: {service_name}")
        
        return self
    
    def register_factory(self, interface: Type[T], factory: Callable[[], T], 
                        name: Optional[str] = None) -> 'DIContainer':
        """注册工厂方法"""
        service_name = name or interface.__name__
        
        with self._lock:
            self._factories[service_name] = factory
            logger.debug(f"Registered factory: {service_name}")
        
        return self
    
    def get(self, interface: Type[T], name: Optional[str] = None) -> T:
        """获取服务实例"""
        service_name = name or interface.__name__
        
        with self._lock:
            # 首先检查已创建的单例
            if service_name in self._singletons:
                return self._singletons[service_name]
            
            # 检查工厂方法
            if service_name in self._factories:
                instance = self._factories[service_name]()
                self._singletons[service_name] = instance
                return instance
            
            # 检查直接注册的服务
            if service_name in self._services:
                return self._services[service_name]
        
        raise ValueError(f"Service not registered: {service_name}")
    
    def clear(self):
        """清空容器"""
        with self._lock:
            self._services.clear()
            self._factories.clear()
            self._singletons.clear()

# 全局DI容器
container = DIContainer()

def get_container() -> DIContainer:
    """获取全局DI容器"""
    return container