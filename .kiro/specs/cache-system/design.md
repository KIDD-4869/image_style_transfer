# 缓存系统设计文档

## 概述

缓存系统是图像处理应用的关键性能优化组件，通过存储已处理的图像结果来避免重复计算。系统采用两层缓存架构：内存缓存（快速访问）和磁盘缓存（持久化存储），使用LRU策略管理缓存容量。

### 设计目标

- **性能**: 缓存命中时响应时间 < 1秒
- **可靠性**: 支持持久化，系统重启后缓存仍可用
- **可扩展性**: 支持配置化的缓存大小和策略
- **可维护性**: 清晰的接口和完善的监控

## 架构

### 系统架构图

```
┌─────────────────────────────────────────────────────────┐
│                    Application Layer                     │
│                  (Image Processing API)                  │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                   Cache Manager                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Cache Key Generator                              │  │
│  │  - Image Hash (MD5/SHA256)                        │  │
│  │  - Strategy Identifier                            │  │
│  └──────────────────────────────────────────────────┘  │
│                         │                                │
│         ┌───────────────┴───────────────┐               │
│         ▼                               ▼               │
│  ┌─────────────┐                ┌─────────────┐        │
│  │   Memory    │                │    Disk     │        │
│  │   Cache     │                │   Cache     │        │
│  │   (LRU)     │                │  (Files)    │        │
│  └─────────────┘                └─────────────┘        │
│         │                               │               │
│         └───────────────┬───────────────┘               │
│                         ▼                                │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Statistics & Monitoring                          │  │
│  │  - Hit/Miss Rate                                  │  │
│  │  - Cache Size                                     │  │
│  │  - Performance Metrics                            │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### 缓存查询流程

```
Request → Generate Cache Key
              │
              ▼
         Check Memory Cache
              │
         ┌────┴────┐
         │         │
      Found?    Not Found
         │         │
         │         ▼
         │    Check Disk Cache
         │         │
         │    ┌────┴────┐
         │    │         │
         │  Found?   Not Found
         │    │         │
         │    ▼         ▼
         │  Load to   Return
         │  Memory    None
         │    │
         └────┴────┐
                   │
                   ▼
              Return Result
```

## 组件和接口

### 1. CacheManager

主缓存管理器，协调内存和磁盘缓存。

```python
class CacheManager:
    """缓存管理器"""
    
    def __init__(
        self,
        memory_size_mb: int = 100,
        disk_size_mb: int = 1000,
        cache_dir: str = "./cache",
        ttl_hours: int = 24
    ):
        """
        初始化缓存管理器
        
        Args:
            memory_size_mb: 内存缓存大小（MB）
            disk_size_mb: 磁盘缓存大小（MB）
            cache_dir: 缓存目录路径
            ttl_hours: 缓存项生存时间（小时）
        """
        pass
    
    def get(
        self,
        image: Image.Image,
        strategy: ProcessingStrategy
    ) -> Optional[ProcessingResult]:
        """
        获取缓存的处理结果
        
        Args:
            image: 输入图像
            strategy: 处理策略
            
        Returns:
            处理结果或None（未命中）
        """
        pass
    
    def set(
        self,
        image: Image.Image,
        strategy: ProcessingStrategy,
        result: ProcessingResult
    ) -> bool:
        """
        保存处理结果到缓存
        
        Args:
            image: 输入图像
            strategy: 处理策略
            result: 处理结果
            
        Returns:
            是否成功保存
        """
        pass
    
    def clear(self) -> None:
        """清空所有缓存"""
        pass
    
    def get_stats(self) -> CacheStats:
        """获取缓存统计信息"""
        pass
```

### 2. CacheKeyGenerator

生成缓存键的组件。

```python
class CacheKeyGenerator:
    """缓存键生成器"""
    
    @staticmethod
    def generate(
        image: Image.Image,
        strategy: ProcessingStrategy
    ) -> str:
        """
        生成缓存键
        
        Args:
            image: 输入图像
            strategy: 处理策略
            
        Returns:
            缓存键字符串
        """
        pass
    
    @staticmethod
    def hash_image(image: Image.Image) -> str:
        """
        计算图像哈希
        
        Args:
            image: 输入图像
            
        Returns:
            图像哈希字符串
        """
        pass
```

### 3. MemoryCache

内存缓存实现（LRU策略）。

```python
class MemoryCache:
    """内存缓存（LRU）"""
    
    def __init__(self, max_size_mb: int, ttl_hours: int):
        """
        初始化内存缓存
        
        Args:
            max_size_mb: 最大缓存大小（MB）
            ttl_hours: 缓存项生存时间（小时）
        """
        pass
    
    def get(self, key: str) -> Optional[CacheEntry]:
        """获取缓存项"""
        pass
    
    def set(self, key: str, value: Any, size_bytes: int) -> bool:
        """设置缓存项"""
        pass
    
    def remove(self, key: str) -> bool:
        """删除缓存项"""
        pass
    
    def clear(self) -> None:
        """清空缓存"""
        pass
    
    def cleanup_expired(self) -> int:
        """清理过期项，返回清理数量"""
        pass
```

### 4. DiskCache

磁盘缓存实现。

```python
class DiskCache:
    """磁盘缓存"""
    
    def __init__(self, cache_dir: str, max_size_mb: int):
        """
        初始化磁盘缓存
        
        Args:
            cache_dir: 缓存目录
            max_size_mb: 最大缓存大小（MB）
        """
        pass
    
    def get(self, key: str) -> Optional[ProcessingResult]:
        """从磁盘加载缓存"""
        pass
    
    def set(self, key: str, result: ProcessingResult) -> bool:
        """保存到磁盘"""
        pass
    
    def remove(self, key: str) -> bool:
        """删除缓存文件"""
        pass
    
    def clear(self) -> None:
        """清空磁盘缓存"""
        pass
    
    def cleanup_old_files(self) -> int:
        """清理旧文件，返回清理数量"""
        pass
```

## 数据模型

### CacheEntry

```python
@dataclass
class CacheEntry:
    """缓存项"""
    key: str
    value: Any
    size_bytes: int
    created_at: datetime
    accessed_at: datetime
    access_count: int
```

### CacheStats

```python
@dataclass
class CacheStats:
    """缓存统计"""
    hits: int
    misses: int
    hit_rate: float
    memory_items: int
    memory_size_mb: float
    disk_items: int
    disk_size_mb: float
    total_requests: int
```

### CacheConfig

```python
@dataclass
class CacheConfig:
    """缓存配置"""
    memory_size_mb: int = 100
    disk_size_mb: int = 1000
    cache_dir: str = "./cache"
    ttl_hours: int = 24
    cleanup_interval_minutes: int = 30
    enable_disk_cache: bool = True
```

## 正确性属性

*属性是应该在系统所有有效执行中保持为真的特征或行为——本质上是关于系统应该做什么的形式化陈述。属性作为人类可读规范和机器可验证正确性保证之间的桥梁。*

### 缓存键生成属性

**属性 1: 缓存键唯一性**
*对于任意*图像和处理策略，生成的缓存键应该是非空字符串
**验证需求: 1.1**

**属性 2: 缓存键确定性**
*对于任意*图像和策略，多次生成缓存键应该返回相同的结果
**验证需求: 1.2**

**属性 3: 策略区分性**
*对于任意*图像，使用不同策略应该生成不同的缓存键
**验证需求: 1.3**

**属性 4: 图像区分性**
*对于任意*两个不同的图像，使用相同策略应该生成不同的缓存键
**验证需求: 1.4**

**属性 5: 键生成性能**
*对于任意*图像和策略，缓存键生成应该在100毫秒内完成
**验证需求: 1.5**

### 内存缓存属性

**属性 6: LRU淘汰正确性**
*对于任意*满容量的缓存，添加新项时应该淘汰最久未访问的项
**验证需求: 2.1**

**属性 7: 访问时间更新**
*对于任意*缓存项，访问后其访问时间应该被更新
**验证需求: 2.2**

**属性 8: 时间戳记录**
*对于任意*新添加的缓存项，应该记录创建时间和访问时间
**验证需求: 2.3**

**属性 9: 统计信息完整性**
*对于任意*缓存状态，统计信息应该包含命中率、未命中率和项数量
**验证需求: 2.4**

**属性 10: TTL过期清理**
*对于任意*超过TTL的缓存项，应该被自动删除
**验证需求: 2.5**

### 持久化属性

**属性 11: 磁盘持久化**
*对于任意*处理结果，保存后应该在磁盘上存在对应文件
**验证需求: 3.1**

**属性 12: 磁盘加载到内存**
*对于任意*磁盘缓存项，加载后应该出现在内存缓存中
**验证需求: 3.3**

**属性 13: 磁盘容量管理**
*对于任意*超过容量的磁盘缓存，应该删除最旧的文件
**验证需求: 3.4**

### 接口操作属性

**属性 14: Get方法正确性**
*对于任意*缓存键，get方法应该返回对应的结果或None
**验证需求: 4.1**

**属性 15: Set-Get往返**
*对于任意*图像和结果，set后立即get应该返回相同的结果
**验证需求: 4.2**

**属性 16: Clear清空完整性**
*对于任意*缓存状态，clear后所有get操作应该返回None
**验证需求: 4.3**

**属性 17: 统计方法完整性**
*对于任意*缓存状态，get_stats应该返回包含所有必要字段的统计信息
**验证需求: 4.4**

### 性能属性

**属性 18: 缓存命中性能**
*对于任意*缓存命中的请求，应该在1秒内返回结果
**验证需求: 5.1**

**属性 19: 缓存未命中开销**
*对于任意*缓存未命中的请求，额外延迟应该小于100毫秒
**验证需求: 5.2**

**属性 20: 线程安全性**
*对于任意*并发访问场景，缓存操作应该保持数据一致性
**验证需求: 5.3**

**属性 21: 低命中率警告**
*对于任意*命中率低于50%的场景，应该记录警告日志
**验证需求: 5.5**

### 配置管理属性

**属性 22: 配置更新生效**
*对于任意*有效的配置更新，新配置应该立即生效
**验证需求: 6.3**

**属性 23: 配置查询完整性**
*对于任意*时刻，配置查询应该返回所有配置参数
**验证需求: 6.4**

**属性 24: 配置验证**
*对于任意*配置参数，系统应该验证其合理性
**验证需求: 6.5**

### 监控统计属性

**属性 25: 统计数据准确性**
*对于任意*操作序列，统计计数器应该准确反映操作次数
**验证需求: 7.1, 7.3**

**属性 26: 状态查询准确性**
*对于任意*缓存状态，使用量统计应该准确反映实际占用
**验证需求: 7.2**

**属性 27: JSON导出格式**
*对于任意*统计数据，导出的JSON应该格式正确且可解析
**验证需求: 7.4**

**属性 28: 统计重置隔离**
*对于任意*缓存状态，重置统计后计数器归零但缓存数据保留
**验证需求: 7.5**

## 错误处理

### 错误类型

1. **CacheKeyError**: 缓存键生成失败
2. **CacheStorageError**: 存储操作失败
3. **CacheLoadError**: 加载操作失败
4. **CacheConfigError**: 配置错误

### 错误处理策略

- 所有缓存操作失败不应影响主流程
- 记录详细的错误日志
- 缓存未命中时返回None，不抛出异常
- 磁盘操作失败时降级到仅使用内存缓存

## 测试策略

### 单元测试

- 测试缓存键生成的各种场景
- 测试LRU淘汰逻辑
- 测试TTL过期清理
- 测试统计计数器更新

### 属性测试

使用`hypothesis`库进行属性测试：

- 每个属性测试运行至少100次迭代
- 使用随机生成的图像和策略
- 测试并发场景
- 测试边界条件

每个属性测试必须使用以下格式标注：
```python
# Feature: cache-system, Property X: <property description>
```

### 集成测试

- 测试完整的缓存工作流
- 测试内存和磁盘缓存协同
- 测试系统重启后的缓存恢复

### 性能测试

- 基准测试：测量缓存命中和未命中的响应时间
- 负载测试：测试高并发场景
- 容量测试：测试大量缓存项的性能

## 实现注意事项

### 线程安全

- 使用`threading.Lock`保护共享数据结构
- 内存缓存使用`OrderedDict`配合锁实现LRU
- 统计计数器使用原子操作

### 性能优化

- 图像哈希使用MD5（快速）而非SHA256
- 内存缓存使用字典实现O(1)查找
- 磁盘缓存使用文件系统的mtime进行排序

### 资源管理

- 定期清理过期缓存项
- 监控磁盘空间使用
- 限制单个缓存项的最大大小

## 监控指标

### 关键指标

- **命中率**: hits / (hits + misses)
- **平均响应时间**: 缓存命中的平均时间
- **内存使用率**: 当前使用 / 最大容量
- **磁盘使用率**: 当前使用 / 最大容量

### 告警阈值

- 命中率 < 50%: 警告
- 内存使用率 > 90%: 警告
- 磁盘使用率 > 95%: 警告
- 平均响应时间 > 2秒: 警告
