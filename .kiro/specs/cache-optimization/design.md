# Design Document

## Overview

本设计文档描述了图像处理系统的缓存优化和前端响应问题的解决方案。核心设计理念是：**先快速返回缓存结果，然后在后台持续优化，最终提供更好的结果**。这种渐进式增强策略既保证了用户体验（快速响应），又确保了结果质量（持续优化）。

## Architecture

### 系统架构图

```
┌─────────────┐
│   Frontend  │
│  (Browser)  │
└──────┬──────┘
       │ HTTP Requests
       ▼
┌─────────────────────────────────────┐
│         Flask Application           │
│  ┌──────────────────────────────┐  │
│  │   /upload Endpoint           │  │
│  │   - Create Task              │  │
│  │   - Check Cache              │  │
│  │   - Start Processing         │  │
│  └──────────────────────────────┘  │
│  ┌──────────────────────────────┐  │
│  │   /progress Endpoint         │  │
│  │   - Return Task Status       │  │
│  └──────────────────────────────┘  │
│  ┌──────────────────────────────┐  │
│  │   /result Endpoint           │  │
│  │   - Return Processed Image   │  │
│  └──────────────────────────────┘  │
└───────────┬─────────────────────────┘
            │
    ┌───────┴────────┐
    │                │
    ▼                ▼
┌─────────┐    ┌──────────────┐
│  Task   │    │    Cache     │
│ Manager │    │   Manager    │
└─────────┘    └──────────────┘
    │                │
    │                ├─ Memory Cache
    │                └─ Disk Cache
    ▼
┌─────────────────────┐
│  Processing Thread  │
│  ┌───────────────┐  │
│  │ Main Process  │  │
│  └───────────────┘  │
│  ┌───────────────┐  │
│  │ Optimization  │  │
│  │   (Background)│  │
│  └───────────────┘  │
└─────────────────────┘
```

### 渐进式处理流程

```
User Upload
    │
    ▼
Check Cache ──Yes──> Return Cached Result (Fast)
    │                        │
    No                       ▼
    │                Start Background Optimization
    ▼                        │
Process Image                ▼
    │                Optimize Result
    ▼                        │
Save to Cache                ▼
    │                Update Cache
    ▼                        │
Return Result                ▼
                      Notify Frontend (Optional)
```

## Components and Interfaces

### 1. Task Manager Enhancement

**新增字段到 TaskInfo 类**:

```python
class TaskInfo:
    # 现有字段...
    cache_hit: bool = False           # 是否缓存命中
    optimization_status: str = None   # 优化状态: None, 'pending', 'running', 'completed', 'failed'
    optimization_progress: int = 0    # 优化进度 0-100
    has_optimized_version: bool = False  # 是否有优化版本可用
```

**新增方法**:

```python
def mark_cache_hit(self):
    """标记任务为缓存命中"""
    self.cache_hit = True
    self.metadata['cache_hit'] = True

def start_optimization(self):
    """开始后台优化"""
    self.optimization_status = 'running'
    self.metadata['optimization_started_at'] = datetime.now().isoformat()

def complete_optimization(self):
    """完成后台优化"""
    self.optimization_status = 'completed'
    self.has_optimized_version = True
    self.metadata['optimization_completed_at'] = datetime.now().isoformat()
```

### 2. Cache Manager Enhancement

**新增方法**:

```python
def get_with_metadata(
    self,
    image: Image.Image,
    strategy: ProcessingStrategy
) -> Optional[Tuple[ProcessingResult, Dict[str, Any]]]:
    """
    获取缓存结果及元数据
    
    Returns:
        (result, metadata) 或 None
        metadata包含: cache_age, hit_count, last_optimized等
    """
    pass

def mark_for_optimization(
    self,
    cache_key: str
) -> bool:
    """
    标记缓存项需要优化
    
    Returns:
        是否成功标记
    """
    pass

def update_optimized_result(
    self,
    cache_key: str,
    optimized_result: ProcessingResult
) -> bool:
    """
    更新为优化后的结果
    
    Returns:
        是否成功更新
    """
    pass
```

### 3. Processing Service

**新增服务类**:

```python
class ProgressiveProcessingService:
    """渐进式处理服务"""
    
    def process_with_cache(
        self,
        task_id: str,
        image: Image.Image,
        strategy: ProcessingStrategy
    ) -> ProcessingResult:
        """
        带缓存的渐进式处理
        
        流程:
        1. 检查缓存
        2. 如果命中，立即返回并启动后台优化
        3. 如果未命中，正常处理并保存到缓存
        """
        pass
    
    def optimize_in_background(
        self,
        task_id: str,
        image: Image.Image,
        strategy: ProcessingStrategy,
        cached_result: ProcessingResult
    ):
        """
        后台优化任务
        
        在独立线程中运行，不阻塞主流程
        """
        pass
```

### 4. Frontend Enhancement

**新增JavaScript函数**:

```javascript
// 轮询配置
const POLL_CONFIG = {
    interval: 2000,        // 轮询间隔（毫秒）
    maxAttempts: 150,      // 最大轮询次数（5分钟）
    timeout: 10000         // 单次请求超时（毫秒）
};

// 轮询状态
let pollState = {
    attempts: 0,
    taskId: null,
    interval: null
};

// 改进的轮询函数
function startPollingProgress() {
    pollState.attempts = 0;
    
    pollState.interval = setInterval(async () => {
        pollState.attempts++;
        
        // 检查是否超过最大尝试次数
        if (pollState.attempts > POLL_CONFIG.maxAttempts) {
            stopPolling();
            showError('处理超时，请重试');
            return;
        }
        
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), POLL_CONFIG.timeout);
            
            const response = await fetch(`/progress/${pollState.taskId}`, {
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);
            
            if (!response.ok) {
                console.warn('进度查询失败，继续重试...');
                return;
            }
            
            const progress = await response.json();
            
            // 更新进度显示
            updateProgress(progress.progress || 0);
            
            // 检查是否完成
            if (progress.status === 'completed' && progress.progress >= 100) {
                stopPolling();
                getFinalResult();
            } else if (progress.status === 'failed') {
                stopPolling();
                showError(progress.error_message || '处理失败');
            }
            
        } catch (err) {
            if (err.name === 'AbortError') {
                console.warn('请求超时，继续重试...');
            } else {
                console.error('轮询错误:', err);
            }
        }
    }, POLL_CONFIG.interval);
}

function stopPolling() {
    if (pollState.interval) {
        clearInterval(pollState.interval);
        pollState.interval = null;
    }
}

// 检查优化版本
function checkForOptimizedVersion() {
    // 定期检查是否有优化版本可用
    setInterval(async () => {
        if (!pollState.taskId) return;
        
        try {
            const response = await fetch(`/progress/${pollState.taskId}`);
            const progress = await response.json();
            
            if (progress.has_optimized_version) {
                showOptimizedVersionButton();
            }
        } catch (err) {
            console.error('检查优化版本失败:', err);
        }
    }, 5000);
}
```

## Data Models

### TaskInfo 扩展

```python
@dataclass
class TaskInfo:
    task_id: str
    task_type: str
    status: TaskStatus
    progress: int
    current_step: int
    total_steps: int
    result: Optional[Any]
    error_message: Optional[str]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    metadata: Dict[str, Any]
    
    # 新增字段
    cache_hit: bool = False
    optimization_status: Optional[str] = None
    optimization_progress: int = 0
    has_optimized_version: bool = False
```

### CacheEntry 扩展

```python
@dataclass
class CacheEntry:
    key: str
    result: ProcessingResult
    created_at: datetime
    last_accessed: datetime
    access_count: int
    size_bytes: int
    
    # 新增字段
    needs_optimization: bool = False
    last_optimized: Optional[datetime] = None
    optimization_count: int = 0
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Acceptence Criteria Testing Prework:

1.1 WHEN 缓存命中时 THEN 系统 SHALL 立即设置任务进度为100%
Thoughts: 这是一个关于系统行为的规则，适用于所有缓存命中的情况。我们可以生成随机的图像和策略，触发缓存命中，然后验证任务进度是否为100%
Testable: yes - property

1.2 WHEN 任务进度为100% THEN 系统 SHALL 确保结果数据已正确保存到任务管理器
Thoughts: 这是一个不变量，对于所有进度为100%的任务都应该成立。我们可以生成随机任务，设置进度为100%，然后验证结果数据是否存在
Testable: yes - property

1.3 WHEN 前端轮询到进度100% THEN 前端 SHALL 能够成功获取最终结果
Thoughts: 这是一个端到端的行为测试，涉及前端和后端交互。这更适合作为集成测试的例子
Testable: yes - example

1.4 WHEN 获取结果接口被调用 THEN 系统 SHALL 在2秒内返回响应
Thoughts: 这是一个性能要求，涉及时间测量。虽然可以测试，但这更适合性能测试而不是属性测试
Testable: no

1.5 WHEN 缓存结果被使用 THEN 系统 SHALL 记录缓存命中日志
Thoughts: 这是关于日志记录的要求，对所有缓存命中都应该成立。我们可以验证日志系统是否被正确调用
Testable: yes - property

2.1 WHEN 缓存命中时 THEN 系统 SHALL 启动后台优化任务
Thoughts: 这是关于系统行为的规则，对所有缓存命中都应该成立。我们可以验证后台线程是否被启动
Testable: yes - property

2.2 WHEN 后台优化完成 THEN 系统 SHALL 更新缓存中的结果
Thoughts: 这是关于优化流程的规则，对所有优化任务都应该成立。我们可以验证缓存是否被更新
Testable: yes - property

2.3 WHEN 用户再次请求相同图像 THEN 系统 SHALL 返回优化后的结果
Thoughts: 这是一个round-trip属性的变体。对于任何图像，如果先处理一次（触发优化），再处理第二次，第二次应该返回优化后的结果
Testable: yes - property

2.4 WHEN 优化任务运行时 THEN 系统 SHALL 不阻塞用户获取当前结果
Thoughts: 这是关于并发性的要求。我们可以测试在优化运行时，主线程是否仍然响应
Testable: yes - property

2.5 WHEN 优化失败 THEN 系统 SHALL 保留原有缓存结果
Thoughts: 这是关于错误处理的不变量。对于任何优化失败的情况，原始缓存应该保持不变
Testable: yes - property

3.1 WHEN 轮询超时 THEN 前端 SHALL 显示友好的错误提示
Thoughts: 这是前端UI行为，涉及用户界面显示。这更适合作为UI测试的例子
Testable: yes - example

3.2 WHEN 轮询次数超过限制 THEN 前端 SHALL 停止轮询并提示用户
Thoughts: 这是关于轮询机制的规则。我们可以模拟轮询次数达到限制，验证轮询是否停止
Testable: yes - property

3.3 WHEN 服务器返回错误 THEN 前端 SHALL 根据错误类型决定是否继续轮询
Thoughts: 这涉及错误分类逻辑。对于不同类型的错误，应该有不同的处理策略
Testable: yes - property

3.4 WHEN 任务状态为completed THEN 前端 SHALL 立即停止轮询并获取结果
Thoughts: 这是关于轮询终止条件的规则。对于所有completed状态的任务，轮询都应该停止
Testable: yes - property

3.5 WHEN 网络断开 THEN 前端 SHALL 显示网络错误提示
Thoughts: 这是前端错误处理的UI行为，更适合作为例子
Testable: yes - example

4.1 WHEN 缓存命中 THEN 系统 SHALL 记录DEBUG级别日志包含缓存键
Thoughts: 这是关于日志格式的要求。对于所有缓存命中，日志应该包含特定信息
Testable: yes - property

4.2 WHEN 任务状态变化 THEN 系统 SHALL 记录INFO级别日志包含任务ID和新状态
Thoughts: 这是关于日志记录的规则。对于所有状态变化，都应该记录日志
Testable: yes - property

4.3 WHEN 结果被设置 THEN 系统 SHALL 记录INFO级别日志包含结果大小
Thoughts: 这是关于日志内容的要求。对于所有结果设置操作，日志应该包含大小信息
Testable: yes - property

4.4 WHEN 发生错误 THEN 系统 SHALL 记录ERROR级别日志包含完整堆栈
Thoughts: 这是关于错误日志的要求。对于所有错误，都应该记录完整堆栈
Testable: yes - property

4.5 WHEN 优化任务启动 THEN 系统 SHALL 记录INFO级别日志包含优化策略
Thoughts: 这是关于日志记录的规则。对于所有优化任务启动，都应该记录策略信息
Testable: yes - property

5.1 WHEN 缓存命中 THEN 系统 SHALL 立即返回缓存结果给用户
Thoughts: 这与1.1类似，是关于响应时间的要求。对于所有缓存命中，都应该立即返回
Testable: yes - property

5.2 WHEN 后台优化完成 THEN 系统 SHALL 通知前端有新版本可用
Thoughts: 这是关于通知机制的要求。对于所有优化完成的情况，都应该发送通知
Testable: yes - property

5.3 WHEN 前端收到优化完成通知 THEN 前端 SHALL 自动刷新显示优化后的图像
Thoughts: 这是前端UI行为，更适合作为集成测试的例子
Testable: yes - example

5.4 WHEN 用户正在查看结果 THEN 系统 SHALL 不强制刷新页面
Thoughts: 这是关于用户体验的要求，涉及UI状态管理。这更适合作为UI测试
Testable: no

5.5 WHEN 优化版本可用 THEN 前端 SHALL 显示"查看优化版本"按钮
Thoughts: 这是前端UI显示要求，更适合作为例子
Testable: yes - example

### Property Reflection

审查所有可测试的属性，识别冗余：

- 1.1 和 5.1 都测试缓存命中时的立即响应 → 可以合并
- 1.5, 4.1 都测试缓存命中的日志记录 → 可以合并
- 2.1 和 4.5 都涉及优化任务启动 → 可以合并日志验证到2.1
- 3.4 和轮询停止逻辑重复 → 保留3.4作为主要属性

### Correctness Properties

Property 1: 缓存命中立即完成
*For any* 图像和处理策略，当缓存命中时，任务进度应该立即设置为100%，并且结果应该立即可用
**Validates: Requirements 1.1, 5.1**

Property 2: 结果完整性
*For any* 进度为100%的任务，任务管理器中必须存在有效的结果数据
**Validates: Requirements 1.2**

Property 3: 缓存命中日志记录
*For any* 缓存命中的情况，系统应该记录包含缓存键的DEBUG级别日志
**Validates: Requirements 1.5, 4.1**

Property 4: 后台优化启动
*For any* 缓存命中的情况，系统应该启动后台优化任务，并记录优化策略
**Validates: Requirements 2.1, 4.5**

Property 5: 优化结果更新
*For any* 成功完成的后台优化，缓存中的结果应该被更新为优化后的版本
**Validates: Requirements 2.2**

Property 6: 优化结果可用性
*For any* 图像，如果第一次处理触发了优化，第二次请求相同图像应该返回优化后的结果
**Validates: Requirements 2.3**

Property 7: 非阻塞优化
*For any* 正在运行的优化任务，主处理流程应该能够继续响应新的请求
**Validates: Requirements 2.4**

Property 8: 优化失败保护
*For any* 优化失败的情况，原始缓存结果应该保持不变且仍然可用
**Validates: Requirements 2.5**

Property 9: 轮询限制
*For any* 轮询会话，当轮询次数超过配置的最大值时，轮询应该自动停止
**Validates: Requirements 3.2**

Property 10: 错误分类处理
*For any* 服务器错误响应，前端应该根据错误类型（临时性vs永久性）决定是否继续轮询
**Validates: Requirements 3.3**

Property 11: 完成状态终止
*For any* 任务，当状态变为completed时，前端轮询应该立即停止
**Validates: Requirements 3.4**

Property 12: 状态变化日志
*For any* 任务状态变化，系统应该记录包含任务ID和新状态的INFO级别日志
**Validates: Requirements 4.2**

Property 13: 结果大小日志
*For any* 结果设置操作，系统应该记录包含结果大小的INFO级别日志
**Validates: Requirements 4.3**

Property 14: 错误堆栈日志
*For any* 系统错误，应该记录包含完整堆栈跟踪的ERROR级别日志
**Validates: Requirements 4.4**

Property 15: 优化通知
*For any* 完成的后台优化，系统应该更新任务元数据以指示有优化版本可用
**Validates: Requirements 5.2**

## Error Handling

### 1. 缓存错误处理

```python
try:
    cached_result = cache_manager.get(image, strategy)
except CacheCorruptionError as e:
    logger.error(f"缓存损坏: {e}")
    # 清除损坏的缓存项
    cache_manager.remove(cache_key)
    # 继续正常处理
    cached_result = None
except CacheTimeoutError as e:
    logger.warning(f"缓存访问超时: {e}")
    # 跳过缓存，直接处理
    cached_result = None
```

### 2. 优化任务错误处理

```python
try:
    optimized_result = processor.process(image, strategy)
    cache_manager.update_optimized_result(cache_key, optimized_result)
except OptimizationError as e:
    logger.error(f"优化失败: {e}")
    # 保留原始缓存结果
    task_manager.set_optimization_status(task_id, 'failed')
    # 不影响用户已获取的结果
```

### 3. 前端轮询错误处理

```javascript
try {
    const response = await fetch(`/progress/${taskId}`, {
        signal: controller.signal
    });
    
    if (!response.ok) {
        if (response.status === 404) {
            // 任务不存在，停止轮询
            stopPolling();
            showError('任务不存在或已过期');
        } else if (response.status >= 500) {
            // 服务器错误，继续重试
            console.warn('服务器错误，继续重试...');
        } else {
            // 其他错误，停止轮询
            stopPolling();
            showError('获取进度失败');
        }
        return;
    }
    
    // 处理响应...
} catch (err) {
    if (err.name === 'AbortError') {
        // 超时，继续重试
        console.warn('请求超时，继续重试...');
    } else if (err.message.includes('network')) {
        // 网络错误，显示提示但继续重试
        showNetworkWarning();
    } else {
        // 未知错误，停止轮询
        stopPolling();
        showError('发生未知错误');
    }
}
```

### 4. 任务状态不一致处理

```python
def ensure_task_consistency(task_id: str):
    """确保任务状态一致性"""
    task = task_manager.get_task(task_id)
    
    if task is None:
        raise TaskNotFoundError(f"任务 {task_id} 不存在")
    
    # 检查状态和结果的一致性
    if task.status == TaskStatus.COMPLETED:
        if task.result is None:
            logger.error(f"任务 {task_id} 状态为completed但结果为空")
            # 修复：重新设置为processing并重新处理
            task.status = TaskStatus.PROCESSING
            task.progress = 0
            raise TaskInconsistentError("任务状态不一致，正在修复")
    
    if task.progress >= 100 and task.status != TaskStatus.COMPLETED:
        logger.warning(f"任务 {task_id} 进度100%但状态不是completed")
        # 修复：更新状态
        task.status = TaskStatus.COMPLETED
```

## Testing Strategy

### Unit Testing

使用pytest进行单元测试，覆盖以下场景：

1. **TaskManager测试**
   - 测试缓存命中标记
   - 测试优化状态更新
   - 测试任务状态一致性

2. **CacheManager测试**
   - 测试缓存获取和设置
   - 测试优化标记
   - 测试缓存更新

3. **ProgressiveProcessingService测试**
   - 测试缓存命中流程
   - 测试后台优化启动
   - 测试错误处理

### Property-Based Testing

使用Hypothesis库进行属性测试，验证correctness properties：

**配置**:
- 每个属性测试运行至少100次迭代
- 使用随机生成的图像和策略
- 每个测试明确标注对应的correctness property

**测试框架**: pytest + Hypothesis

**示例测试结构**:

```python
from hypothesis import given, strategies as st
import pytest

@given(
    image=st.builds(generate_random_image),
    strategy=st.sampled_from(list(ProcessingStrategy))
)
def test_property_1_cache_hit_immediate_completion(image, strategy):
    """
    **Feature: cache-optimization, Property 1: 缓存命中立即完成**
    
    For any 图像和处理策略，当缓存命中时，
    任务进度应该立即设置为100%，并且结果应该立即可用
    """
    # 先处理一次以填充缓存
    task_id_1 = create_task()
    process_image(task_id_1, image, strategy)
    
    # 第二次应该缓存命中
    task_id_2 = create_task()
    result = process_image(task_id_2, image, strategy)
    
    task = task_manager.get_task(task_id_2)
    
    # 验证
    assert task.cache_hit == True
    assert task.progress == 100
    assert task.status == TaskStatus.COMPLETED
    assert task.result is not None
```

### Integration Testing

测试端到端流程：

1. **缓存命中流程测试**
   - 上传图像 → 处理 → 再次上传相同图像 → 验证缓存命中

2. **优化流程测试**
   - 缓存命中 → 验证后台优化启动 → 等待优化完成 → 验证结果更新

3. **前端轮询测试**
   - 使用Selenium测试前端轮询行为
   - 验证超时处理
   - 验证错误处理

### Performance Testing

1. **响应时间测试**
   - 缓存命中响应时间 < 100ms
   - 正常处理响应时间 < 2s

2. **并发测试**
   - 测试多个并发请求
   - 验证优化任务不阻塞主流程

3. **压力测试**
   - 测试大量轮询请求
   - 验证系统稳定性
