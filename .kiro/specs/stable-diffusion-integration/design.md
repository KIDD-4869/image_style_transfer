# Design Document

## Overview

本设计文档描述了将 Stable Diffusion 模型集成到宫崎骏风格图片转换器的技术方案。该集成将提供基于最先进的扩散模型技术的 AI 艺术风格转换能力，使用户能够将普通照片转换为高质量的宫崎骏动漫风格艺术作品。

核心设计理念：
- 使用 Stable Diffusion v1.5 的 img2img 功能进行风格转换
- 提供多种处理策略以平衡质量和速度
- 自动检测和利用可用的硬件加速
- 提供清晰的进度反馈和错误处理
- 与现有的处理器架构无缝集成

## Architecture

### 系统架构

```
用户请求
    ↓
Flask Web应用 (app.py)
    ↓
任务管理器 (TaskManager)
    ↓
处理器选择逻辑
    ↓
StableDiffusionProcessor
    ↓
    ├─ 模型加载 (Hugging Face Diffusers)
    ├─ 图像预处理
    ├─ SD推理 (img2img)
    └─ 后处理
    ↓
ProcessingResult
    ↓
返回给用户
```

### 处理流程

1. **初始化阶段**
   - 检测可用设备（GPU/CPU）
   - 设置设备特定的优化参数
   - 准备提示词模板

2. **模型加载阶段**
   - 从 Hugging Face Hub 加载 SD v1.5 模型
   - 根据设备类型选择精度（float16/float32）
   - 应用优化设置（attention slicing等）

3. **图像处理阶段**
   - 预处理：格式转换、尺寸调整、归一化
   - 推理：使用 img2img pipeline 进行风格转换
   - 后处理：调整回原始尺寸

4. **结果返回阶段**
   - 封装处理结果和元数据
   - 更新任务状态
   - 返回给前端展示

## Components and Interfaces

### StableDiffusionProcessor

主要的处理器类，继承自 `BaseProcessor`。

**职责**:
- 管理 Stable Diffusion 模型的生命周期
- 执行图像预处理和后处理
- 根据策略配置推理参数
- 提供进度更新和错误处理

**接口**:
```python
class StableDiffusionProcessor(BaseProcessor):
    def __init__(self):
        """初始化处理器，设置设备和提示词"""
        
    def process(
        self, 
        image: Image.Image,
        strategy: ProcessingStrategy = ProcessingStrategy.BALANCED,
        **kwargs
    ) -> ProcessingResult:
        """处理图像并返回结果"""
        
    def _load_model(self) -> bool:
        """加载 Stable Diffusion 模型"""
        
    def _preprocess(self, image: Image.Image, target_size=512) -> Image.Image:
        """预处理输入图像"""
```

### ProcessingStrategy

定义处理策略的枚举类型。

```python
class ProcessingStrategy(Enum):
    FAST = "fast"           # 快速模式：20步，0.5强度
    BALANCED = "balanced"   # 平衡模式：30步，0.65强度
    QUALITY = "quality"     # 高质量：50步，0.75强度
```

### ProcessingResult

封装处理结果的数据类。

```python
@dataclass
class ProcessingResult:
    success: bool
    image: Optional[Image.Image] = None
    processing_time: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### 集成点

**app.py 中的处理器选择逻辑**:
```python
# 根据用户选择的模式决定使用哪个处理器
if use_sd:
    processor = StableDiffusionProcessor()
elif use_gan:
    processor = AnimeGANProcessor()
else:
    processor = GhibliProcessor()

# 映射策略
strategy_map = {
    'sd': ProcessingStrategy.BALANCED,
    'sd_fast': ProcessingStrategy.FAST,
    'sd_quality': ProcessingStrategy.QUALITY
}
```

## Data Models

### 输入数据

**图像输入**:
- 格式：PIL Image 对象
- 支持的模式：任意（自动转换为RGB）
- 尺寸：任意（自动调整）

**策略参数**:
```python
{
    'strategy': ProcessingStrategy,  # 处理策略
    'target_size': int,              # 目标尺寸（默认512）
}
```

### 输出数据

**ProcessingResult**:
```python
{
    'success': bool,                 # 是否成功
    'image': Image.Image,            # 处理后的图像
    'processing_time': float,        # 处理时间（秒）
    'error_message': str,            # 错误信息（如果失败）
    'metadata': {
        'strategy': str,             # 使用的策略
        'model': str,                # 模型名称
        'device': str,               # 使用的设备
        'strength': float,           # 转换强度
        'steps': int,                # 推理步数
        'guidance_scale': float      # 引导系数
    }
}
```

### 配置数据

**策略配置映射**:
```python
STRATEGY_CONFIGS = {
    ProcessingStrategy.FAST: {
        'strength': 0.5,
        'num_inference_steps': 20,
        'guidance_scale': 7.0
    },
    ProcessingStrategy.BALANCED: {
        'strength': 0.65,
        'num_inference_steps': 30,
        'guidance_scale': 7.5
    },
    ProcessingStrategy.QUALITY: {
        'strength': 0.75,
        'num_inference_steps': 50,
        'guidance_scale': 8.0
    }
}
```

**提示词模板**:
```python
GHIBLI_PROMPT = (
    "Studio Ghibli anime style, Hayao Miyazaki art, "
    "vibrant colors, soft lighting, dreamy atmosphere, "
    "hand-drawn animation, detailed, high quality, "
    "whimsical, magical realism"
)

NEGATIVE_PROMPT = (
    "photorealistic, photo, realistic, ugly, blurry, "
    "low quality, bad anatomy, watermark, text"
)
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Model consistency

*For any* processing request using Stable Diffusion mode, the returned metadata should indicate that Stable Diffusion v1.5 model was used.

**Validates: Requirements 1.1**

### Property 2: Aspect ratio preservation in preprocessing

*For any* input image with a given aspect ratio, after preprocessing, the output image should maintain the same aspect ratio (within a small tolerance for rounding to multiples of 8).

**Validates: Requirements 1.2, 6.2**

### Property 3: Strategy parameter mapping

*For any* processing strategy (FAST, BALANCED, QUALITY), the returned metadata should contain the correct strength, steps, and guidance_scale values as defined in the strategy configuration.

**Validates: Requirements 2.1, 2.2, 2.3**

### Property 4: Progress reporting ranges

*For any* processing operation, the progress values reported at each stage should fall within the expected ranges: model loading (0-15%), preprocessing (15-30%), inference (30-90%), postprocessing (90-100%).

**Validates: Requirements 3.1, 3.2, 3.3, 3.4**

### Property 5: Error information on failure

*For any* processing operation that fails, the result should have success=False and should contain a non-empty error_message field.

**Validates: Requirements 3.5, 7.4**

### Property 6: Device information logging

*For any* successful model initialization, the metadata should contain the device type (cpu or cuda) that was used.

**Validates: Requirements 4.5**

### Property 7: Prompt keyword inclusion

*For any* processing operation, the internal prompt string should contain the keywords "Studio Ghibli" and "Hayao Miyazaki", and the negative prompt should contain "photorealistic".

**Validates: Requirements 5.1, 5.2**

### Property 8: RGB mode conversion

*For any* input image in a non-RGB color mode (e.g., RGBA, L, CMYK), after preprocessing, the image should be in RGB mode.

**Validates: Requirements 6.1**

### Property 9: Dimension divisibility by 8

*For any* preprocessed image, both width and height should be divisible by 8.

**Validates: Requirements 6.3**

### Property 10: Output size matching input

*For any* input image with dimensions (W, H), the final output image should have the same dimensions (W, H).

**Validates: Requirements 6.4**

### Property 11: Metadata completeness

*For any* successful processing operation, the result metadata should contain all required fields: strategy, model, device, strength, steps, and guidance_scale.

**Validates: Requirements 7.2, 7.3**

### Property 12: Processing time recording

*For any* processing operation (successful or failed), the result should contain a non-negative processing_time value.

**Validates: Requirements 7.1**

### Property 13: Logging of key operations

*For any* processing operation, the log should contain entries for key steps: model loading, preprocessing, inference start, and completion/failure.

**Validates: Requirements 7.5**

## Error Handling

### 错误类型和处理策略

**1. 模型加载失败**
- 原因：网络问题、磁盘空间不足、依赖缺失
- 处理：记录详细错误日志，返回失败结果，在 app.py 中回退到备用处理器
- 用户反馈：显示清晰的错误信息和建议（检查网络、磁盘空间等）

**2. 内存不足**
- 原因：图像过大、设备内存有限
- 处理：捕获 OOM 异常，记录错误，返回失败结果
- 用户反馈：建议使用更小的图像或快速模式

**3. 图像格式错误**
- 原因：损坏的图像文件、不支持的格式
- 处理：在预处理阶段捕获异常，返回失败结果
- 用户反馈：提示上传有效的图像文件

**4. 推理超时**
- 原因：设备性能不足、参数设置过高
- 处理：设置合理的超时时间，超时后终止处理
- 用户反馈：建议使用快速模式或更小的图像

**5. 依赖缺失**
- 原因：未安装 diffusers、transformers 等库
- 处理：在导入时捕获 ImportError，记录错误
- 用户反馈：提示安装所需依赖

### 错误恢复机制

```python
try:
    # 尝试使用 Stable Diffusion
    processor = StableDiffusionProcessor()
    result = processor.process(image, strategy)
except Exception as e:
    logger.warning(f"SD处理失败: {e}，回退到CV处理器")
    # 回退到传统CV方法
    processor = GhibliProcessor()
    result = processor.process(image, strategy)
```

### 日志记录

所有错误都应记录到日志系统，包括：
- 错误类型和消息
- 完整的堆栈跟踪
- 相关的上下文信息（图像尺寸、策略、设备等）
- 时间戳

```python
logger.error(f"处理失败: {e}", exc_info=True, extra={
    'image_size': image.size,
    'strategy': strategy.value,
    'device': self.device
})
```

## Testing Strategy

### 单元测试

单元测试将验证各个组件的独立功能：

**测试 StableDiffusionProcessor 类**:
- 初始化测试：验证设备检测、提示词设置
- 预处理测试：验证图像格式转换、尺寸调整、宽高比保持
- 配置测试：验证不同策略的参数映射
- 错误处理测试：验证各种异常情况的处理

**测试工具函数**:
- 图像尺寸调整算法
- 宽高比计算
- 8的倍数对齐

**测试集成点**:
- app.py 中的处理器选择逻辑
- 策略映射
- 错误回退机制

### 属性测试（Property-Based Testing）

属性测试将使用 Hypothesis 库验证通用属性在大量随机输入下都成立。

**配置**:
- 使用 Hypothesis 作为属性测试库
- 每个属性测试至少运行 100 次迭代
- 每个测试必须用注释标注对应的设计文档属性

**测试标注格式**:
```python
# Feature: stable-diffusion-integration, Property 1: Model consistency
def test_model_consistency_property():
    ...
```

**属性测试覆盖**:
- Property 1-13：所有可测试的正确性属性
- 使用策略生成器生成各种输入组合
- 验证输出满足规定的约束

**生成器设计**:
```python
from hypothesis import given, strategies as st
from hypothesis.strategies import composite

@composite
def image_strategy(draw):
    """生成随机测试图像"""
    width = draw(st.integers(min_value=64, max_value=2048))
    height = draw(st.integers(min_value=64, max_value=2048))
    mode = draw(st.sampled_from(['RGB', 'RGBA', 'L', 'CMYK']))
    return Image.new(mode, (width, height), color='red')

@composite
def strategy_enum(draw):
    """生成随机处理策略"""
    return draw(st.sampled_from(list(ProcessingStrategy)))
```

### 集成测试

集成测试将验证完整的工作流程：

**端到端测试**:
- 从图像上传到结果返回的完整流程
- 不同模式（SD、GAN、CV）之间的切换
- 错误情况下的回退机制

**API测试**:
- 测试 /upload 端点
- 测试 /progress 端点
- 测试 /result 端点
- 测试错误响应

**性能测试**:
- 不同尺寸图像的处理时间
- 不同策略的性能对比
- 内存使用监控

### 测试环境

**CPU测试环境**:
- 验证CPU模式下的功能
- 验证float32精度
- 验证CPU优化设置

**GPU测试环境（如果可用）**:
- 验证CUDA加速
- 验证float16精度
- 验证GPU优化设置

**模拟环境**:
- 模拟模型加载失败
- 模拟内存不足
- 模拟网络错误

### 测试数据

**测试图像集**:
- 各种尺寸：小（256x256）、中（512x512）、大（1024x1024）
- 各种宽高比：正方形、横向、纵向
- 各种格式：RGB、RGBA、L、CMYK
- 边缘情况：极小（64x64）、极大（2048x2048）

**预期结果**:
- 所有测试图像都应成功处理
- 输出尺寸应与输入匹配
- 元数据应完整且正确
