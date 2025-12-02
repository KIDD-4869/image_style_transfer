# Enhanced Ghibli Processor 使用指南

## 概述

Enhanced Ghibli Processor 是一个专业的宫崎骏风格图片转换系统，旨在达到或超越豆包AI的转换质量。

### 核心特性

🌟 **专业动漫模型** - 使用 Anything V3.0 专门训练的动漫风格模型  
🎯 **结构保持** - 集成 ControlNet 保持原图构图和结构  
✨ **风格增强** - 可选的 Ghibli 风格 LoRA 模型  
🎨 **智能提示词** - 优化的提示词工程，精确描述吉卜力风格  
⚙️ **参数优化** - 更高的 strength (0.75-0.95)、steps (30-80)、guidance (8.0-9.5)  
🔧 **高级调度器** - 使用 DPM++ 2M Karras 或 Euler Ancestral  
🖼️ **后处理增强** - 锐化、色彩调整、暖色调映射  
📊 **质量评估** - 完整的质量指标系统

## 快速开始

### 1. 安装依赖

```bash
# 安装基础依赖
pip install -r requirements.txt

# 安装 Enhanced 处理器专用依赖
pip install diffusers==0.25.0 transformers==4.36.0 accelerate==0.25.0
pip install peft==0.7.1 controlnet-aux==0.0.7 safetensors==0.4.1
pip install omegaconf==2.3.0
```

### 2. 下载模型

```bash
# 下载所有必需的模型（约6-8GB）
python scripts/download_models.py

# 或者只下载基础模型
python scripts/download_models.py --models base

# 或者只下载 ControlNet
python scripts/download_models.py --models controlnet
```

### 3. 运行测试

```bash
# 运行基础功能测试
python3 test_enhanced_processor.py
```

### 4. 启动Web应用

```bash
# 启动Flask应用
python run.py

# 访问 http://localhost:5000
```

## 使用方法

### Web界面使用

1. 打开浏览器访问 `http://localhost:5000`
2. 选择处理策略：
   - 🌟 **Enhanced 超高质量** - 最佳质量，处理时间约90秒
   - ✨ **Enhanced 高质量** - 优秀质量，处理时间约70秒
   - ⚖️ **Enhanced 平衡模式** - 平衡质量和速度，约50秒
   - ⚡ **Enhanced 快速模式** - 快速处理，约30秒
3. 上传图片
4. 等待处理完成
5. 下载结果

### Python API使用

```python
from PIL import Image
from core.processors.enhanced_ghibli_processor import EnhancedGhibliProcessor
from core.models import ProcessingMode

# 初始化处理器
processor = EnhancedGhibliProcessor()

# 加载图片
image = Image.open("input.jpg")

# 处理图片
result = processor.process(
    image,
    mode=ProcessingMode.QUALITY,  # 或 FAST, BALANCED, ULTRA
    progress_callback=lambda percent, msg: print(f"{percent}%: {msg}")
)

# 保存结果
if result.success:
    result.image.save("output.jpg")
    print(f"处理时间: {result.processing_time:.2f}秒")
    print(f"质量分数: {result.quality_metrics.overall_score:.1f}")
else:
    print(f"处理失败: {result.error_message}")
```

## 处理模式对比

| 模式 | Strength | Steps | Guidance | 处理时间(CPU) | 质量 |
|------|----------|-------|----------|---------------|------|
| FAST | 0.70 | 25 | 7.5 | ~30秒 | 良好 |
| BALANCED | 0.80 | 40 | 8.5 | ~50秒 | 优秀 |
| QUALITY | 0.90 | 60 | 9.0 | ~70秒 | 卓越 |
| ULTRA | 0.95 | 80 | 9.5 | ~90秒 | 完美 |

## 配置说明

配置文件位于 `config/enhanced_processor_config.yaml`

### 主要配置项

```yaml
# 模型配置
models:
  base_model: "Linaqruf/anything-v3.0"  # 基础模型
  lora_model: "ghibli-style-lora-v1"    # LoRA模型（可选）
  lora_weight: 0.8                       # LoRA权重
  controlnet_model: "lllyasviel/control_v11p_sd15_canny"

# 设备配置
device:
  auto_detect: true
  preferred: "cuda"  # cuda, cpu, 或 mps (Mac)
  dtype: "float16"   # float16 (GPU) 或 float32 (CPU)

# 预处理配置
preprocessing:
  brightness_threshold: 100  # 暗图像阈值
  brightness_boost: 20       # 亮度提升量
  contrast_threshold: 3.0    # 高对比度阈值
  contrast_reduction: 0.15   # 对比度降低比例

# 后处理配置
postprocessing:
  sharpen_amount: 0.3        # 锐化强度
  saturation_factor: 1.2     # 饱和度因子
  warm_tone_strength: 0.15   # 暖色调强度
```

## 质量优化建议

### 获得最佳效果

1. **使用 ULTRA 或 QUALITY 模式** - 更高的质量需要更多的处理时间
2. **确保输入图片质量** - 高分辨率、清晰的输入图片会产生更好的结果
3. **选择合适的内容类型** - 系统会自动检测，但人物、风景、建筑各有优化
4. **启用 ControlNet** - 保持原图构图和结构
5. **使用 GPU 加速** - CUDA 或 MPS 可以显著提升速度

### 常见问题

**Q: 处理速度太慢怎么办？**  
A: 使用 FAST 或 BALANCED 模式，或者使用 GPU 加速

**Q: 结果不够"动漫化"？**  
A: 尝试使用更高的模式（QUALITY 或 ULTRA），或调整 strength 参数

**Q: 模型下载失败？**  
A: 检查网络连接，或手动从 Hugging Face 下载模型到 `models/cache` 目录

**Q: 内存不足错误？**  
A: 降低图片分辨率，或使用 FAST 模式

**Q: 颜色不够鲜艳？**  
A: 调整配置文件中的 `saturation_factor` 参数

## 与其他模式对比

| 特性 | Enhanced | SD Standard | AnimeGAN | CV算法 |
|------|----------|-------------|----------|--------|
| 质量 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| 速度 | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 结构保持 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 风格准确度 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| 模型大小 | 6-8GB | 4GB | 2GB | - |

## 技术架构

```
EnhancedGhibliProcessor
├── ModelManager (模型管理)
│   ├── Base Model (Anything V3.0)
│   ├── LoRA (Ghibli Style)
│   └── ControlNet (Canny/Depth)
├── PromptEngineer (提示词工程)
├── PreprocessingPipeline (预处理)
│   ├── Content Analysis
│   ├── Brightness/Contrast Adjustment
│   └── ControlNet Condition Generation
├── GenerationEngine (生成引擎)
│   ├── Parameter Configuration
│   ├── Scheduler Selection
│   └── Image Generation
└── PostprocessingPipeline (后处理)
    ├── Sharpening
    ├── Saturation Adjustment
    ├── Warm Tone Mapping
    └── Quality Assessment
```

## 开发和贡献

### 运行测试

```bash
# 基础功能测试
python3 test_enhanced_processor.py

# 完整测试套件（需要模型）
pytest tests/
```

### 添加新功能

1. 在相应的组件中添加功能
2. 更新配置文件
3. 添加测试
4. 更新文档

## 许可证

本项目使用的模型和库遵循各自的许可证：
- Anything V3.0: CreativeML Open RAIL-M
- ControlNet: Apache 2.0
- Diffusers: Apache 2.0

## 致谢

- Stable Diffusion 团队
- Anything V3.0 模型作者
- ControlNet 作者
- Hugging Face 团队

## 更新日志

### v1.0.0 (2024-12)
- 🎉 首次发布
- ✨ 完整的 Enhanced Ghibli Processor 实现
- 🎨 四种处理模式（FAST, BALANCED, QUALITY, ULTRA）
- 🔧 ControlNet 结构保持
- 📊 质量评估系统
- 🌐 Web界面集成
