# Stable Diffusion 使用指南

## 📋 目录

- [简介](#简介)
- [安装](#安装)
- [使用方法](#使用方法)
- [性能优化](#性能优化)
- [故障排查](#故障排查)
- [技术细节](#技术细节)

## 简介

Stable Diffusion 是一个最先进的AI图像生成模型，本项目集成了 Stable Diffusion v1.5 的 img2img 功能，用于将普通照片转换为宫崎骏动漫风格的艺术作品。

### 特点

- ✅ 真正的AI艺术风格转换
- ✅ 完全重新绘制，不是简单滤镜
- ✅ 三种处理策略（快速/标准/高质量）
- ✅ 自动硬件检测（CPU/GPU）
- ✅ 智能错误回退机制

## 安装

### 基础要求

- Python 3.9+
- 至少 8GB RAM
- 至少 5GB 磁盘空间（用于模型）

### 安装步骤

1. **安装 PyTorch**：

```bash
# CPU 版本
pip3 install torch torchvision

# GPU 版本（NVIDIA CUDA）
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

2. **安装 Diffusers 和依赖**：

```bash
pip3 install diffusers transformers accelerate
```

3. **验证安装**：

```bash
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import diffusers; print(f'Diffusers: {diffusers.__version__}')"
```

## 使用方法

### Web 界面

1. 启动应用：`python3 app.py`
2. 访问：`http://localhost:5003`
3. 选择 "✨ Stable Diffusion (AI艺术)"
4. 上传图片并开始转换

### 处理模式

| 模式 | 步数 | 强度 | 引导系数 | CPU时间 | GPU时间 | 重绘程度 |
|------|------|------|----------|---------|---------|----------|
| 快速 | 25 | 0.75 | 8.0 | ~25秒 | ~6秒 | 强力重绘 |
| 标准 | 40 | 0.85 | 8.5 | ~40秒 | ~10秒 | 深度重绘 |
| 高质量 | 60 | 0.95 | 9.0 | ~60秒 | ~15秒 | 完全重绘 |

**重绘强度说明**：
- **0.75 (快速)**：保留约25%原图特征，75%重新生成
- **0.85 (标准)**：保留约15%原图特征，85%重新生成
- **0.95 (高质量)**：保留约5%原图特征，95%重新生成，几乎完全重新创作

### 首次使用

首次使用时，系统会自动从 Hugging Face Hub 下载模型：

```
正在加载Stable Diffusion模型...
⚠️ 首次加载需要下载模型（约4GB），请耐心等待...
```

下载完成后，模型会缓存在本地，后续使用无需重新下载。

## 性能优化

### GPU 加速

如果系统有 NVIDIA GPU：

1. **安装 CUDA 版本的 PyTorch**：
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

2. **验证 GPU 可用**：
```bash
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

3. **系统会自动使用 GPU**，无需额外配置

### 内存优化

如果遇到内存不足：

1. **使用快速模式**：减少推理步数
2. **减小图像尺寸**：在代码中调整 `target_size` 参数
3. **关闭其他应用**：释放系统内存

### 模型缓存

模型默认缓存位置：
- Linux/Mac: `~/.cache/huggingface/hub/`
- Windows: `C:\Users\<用户名>\.cache\huggingface\hub\`

## 故障排查

### 问题1：模型下载失败

**症状**：
```
Connection error
Timeout
```

**解决方案**：

1. **设置镜像源**：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

2. **手动下载模型**：
```bash
huggingface-cli download runwayml/stable-diffusion-v1-5
```

3. **检查网络连接**

### 问题2：内存不足

**症状**：
```
RuntimeError: out of memory
CUDA out of memory
```

**解决方案**：

1. **使用快速模式**（20步）
2. **减小图像尺寸**：
```python
# 在 stable_diffusion_processor.py 中
def _preprocess(self, image, target_size=256):  # 从512改为256
```
3. **关闭其他应用**
4. **使用 CPU 模式**（如果在 GPU 上失败）

### 问题3：处理太慢

**症状**：处理时间超过2分钟

**解决方案**：

1. **使用快速模式**（20步）
2. **减小图像尺寸**
3. **使用 GPU**（如果可用）
4. **检查系统资源**：
```bash
# 查看 CPU 使用率
top

# 查看内存使用
free -h
```

### 问题4：依赖冲突

**症状**：
```
ImportError: cannot import name 'StableDiffusionImg2ImgPipeline'
```

**解决方案**：

1. **重新安装依赖**：
```bash
pip3 uninstall diffusers transformers torch
pip3 install torch torchvision diffusers transformers accelerate
```

2. **检查版本**：
```bash
pip3 list | grep -E "torch|diffusers|transformers"
```

### 问题5：自动回退到 CV 算法

**症状**：选择 SD 模式但使用了 CV 算法

**原因**：SD 处理器加载失败，系统自动回退

**解决方案**：

1. **查看日志**：
```bash
tail -f logs/app.log
```

2. **检查依赖安装**：
```bash
python3 -c "from diffusers import StableDiffusionImg2ImgPipeline"
```

3. **手动测试**：
```bash
python3 tests/test_sd_core_functionality.py
```

## 技术细节

### 模型信息

- **模型**：Stable Diffusion v1.5
- **来源**：runwayml/stable-diffusion-v1-5
- **大小**：约4GB
- **架构**：Latent Diffusion Model

### 处理流程

```
输入图像
    ↓
预处理（RGB转换、尺寸调整）
    ↓
编码到潜在空间
    ↓
扩散过程（去噪，N步）
    ↓
解码到图像空间
    ↓
后处理（调整回原始尺寸）
    ↓
输出图像
```

### 参数说明

**strength**（转换强度）：
- 范围：0.0 - 1.0
- 0.5：保留更多原图特征
- 0.75：更强的风格转换

**num_inference_steps**（推理步数）：
- 范围：1 - 150
- 20：快速但质量一般
- 50：慢但质量最好

**guidance_scale**（引导系数）：
- 范围：1.0 - 20.0
- 7.0：更自然
- 8.0：更符合提示词

### 提示词

**正向提示词**：
```
Studio Ghibli anime style, Hayao Miyazaki art,
vibrant colors, soft lighting, dreamy atmosphere,
hand-drawn animation, detailed, high quality,
whimsical, magical realism
```

**负向提示词**：
```
photorealistic, photo, realistic, ugly, blurry,
low quality, bad anatomy, watermark, text
```

## 参考资源

- [Stable Diffusion 官方文档](https://github.com/Stability-AI/stablediffusion)
- [Diffusers 库文档](https://huggingface.co/docs/diffusers)
- [Hugging Face Hub](https://huggingface.co/runwayml/stable-diffusion-v1-5)
- [PyTorch 文档](https://pytorch.org/docs/stable/index.html)

## 常见问题

**Q: 为什么首次使用需要下载模型？**
A: Stable Diffusion 模型约4GB，需要从 Hugging Face Hub 下载。下载后会缓存，后续使用无需重新下载。

**Q: 可以离线使用吗？**
A: 模型下载后可以离线使用。首次使用需要联网下载模型。

**Q: GPU 和 CPU 性能差异有多大？**
A: GPU 通常比 CPU 快4-6倍。例如，标准模式在 CPU 上需要30秒，在 GPU 上只需8秒。

**Q: 可以自定义提示词吗？**
A: 可以。在 `stable_diffusion_processor.py` 中修改 `ghibli_prompt` 和 `negative_prompt`。

**Q: 支持哪些图像格式？**
A: 支持所有 PIL 支持的格式（JPG、PNG、BMP、TIFF等）。系统会自动转换为 RGB 模式。

**Q: 处理后的图像质量如何？**
A: 使用高质量模式可以获得最佳效果。输出图像会保持原始尺寸，使用 LANCZOS 插值算法保证质量。
