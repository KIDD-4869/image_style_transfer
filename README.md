# 🎨 宫崎骏风格图像转换器

一个基于 Python 的高性能 Web 应用，使用 Stable Diffusion AI 将普通图片转换为宫崎骏动画风格。

[![Python版本](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org)
[![许可证](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## ✨ 功能特点

- 🎨 **Enhanced 模式**：基于 Stable Diffusion + ControlNet 的专业级转换
- 🤖 **AnimeGAN 模式**：深度学习动漫风格转换
- ⚡ **CV 模式**：快速传统图像处理
- 🚀 **模型缓存**：首次加载后，后续处理速度提升 50 倍
- 📊 **实时进度**：详细的处理阶段显示
- 🎯 **内容保留**：优化的参数确保保留原图主体
- 💾 **一键下载**：轻松保存转换后的图片
- 📱 **响应式界面**：支持桌面和移动设备

## 🏗️ 技术架构

### 核心技术栈
- **Flask**: Web 框架
- **Stable Diffusion 1.5**: AI 图像生成
- **ControlNet**: 结构保持
- **PyTorch**: 深度学习框架
- **Diffusers**: Hugging Face 扩散模型库
- **OpenCV**: 图像处理
- **PIL/Pillow**: 图像处理

### AI 模型
- **Stable Diffusion**: runwayml/stable-diffusion-v1-5
- **ControlNet**: lllyasviel/control_v11p_sd15_canny
- **AnimeGAN**: AnimeGAN v2

## 🚀 快速开始

### 1. 克隆项目
```bash
git clone <repository-url>
cd image_style_transfer
```

### 2. 创建虚拟环境
```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# 或
.venv\Scripts\activate  # Windows
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 启动应用
```bash
# 设置镜像（可选，加速模型下载）
export HF_ENDPOINT=https://hf-mirror.com

# 启动服务
python app.py
```

### 5. 访问应用
打开浏览器访问：http://127.0.0.1:5003

## 📖 使用说明

### 基本使用流程

1. **上传图片**
   - 点击上传区域或拖拽图片文件
   - 支持 JPG、PNG、BMP、GIF 格式
   - 最大文件大小：16MB

2. **选择处理模式**
   - ✨ **Enhanced 高质量**：最佳效果，约 3 分钟
   - 🎯 **Enhanced 平衡**：平衡速度和质量，约 2 分钟
   - ⚡ **Enhanced 快速**：快速预览，约 1 分钟
   - 🚀 **Enhanced 超高质量**：极致效果，约 4 分钟

3. **开始转换**
   - 点击"开始转换"按钮
   - 实时查看详细进度
   - 等待处理完成

4. **查看和下载结果**
   - 对比原图和转换后的效果
   - 点击"下载结果"保存图片

### Enhanced 模式说明

#### 首次使用
- 首次使用需要下载模型（约 4GB）
- 下载时间取决于网络速度（国内镜像约 5-10 分钟）
- 模型下载后会缓存，后续使用无需重新下载

#### 处理时间
| 模式 | 首次处理 | 后续处理 |
|------|----------|----------|
| Enhanced 快速 | ~1.5 分钟 | ~1 分钟 |
| Enhanced 平衡 | ~2.5 分钟 | ~2 分钟 |
| Enhanced 高质量 | ~3.5 分钟 | ~3 分钟 |
| Enhanced 超高质量 | ~4.5 分钟 | ~4 分钟 |

**注意**: 首次处理包含模型加载时间（20-50秒），后续处理使用缓存的模型（< 1秒）

#### 参数说明

**Strength（重绘强度）**:
- Fast: 0.30 - 轻微风格化，保留大部分原图
- Balanced: 0.35 - 适度风格化，平衡内容和风格
- Quality: 0.40 - 明显风格化，保留主体内容 ✅
- Ultra: 0.45 - 较强风格化，艺术创作

**ControlNet**:
- 启用边缘检测保持图像结构
- 防止内容被完全改变
- 确保狗狗还是狗狗，不会变成其他东西

### 处理策略对比

| 策略 | 处理器 | 速度 | 质量 | 推荐场景 |
|------|--------|------|------|----------|
| CV Fast | OpenCV | ⚡⚡⚡ | ⭐⭐ | 快速预览 |
| AnimeGAN Balanced | AnimeGAN | ⚡⚡ | ⭐⭐⭐⭐ | 日常使用 |
| **Enhanced Quality** | **Stable Diffusion** | ⚡ | **⭐⭐⭐⭐⭐** | **最佳质量** |

## 🎯 缓存策略

### ✅ 模型缓存（已启用）
- AI 模型常驻内存
- 首次加载后，后续使用秒级响应
- 速度提升 50 倍

### ❌ 结果缓存（已禁用）
- 每次都重新生成
- 保持 AI 创作的随机性和多样性
- 同一张图片多次转换会得到不同效果

**原则**: 缓存工具（模型），不缓存作品（结果）

## 📡 API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 主页 |
| `/health` | GET | 健康检查 |
| `/stats` | GET | 系统统计 |
| `/config` | GET | 配置查询 |
| `/model/status` | GET | 模型缓存状态 |
| `/model/clear` | POST | 清除模型缓存 |
| `/upload` | POST | 上传图片 |
| `/progress/<id>` | GET | 查询进度 |
| `/result/<id>` | GET | 获取结果 |

### API 使用示例

#### 上传图片
```bash
curl -X POST http://localhost:5003/upload \
  -F "file=@your_image.jpg" \
  -F "style_strategy=enhanced_quality"
```

#### 查询进度
```bash
curl http://localhost:5003/progress/{task_id}
```

#### 获取结果
```bash
curl http://localhost:5003/result/{task_id} -o result.jpg
```

## 🔧 配置

### 环境变量

```bash
# 应用配置
PORT=5003
DEBUG=false

# 缓存配置
CACHE_ENABLED=true
CACHE_MEMORY_SIZE=100  # MB
CACHE_DISK_SIZE=1000   # MB
```

### 参数调整

编辑 `config/enhanced_processor_config.yaml`:

```yaml
modes:
  quality:
    strength: 0.40  # 重绘强度 (0.3-0.5)
    num_inference_steps: 60  # 推理步数 (20-100)
    guidance_scale: 8.0  # 风格强度 (7.0-9.0)
```

## 📊 性能指标

### 处理速度

| 模式 | 首次处理 | 后续处理 | 改进 |
|------|----------|----------|------|
| Enhanced Fast | ~1.5 分钟 | ~1 分钟 | ⚡ |
| Enhanced Balanced | ~2.5 分钟 | ~2 分钟 | ⚡ |
| Enhanced Quality | ~3.5 分钟 | ~3 分钟 | ⚡ |
| Enhanced Ultra | ~4.5 分钟 | ~4 分钟 | ⚡ |

**注意**: 首次处理包含模型加载时间（20-50秒），后续处理使用缓存的模型（< 1秒）

### 模型缓存效果

| 指标 | 首次 | 后续 | 提升 |
|------|------|------|------|
| 模型加载 | 20-50秒 | < 1秒 | 50倍 ⚡ |

## 🐛 故障排除

### 端口被占用
```bash
# 修改端口
export PORT=8080
python app.py
```

### 模型下载失败
```bash
# 使用国内镜像
export HF_ENDPOINT=https://hf-mirror.com
python app.py
```

### 内存不足
```bash
# 清除模型缓存
curl -X POST http://localhost:5003/model/clear
```

### 查看日志
```bash
tail -f logs/enhanced_processor.log
```

## 📁 项目结构

```
image_style_transfer/
├── app.py                 # Flask 主应用
├── run.py                 # 运行脚本
├── requirements.txt       # 依赖列表
├── core/                  # 核心处理模块
│   ├── processors/        # 图像处理器
│   └── components/        # Enhanced 组件
├── utils/                 # 工具模块
├── config/                # 配置文件
├── static/                # 静态资源
├── templates/             # HTML 模板
└── models/                # AI 模型文件
```

## 📚 文档

- **README.md**: 项目介绍和快速开始（本文档）
- **DOCUMENTATION.md**: 详细技术文档
- **IMPLEMENTATION_PLAN.md**: 实现计划
- **PROJECT_STATUS.md**: 项目状态

## 🤝 贡献

欢迎贡献！请：
1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📝 更新日志

### v2.0.0 (2024-12-02)
- ✅ 实现 Enhanced 模式（Stable Diffusion + ControlNet）
- ✅ 优化内容识别（降低 Strength 参数）
- ✅ 实现全局模型缓存（速度提升 50 倍）
- ✅ 优化进度显示（详细的阶段信息）
- ✅ 优化前端 UI（卡片式设计）
- ✅ 完善缓存策略（缓存模型，不缓存结果）

### v1.0.0 (2024-11-26)
- ✅ 基础功能实现
- ✅ CV 和 AnimeGAN 处理器
- ✅ Web 界面

## 📄 许可证

MIT License

## 🙏 致谢

感谢所有为项目做出贡献的开发者！

---

**享受创造宫崎骏风格图片的乐趣！** 🎨✨

**项目状态**: ✅ 生产就绪 | **版本**: 2.0 | **性能**: 优秀
