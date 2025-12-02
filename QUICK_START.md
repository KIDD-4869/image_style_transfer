# 🚀 快速开始指南

## 安装和运行

### 1. 安装依赖

```bash
pip3 install -r requirements.txt
```

### 2. 启动应用

```bash
python3 run.py
```

应用将自动在浏览器中打开：http://localhost:5003

---

## 使用方法

### Web界面

1. 打开浏览器访问 http://localhost:5003
2. 点击上传区域或拖拽图片
3. 选择处理策略：
   - **快速模式**: 5-10秒，基础色彩调整
   - **平衡模式**: 15-30秒，标准宫崎骏风格（推荐）
   - **高质量模式**: 30-60秒，完整风格转换
4. 点击"开始转换"
5. 等待处理完成
6. 下载结果图片

### API使用

#### 上传图片

```bash
curl -X POST http://localhost:5003/upload \
  -F "file=@your_image.jpg" \
  -F "style_strategy=balanced"
```

响应：
```json
{
  "success": true,
  "task_id": "1732588800000",
  "message": "转换任务已开始"
}
```

#### 查询进度

```bash
curl http://localhost:5003/progress/{task_id}
```

#### 获取结果

```bash
curl http://localhost:5003/result/{task_id}
```

#### 健康检查

```bash
curl http://localhost:5003/health
```

---

## Python代码示例

```python
from core.processors import GhibliProcessor, ProcessingStrategy
from PIL import Image

# 创建处理器
processor = GhibliProcessor()

# 加载图像
image = Image.open("input.jpg")

# 处理图像（平衡模式）
result = processor.process(image, strategy=ProcessingStrategy.BALANCED)

# 保存结果
if result.success:
    result.image.save("output.jpg")
    print(f"✅ 处理成功，耗时: {result.processing_time:.2f}秒")
else:
    print(f"❌ 处理失败: {result.error_message}")
```

---

## 处理策略对比

| 策略 | 速度 | 质量 | 适用场景 |
|------|------|------|----------|
| **FAST** | ⚡ 5-10秒 | ⭐⭐⭐ | 快速预览 |
| **BALANCED** | ⚡⚡ 15-30秒 | ⭐⭐⭐⭐ | 日常使用（推荐） |
| **QUALITY** | ⚡⚡⚡ 30-60秒 | ⭐⭐⭐⭐⭐ | 高质量输出 |

---

## 系统要求

- Python 3.7+
- 2GB+ RAM
- 支持的图片格式：JPG, PNG, BMP, GIF
- 最大文件大小：20MB
- 推荐图片尺寸：≤2048px

---

## 故障排除

### 问题1: 依赖安装失败

```bash
# 升级pip
python3 -m pip install --upgrade pip

# 重新安装
pip3 install -r requirements.txt
```

### 问题2: 端口被占用

修改 `app.py` 中的端口：
```python
app.run(debug=True, host='0.0.0.0', port=5004)  # 改为5004
```

### 问题3: 处理速度慢

- 使用FAST模式
- 减小图片尺寸
- 确保有足够的内存

---

## 更多信息

- 📖 完整文档：[DOCUMENTATION.md](DOCUMENTATION.md)
- 🏗️ 架构说明：[ARCHITECTURE_REVIEW.md](ARCHITECTURE_REVIEW.md)
- 🔄 重构总结：[REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md)
- 📋 实施计划：[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)

---

**享受创造宫崎骏风格图片的乐趣！** 🎨✨
