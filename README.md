# Image Style Transfer

这是一个图像风格迁移项目，支持多种风格转换算法，包括吉卜力工作室风格、动漫风格等。

## 项目结构

- `core/`: 核心算法实现
- `models/`: 预训练模型和训练脚本
- `utils/`: 工具函数
- `config/`: 配置文件
- `tests/`: 测试代码
- `templates/`: 网页模板

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

```bash
python run.py --input input.jpg --style ghibli --output output.jpg
```

## 大型模型文件说明

由于GitHub对单个文件大小有限制（100MB），本项目使用的大型预训练模型文件不直接存储在仓库中。

大型模型文件列表：
- `models/anime_gan/AnimeGANv2_Hayao.pth` (~165MB)

要使用这些模型，请手动下载或使用Git LFS：

### 方法1：使用Git LFS（推荐）

```bash
# 安装Git LFS
git lfs install

# 下载LFS文件
git lfs pull
```

### 方法2：手动下载

请从以下链接下载所需模型文件并放置在对应目录中：
- [AnimeGANv2_Hayao.pth](https://github.com/) (需要补充实际下载链接)

## 训练新模型

```bash
python real_auto_learning.py
```

## API服务

启动Web服务：
```bash
python app.py
```

访问 http://localhost:5000 查看界面。
