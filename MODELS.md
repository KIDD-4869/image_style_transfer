# 大型模型文件说明

由于GitHub对单个文件有100MB的大小限制，我们使用Git LFS来管理大型模型文件。

## 安装和使用Git LFS

1. 安装Git LFS:
   ```bash
   git lfs install
   ```

2. 克隆仓库时自动下载LFS文件:
   ```bash
   git clone https://github.com/KIDD-4869/image_style_transfer.git
   ```

3. 或者在现有仓库中拉取LFS文件:
   ```bash
   git lfs pull
   ```

## 跟踪的文件类型

我们使用Git LFS跟踪以下类型的文件:
- `*.pth` - PyTorch模型文件
- `*.pt` - PyTorch模型文件
- `*.ckpt` - Checkpoint文件
- `*.h5` - HDF5模型文件

## 当前跟踪的大型文件

| 文件路径 | 大小 | 说明 |
|---------|------|------|
| models/anime_gan/AnimeGANv2_Hayao.pth | ~165MB | AnimeGANv2模型文件，用于动漫风格迁移 |

## 手动下载模型文件

如果不想使用Git LFS，可以从以下链接手动下载模型文件:

1. AnimeGANv2_Hayao.pth:
   - 下载链接: (需要补充实际下载链接)
   - 放置位置: `models/anime_gan/AnimeGANv2_Hayao.pth`

## 添加新的大型模型文件

如果需要添加新的大型模型文件到项目中:

1. 使用Git LFS跟踪该文件:
   ```bash
   git lfs track "models/new_model/new_model.pth"
   ```

2. 提交更改:
   ```bash
   git add .gitattributes
   git commit -m "Track new_model.pth with Git LFS"
   ```

3. 推送更改:
   ```bash
   git push origin main
   ```