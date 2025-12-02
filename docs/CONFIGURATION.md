# 配置管理指南

## 概述

应用使用统一的配置管理系统，支持通过环境变量和代码两种方式配置。配置系统提供参数验证、默认值和单例模式。

## 配置方式

### 1. 环境变量（推荐）

创建`.env`文件或设置环境变量：

```bash
# 复制示例文件
cp .env.example .env

# 编辑配置
vim .env
```

### 2. 代码配置

```python
from config import get_config

config = get_config()
# 配置会自动从环境变量加载
```

## 配置参数

### 应用基本配置

| 参数 | 环境变量 | 默认值 | 说明 |
|------|---------|--------|------|
| app_name | - | "Ghibli Style Transfer" | 应用名称 |
| app_version | - | "1.0.0" | 应用版本 |
| debug | DEBUG | false | 调试模式 |
| host | HOST | "0.0.0.0" | 监听地址 |
| port | PORT | 5003 | 监听端口 |

**示例**:
```bash
export DEBUG=true
export HOST=127.0.0.1
export PORT=8080
```

### 上传配置

| 参数 | 环境变量 | 默认值 | 说明 |
|------|---------|--------|------|
| upload_folder | UPLOAD_FOLDER | "static/uploads" | 上传目录 |
| max_content_length | MAX_CONTENT_LENGTH | 20971520 | 最大文件大小（字节） |
| max_image_size | MAX_IMAGE_SIZE | 2048 | 最大图像尺寸（像素） |

**示例**:
```bash
export UPLOAD_FOLDER=/data/uploads
export MAX_CONTENT_LENGTH=52428800  # 50MB
export MAX_IMAGE_SIZE=4096
```

### 处理配置

| 参数 | 环境变量 | 默认值 | 说明 |
|------|---------|--------|------|
| max_concurrent_tasks | MAX_CONCURRENT_TASKS | 3 | 最大并发任务数 |
| default_strategy | DEFAULT_STRATEGY | "balanced" | 默认处理策略 |

**示例**:
```bash
export MAX_CONCURRENT_TASKS=5
export DEFAULT_STRATEGY=quality
```

**策略选项**:
- `fast`: 快速处理（5-10秒）
- `balanced`: 平衡模式（15-30秒）
- `quality`: 高质量（30-60秒）

### 任务管理配置

| 参数 | 环境变量 | 默认值 | 说明 |
|------|---------|--------|------|
| task_max_tasks | TASK_MAX_TASKS | 100 | 最大任务数 |
| task_ttl_hours | TASK_TTL_HOURS | 24 | 任务生存时间（小时） |

**示例**:
```bash
export TASK_MAX_TASKS=200
export TASK_TTL_HOURS=48
```

### 缓存配置

| 参数 | 环境变量 | 默认值 | 说明 |
|------|---------|--------|------|
| cache_enabled | CACHE_ENABLED | true | 是否启用缓存 |
| cache_memory_size_mb | CACHE_MEMORY_SIZE | 100 | 内存缓存大小（MB） |
| cache_disk_size_mb | CACHE_DISK_SIZE | 1000 | 磁盘缓存大小（MB） |
| cache_dir | CACHE_DIR | "./cache" | 缓存目录 |
| cache_ttl_hours | CACHE_TTL_HOURS | 24 | 缓存生存时间（小时） |
| cache_enable_disk | CACHE_ENABLE_DISK | true | 是否启用磁盘缓存 |

**示例**:
```bash
export CACHE_ENABLED=true
export CACHE_MEMORY_SIZE=200
export CACHE_DISK_SIZE=5000
export CACHE_DIR=/data/cache
export CACHE_TTL_HOURS=48
```

## 配置验证

配置系统会自动验证参数：

### 端口验证
- 推荐范围: 1024-65535
- 超出范围会记录警告

### 图像大小验证
- 最小值: 256像素
- 推荐最大值: 4096像素
- 超过推荐值会记录警告

### 并发任务验证
- 最小值: 1
- 推荐最大值: 10
- 超过推荐值会记录警告

### 缓存大小验证
- 内存缓存最小推荐: 10MB
- 磁盘缓存最小推荐: 100MB
- 低于推荐值会记录警告

## API端点

### 获取配置

```bash
GET /config
```

**响应示例**:
```json
{
  "success": true,
  "config": {
    "app_name": "Ghibli Style Transfer",
    "app_version": "1.0.0",
    "host": "0.0.0.0",
    "port": 5003,
    "upload_folder": "static/uploads",
    "max_content_length": 20971520,
    "max_image_size": 2048,
    "max_concurrent_tasks": 3,
    "default_strategy": "balanced",
    "task_max_tasks": 100,
    "task_ttl_hours": 24,
    "cache_enabled": true,
    "cache_memory_size_mb": 100,
    "cache_disk_size_mb": 1000,
    "cache_dir": "./cache",
    "cache_ttl_hours": 24,
    "cache_enable_disk": true
  }
}
```

## 使用场景

### 开发环境

```bash
# .env
DEBUG=true
HOST=127.0.0.1
PORT=5003
CACHE_MEMORY_SIZE=50
CACHE_DISK_SIZE=500
```

### 生产环境

```bash
# .env
DEBUG=false
HOST=0.0.0.0
PORT=5003
MAX_CONCURRENT_TASKS=5
CACHE_MEMORY_SIZE=500
CACHE_DISK_SIZE=5000
CACHE_TTL_HOURS=48
```

### 高性能环境

```bash
# .env
MAX_CONCURRENT_TASKS=10
MAX_IMAGE_SIZE=4096
CACHE_MEMORY_SIZE=1000
CACHE_DISK_SIZE=10000
```

### 低资源环境

```bash
# .env
MAX_CONCURRENT_TASKS=2
MAX_IMAGE_SIZE=1024
CACHE_MEMORY_SIZE=50
CACHE_DISK_SIZE=500
CACHE_ENABLED=false  # 禁用缓存以节省资源
```

## 编程接口

### 获取配置

```python
from config import get_config

config = get_config()
print(f"应用运行在端口: {config.port}")
print(f"缓存启用: {config.cache_enabled}")
```

### 重新加载配置

```python
from config import reload_config

# 修改环境变量后重新加载
config = reload_config()
```

### 转换为字典

```python
config = get_config()
config_dict = config.to_dict()
print(config_dict)
```

## 最佳实践

### 1. 使用环境变量

推荐使用环境变量而不是硬编码配置：

```bash
# 好的做法
export PORT=8080
python app.py

# 不推荐
# 直接修改代码中的配置
```

### 2. 不同环境使用不同配置

```bash
# 开发环境
cp .env.development .env

# 生产环境
cp .env.production .env
```

### 3. 敏感信息不要提交到版本控制

```bash
# .gitignore
.env
.env.local
.env.*.local
```

### 4. 定期检查配置

```bash
# 查看当前配置
curl http://localhost:5003/config
```

### 5. 监控配置影响

根据实际使用情况调整配置：

- 监控缓存命中率，调整缓存大小
- 监控并发任务数，调整max_concurrent_tasks
- 监控内存使用，调整cache_memory_size_mb

## 故障排除

### 配置未生效

**问题**: 修改环境变量后配置未生效

**解决**:
```bash
# 确保环境变量已设置
echo $PORT

# 重启应用
python app.py
```

### 端口被占用

**问题**: 端口5003已被占用

**解决**:
```bash
# 使用不同端口
export PORT=8080
python app.py
```

### 缓存目录权限问题

**问题**: 无法创建缓存目录

**解决**:
```bash
# 创建目录并设置权限
mkdir -p ./cache
chmod 755 ./cache

# 或使用不同目录
export CACHE_DIR=/tmp/cache
```

### 内存不足

**问题**: 缓存占用内存过多

**解决**:
```bash
# 减少内存缓存
export CACHE_MEMORY_SIZE=50

# 或禁用缓存
export CACHE_ENABLED=false
```

## 配置模板

### 小型应用

```bash
MAX_CONCURRENT_TASKS=2
CACHE_MEMORY_SIZE=50
CACHE_DISK_SIZE=500
TASK_MAX_TASKS=50
```

### 中型应用

```bash
MAX_CONCURRENT_TASKS=3
CACHE_MEMORY_SIZE=100
CACHE_DISK_SIZE=1000
TASK_MAX_TASKS=100
```

### 大型应用

```bash
MAX_CONCURRENT_TASKS=5
CACHE_MEMORY_SIZE=500
CACHE_DISK_SIZE=5000
TASK_MAX_TASKS=200
```

## 总结

配置管理系统提供了灵活、可靠的配置方式：

✅ **环境变量支持**: 易于部署和管理  
✅ **参数验证**: 自动验证配置合理性  
✅ **默认值**: 开箱即用  
✅ **单例模式**: 全局统一配置  
✅ **API查询**: 运行时查看配置  

合理配置可以显著提升应用性能和稳定性！

---

**文档版本**: 1.0.0  
**最后更新**: 2025年12月1日
