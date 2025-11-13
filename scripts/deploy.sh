#!/bin/bash

# 部署脚本
set -e

echo "🚀 开始部署宫崎骏风格图片转换器..."

# 检查Docker是否安装
if ! command -v docker &> /dev/null; then
    echo "❌ Docker未安装，请先安装Docker"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose未安装，请先安装Docker Compose"
    exit 1
fi

# 设置环境变量
export COMPOSE_PROJECT_NAME=ghibli-style-transfer

# 创建必要的目录
echo "📁 创建必要的目录..."
mkdir -p logs cache models static/uploads monitoring

# 构建Docker镜像
echo "🔨 构建Docker镜像..."
docker-compose build --no-cache

# 启动服务
echo "🚀 启动服务..."
docker-compose up -d

# 等待服务启动
echo "⏳ 等待服务启动..."
sleep 30

# 健康检查
echo "🔍 执行健康检查..."
if curl -f http://localhost:5003/health > /dev/null 2>&1; then
    echo "✅ 服务启动成功！"
    echo "🌐 应用访问地址: http://localhost:5003"
    echo "📊 健康检查: http://localhost:5003/health"
    echo "📈 系统指标: http://localhost:5003/metrics"
else
    echo "❌ 服务启动失败，请检查日志"
    docker-compose logs
    exit 1
fi

echo "🎉 部署完成！"