#!/bin/bash

# 快速设置脚本
set -e

echo "🔧 检查系统环境..."

# 检查Docker
if ! command -v docker &> /dev/null; then
    echo "❌ 请先安装Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# 检查Docker Compose
if command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
elif docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
else
    echo "❌ Docker Compose未安装"
    echo "💡 安装方法："
    echo "   macOS: brew install docker-compose"
    echo "   或使用Docker Desktop内置的compose"
    exit 1
fi

echo "✅ 使用命令: $COMPOSE_CMD"

# 创建目录
mkdir -p logs cache models static/uploads

# 启动服务
echo "🚀 启动服务..."
$COMPOSE_CMD up -d

echo "✅ 设置完成！访问: http://localhost:5003"