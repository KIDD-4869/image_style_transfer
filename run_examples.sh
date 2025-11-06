#!/bin/bash

# 改进后的自动学习脚本使用示例

echo "🎨 宫崎骏风格自动学习脚本 - 使用示例"
echo "=========================================="

echo ""
echo "1. 🚀 快速测试（少量照片，快速验证）"
echo "python real_auto_learning.py --photo-count 10 --epochs 5"

echo ""
echo "2. 📊 标准训练（默认100张照片）"
echo "python real_auto_learning.py"

echo ""
echo "3. 🌐 大规模训练（500张照片，50轮训练）"
echo "python real_auto_learning.py --photo-count 500 --epochs 50"

echo ""
echo "4. 🔧 离线训练（不下载，仅使用现有照片和数据增强）"
echo "python real_auto_learning.py --photo-count 50 --no-download"

echo ""
echo "5. ⚡ 快速实验（自定义配置文件）"
echo "python real_auto_learning.py --config custom_config.json --photo-count 20 --epochs 10"

echo ""
echo "📋 查看完整帮助："
echo "python real_auto_learning.py --help"

echo ""
echo "💡 提示："
echo "- 照片数量越多，训练效果通常越好，但训练时间也越长"
echo "- 启用下载功能需要网络连接"
echo "- 使用 --no-download 可以在没有网络时进行训练"
echo "- 训练日志保存在 training_logs/ 目录"