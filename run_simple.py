#!/usr/bin/env python3
"""
简化版启动脚本 - 不依赖Docker
"""

import sys
import os
import subprocess

def check_dependencies():
    """检查依赖"""
    required = ['torch', 'torchvision', 'opencv-python', 'flask', 'pillow']
    missing = []
    
    for pkg in required:
        try:
            __import__(pkg.replace('-', '_'))
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"❌ 缺少依赖: {', '.join(missing)}")
        print("💡 安装命令: pip install " + ' '.join(missing))
        return False
    return True

def main():
    print("🎨 宫崎骏风格图片转换器")
    
    if not check_dependencies():
        sys.exit(1)
    
    # 创建目录
    os.makedirs('logs', exist_ok=True)
    os.makedirs('cache', exist_ok=True)
    os.makedirs('static/uploads', exist_ok=True)
    
    print("✅ 依赖检查通过")
    print("🚀 启动应用...")
    print("🌐 访问地址: http://localhost:5003")
    
    # 启动应用
    try:
        if os.path.exists('app_optimized.py'):
            subprocess.run([sys.executable, 'app_optimized.py'])
        else:
            subprocess.run([sys.executable, 'app.py'])
    except KeyboardInterrupt:
        print("\n👋 应用已停止")

if __name__ == '__main__':
    main()