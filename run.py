#!/usr/bin/env python3
"""
宫崎骏风格图片转换器 - 启动脚本
"""

import sys
import webbrowser
import threading
import time
import subprocess

def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    print(f"Python版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("❌ Python版本过低，需要Python 3.7或更高版本")
        return False
    return True

def check_dependencies():
    """检查依赖是否已安装"""
    required_packages = [
        'cv2',           # opencv-python
        'numpy', 
        'PIL',           # Pillow
        'matplotlib',
        'torch',
        'torchvision',
        'skimage',       # scikit-image
        'requests',
        'flask'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
                missing_packages.append(package)
    
    if missing_packages:
        print("缺少以下依赖包:")
        for package in missing_packages:
            print(f"  - {package}")  # type: ignore
        print("\n请运行: pip3 install -r requirements.txt")
        return False
    
    return True

def open_browser():
    """在浏览器中打开应用"""
    time.sleep(2)  # 等待服务器启动
    webbrowser.open('http://localhost:5003')

def install_dependencies():
    """安装依赖包"""
    print("📦 正在安装依赖包...")
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ 依赖包安装成功")
            return True
        else:
            print("❌ 依赖包安装失败:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ 安装过程中出错: {e}")
        return False

def main():
    """主函数"""
    print("=" * 50)
    print("🎨 宫崎骏风格图片转换器")
    print("=" * 50)
    
    # 检查Python版本
    if not check_python_version():
        sys.exit(1)
    
    # 检查依赖
    if not check_dependencies():
        print("\n是否自动安装依赖包？(y/n): ")
        choice = input().strip().lower()
        if choice in ['y', 'yes', '是']:
            if not install_dependencies():
                sys.exit(1)
        else:
            print("请手动运行: pip3 install -r requirements.txt")
            sys.exit(1)
    
    # 重新检查依赖
    if not check_dependencies():
        print("❌ 依赖安装后仍然存在问题")
        sys.exit(1)
    
    # 导入并启动应用
    try:
        from app import app
        
        print("✅ 依赖检查通过")
        print("🚀 启动服务器...")
        print("📱 应用将在浏览器中自动打开")
        print("⏹️  按 Ctrl+C 停止服务器")
        print("-" * 50)
        
        # 在新线程中打开浏览器
        threading.Thread(target=open_browser, daemon=True).start()
        
        # 启动Flask应用
        app.run(debug=True, host='0.0.0.0', port=5003, use_reloader=False)
        
    except KeyboardInterrupt:
        print("\n👋 服务器已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()