#!/usr/bin/env python3
"""
自动测试脚本 - 无需确认直接运行
"""

import os
import sys
import subprocess
import time

def run_command(cmd, description):
    """运行命令并显示结果"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(f"✅ {description} 成功")
            return True
        else:
            print(f"❌ {description} 失败: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} 超时")
        return False
    except Exception as e:
        print(f"❌ {description} 异常: {e}")
        return False

def test_environment():
    """测试环境"""
    print("🧪 环境测试")
    
    # 测试Python导入
    test_code = '''
import cv2, torch, flask
from PIL import Image
import psutil, lz4
print("All dependencies imported successfully")
'''
    
    return run_command(f'python3 -c "{test_code}"', "依赖检查")

def test_app_import():
    """测试应用导入"""
    print("📱 应用测试")
    
    test_code = '''
import sys
sys.path.insert(0, ".")
import app
print("App imported successfully")
'''
    
    return run_command(f'python3 -c "{test_code}"', "应用导入")

def test_ghibli_processor():
    """测试宫崎骏处理器"""
    print("🎨 宫崎骏处理器测试")
    
    test_code = '''
import sys
sys.path.insert(0, ".")
from core.true_ghibli_processor import TrueGhibliProcessor
from PIL import Image
import numpy as np

processor = TrueGhibliProcessor()
test_img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
result = processor.process(test_img)
print(f"Processor test: {'success' if result.success else 'failed'}")
'''
    
    return run_command(f'python3 -c "{test_code}"', "宫崎骏处理器")

def main():
    """主函数"""
    print("🚀 自动测试开始")
    print("=" * 50)
    
    results = []
    
    # 创建目录
    os.makedirs('logs', exist_ok=True)
    os.makedirs('cache', exist_ok=True)
    os.makedirs('static/uploads', exist_ok=True)
    print("✅ 目录创建完成")
    
    # 运行测试
    results.append(test_environment())
    results.append(test_app_import())
    results.append(test_ghibli_processor())
    
    # 总结
    print("\n" + "=" * 50)
    success_count = sum(results)
    total_count = len(results)
    
    if success_count == total_count:
        print("🎉 所有测试通过！系统就绪")
        print("🌐 启动命令: python3 app.py")
        print("🌐 访问地址: http://localhost:5003")
        return True
    else:
        print(f"❌ {total_count - success_count}/{total_count} 测试失败")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)