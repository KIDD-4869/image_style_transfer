#!/usr/bin/env python3
"""
快速测试脚本
"""

import sys
import os

def main():
    print("🚀 快速测试开始")
    
    # 测试1: 依赖检查
    try:
        import cv2, torch, flask
        from PIL import Image
        import psutil, lz4
        print("✅ 所有依赖导入成功")
    except ImportError as e:
        print(f"❌ 依赖缺失: {e}")
        return False
    
    # 测试2: 应用导入
    try:
        sys.path.insert(0, ".")
        import app
        print("✅ 应用导入成功")
    except Exception as e:
        print(f"❌ 应用导入失败: {e}")
        return False
    
    # 测试3: 宫崎骏处理器
    try:
        from core.true_ghibli_processor import TrueGhibliProcessor
        import numpy as np
        
        processor = TrueGhibliProcessor()
        test_img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        result = processor.process(test_img)
        
        if result.success:
            print("✅ 宫崎骏处理器测试成功")
        else:
            print(f"❌ 宫崎骏处理器测试失败: {result.error_message}")
            return False
            
    except Exception as e:
        print(f"❌ 宫崎骏处理器异常: {e}")
        return False
    
    print("🎉 所有测试通过！")
    print("🌐 启动命令: python3 app.py")
    print("🌐 访问地址: http://localhost:5003")
    return True

if __name__ == '__main__':
    success = main()
    if not success:
        sys.exit(1)