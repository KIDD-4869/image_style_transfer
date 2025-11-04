#!/usr/bin/env python3
"""
分析宫崎骏风格图片特点
"""

import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def analyze_ghibli_images():
    """分析宫崎骏风格图片的特点"""
    
    ghibli_dir = "ghibli_images"
    
    if not os.path.exists(ghibli_dir):
        print("❌ 宫崎骏风格图片目录不存在")
        return
    
    image_files = [f for f in os.listdir(ghibli_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("❌ 没有找到宫崎骏风格图片")
        return
    
    print(f"🎨 找到 {len(image_files)} 张宫崎骏风格图片")
    
    # 分析每张图片的特点
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(ghibli_dir, image_file)
        
        print(f"\n📊 分析图片 {i+1}: {image_file}")
        
        try:
            # 加载图片
            img = cv2.imread(image_path)
            if img is None:
                print(f"❌ 无法加载图片: {image_path}")
                continue
            
            # 转换为RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 分析图片特点
            h, w = img.shape[:2]
            print(f"📏 尺寸: {w}x{h}")
            
            # 分析色彩特点
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            
            print(f"🎨 色彩分析:")
            print(f"   - 平均饱和度: {np.mean(s):.1f}")
            print(f"   - 平均亮度: {np.mean(v):.1f}")
            print(f"   - 色调分布: {np.histogram(h, bins=12)[0]}")
            
            # 分析边缘特点
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (w * h)
            print(f"✏️ 边缘密度: {edge_density:.4f}")
            
            # 分析颜色数量（简化程度）
            img_flat = img.reshape(-1, 3)
            unique_colors = len(np.unique(img_flat, axis=0))
            print(f"🎨 独特颜色数量: {unique_colors}")
            
            # 分析颜色分布
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            print(f"🌈 LAB色彩空间:")
            print(f"   - L(亮度)范围: {l.min()}-{l.max()}")
            print(f"   - A(红绿)范围: {a.min()}-{a.max()}")
            print(f"   - B(黄蓝)范围: {b.min()}-{b.max()}")
            
            # 显示图片
            plt.figure(figsize=(12, 6))
            
            # 原图
            plt.subplot(2, 3, 1)
            plt.imshow(img_rgb)
            plt.title(f'原图 {image_file}')
            plt.axis('off')
            
            # 饱和度图
            plt.subplot(2, 3, 2)
            plt.imshow(s, cmap='viridis')
            plt.title('饱和度')
            plt.axis('off')
            
            # 亮度图
            plt.subplot(2, 3, 3)
            plt.imshow(v, cmap='gray')
            plt.title('亮度')
            plt.axis('off')
            
            # 边缘图
            plt.subplot(2, 3, 4)
            plt.imshow(edges, cmap='gray')
            plt.title('边缘检测')
            plt.axis('off')
            
            # 色调分布
            plt.subplot(2, 3, 5)
            plt.hist(h.ravel(), bins=12, range=[0, 180], alpha=0.7)
            plt.title('色调分布')
            plt.xlabel('色调值')
            plt.ylabel('像素数量')
            
            # 颜色简化程度
            plt.subplot(2, 3, 6)
            # 显示颜色简化的版本
            simplified = cv2.pyrMeanShiftFiltering(img, 20, 40)
            simplified_rgb = cv2.cvtColor(simplified, cv2.COLOR_BGR2RGB)
            plt.imshow(simplified_rgb)
            plt.title('颜色简化效果')
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"❌ 分析图片时出错: {e}")

def extract_ghibli_style_features():
    """提取宫崎骏风格特征"""
    
    ghibli_dir = "ghibli_images"
    image_files = [f for f in os.listdir(ghibli_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    style_features = {
        'saturation_range': [],
        'brightness_range': [],
        'edge_density': [],
        'color_simplification': [],
        'color_palette': []
    }
    
    for image_file in image_files:
        image_path = os.path.join(ghibli_dir, image_file)
        
        try:
            img = cv2.imread(image_path)
            if img is None:
                continue
            
            # 分析色彩特点
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            
            style_features['saturation_range'].append(np.mean(s))
            style_features['brightness_range'].append(np.mean(v))
            
            # 分析边缘特点
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            h, w = img.shape[:2]
            edge_density = np.sum(edges > 0) / (w * h)
            style_features['edge_density'].append(edge_density)
            
            # 分析颜色简化程度
            img_flat = img.reshape(-1, 3)
            unique_colors = len(np.unique(img_flat, axis=0))
            total_pixels = w * h
            color_simplification = unique_colors / total_pixels
            style_features['color_simplification'].append(color_simplification)
            
        except Exception as e:
            print(f"❌ 分析 {image_file} 时出错: {e}")
    
    # 计算平均特征
    avg_features = {}
    for key, values in style_features.items():
        if values:
            avg_features[key] = np.mean(values)
            print(f"📊 {key}: {avg_features[key]:.4f}")
    
    return avg_features

if __name__ == '__main__':
    print("=" * 60)
    print("🎨 宫崎骏风格图片分析")
    print("=" * 60)
    
    # 分析单张图片特点
    analyze_ghibli_images()
    
    print("\n" + "=" * 60)
    print("📊 宫崎骏风格特征提取")
    print("=" * 60)
    
    # 提取整体风格特征
    features = extract_ghibli_style_features()
    
    print("\n" + "=" * 60)
    print("🎯 宫崎骏风格特点总结")
    print("=" * 60)
    print("根据分析，宫崎骏风格可能具有以下特点:")
    print("1. 中等饱和度，色彩鲜艳但不刺眼")
    print("2. 较高的亮度，画面明亮")
    print("3. 清晰的边缘线条")
    print("4. 适度的颜色简化")
    print("5. 温暖柔和的色调")
    print("=" * 60)