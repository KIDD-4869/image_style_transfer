#!/usr/bin/env python3
"""
真正的宫崎骏风格转换器 - 基于参考图片分析的精确风格复现
"""

import cv2
import numpy as np
from PIL import Image
import os
from typing import Optional

class TrueGhibliStyleTransfer:
    """真正的宫崎骏风格转换器"""
    
    def __init__(self):
        self.progress_callback = None
        self.task_id = None
        
    def set_progress_callback(self, callback, task_id):
        """设置进度回调"""
        self.progress_callback = callback
        self.task_id = task_id
        
    def apply_ghibli_style(self, image: Image.Image) -> Image.Image:
        """应用宫崎骏风格转换 - 结合真正的动漫化效果"""
        print("🎨 开始真正的宫崎骏风格转换...")
        
        # 转换为OpenCV格式
        img_np = np.array(image)
        if img_np.ndim == 3 and img_np.shape[2] == 3:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
            
        # 更新进度
        self._update_progress(10)
        
        # 1. 真正的动漫化 - 创建扁平色块
        anime_base = self._create_anime_base(img_bgr)
        self._update_progress(30)
        
        # 2. 宫崎骏色彩风格
        ghibli_colored = self._apply_ghibli_colors(anime_base)
        self._update_progress(60)
        
        # 3. 清晰的动漫轮廓线
        final = self._add_clean_anime_lines(ghibli_colored, img_bgr)
        self._update_progress(100)
        
        # 转换回RGB
        result_rgb = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result_rgb)
    
    def _create_anime_base(self, img_bgr):
        """创建真正的动漫基础 - 扁平色块效果"""
        # 1. 强力平滑，移除所有纹理细节
        smooth = img_bgr.copy()
        
        # 多次双边滤波，彻底移除纹理
        smooth = cv2.bilateralFilter(smooth, 15, 100, 100)
        smooth = cv2.bilateralFilter(smooth, 9, 80, 80)
        
        # 2. 颜色量化 - 创建扁平色块
        data = smooth.reshape((-1, 3))
        data = np.float32(data)
        
        # K-means聚类，大幅减少颜色数量
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
        K = 12  # 保留更多颜色，但仍然创造扁平效果
        
        _, labels, centers = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        quantized_data = centers[labels.flatten()]
        quantized = quantized_data.reshape(smooth.shape)
        
        # 3. 进一步平滑色块边界
        quantized = cv2.medianBlur(quantized, 3)
        
        return quantized
    
    def _apply_ghibli_colors(self, img_bgr):
        """应用宫崎骏色彩风格"""
        # 转换到HSV空间
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # 宫崎骏风格：温暖、明亮、饱和
        s = np.clip(s * 1.3, 0, 255).astype(np.uint8)  # 增强饱和度
        v = np.clip(v * 1.1, 0, 255).astype(np.uint8)  # 增强亮度
        
        # 色调偏向温暖
        h = np.where((h >= 10) & (h <= 40), np.clip(h + 5, 0, 179), h)
        
        # 合并通道
        hsv_enhanced = cv2.merge([h, s, v])
        enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
        
        # LAB空间微调
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 增强色彩鲜艳度
        a = np.clip(a + 10, 0, 255).astype(np.uint8)
        b = np.clip(b + 15, 0, 255).astype(np.uint8)
        
        lab_enhanced = cv2.merge([l, a, b])
        result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        return result
    
    def _add_clean_anime_lines(self, img_bgr, original_bgr):
        """添加清晰的动漫轮廓线"""
        # 生成清晰的边缘
        gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
        
        # 高斯模糊，为边缘检测做准备
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Canny边缘检测 - 调整参数获得清晰边缘
        edges = cv2.Canny(blurred, 30, 80)
        
        # 形态学操作，连接断裂的边缘
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # 轻微膨胀，让线条更明显
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # 在边缘位置直接绘制黑线
        result = img_bgr.copy()
        mask = edges > 0
        result[mask] = [0, 0, 0]  # 黑色轮廓线
        
        return result
    

    
    def _update_progress(self, progress):
        """更新进度"""
        if self.progress_callback and self.task_id:
            self.progress_callback(self.task_id, progress, progress//10, 10, 0)

# 全局实例
true_ghibli_processor = TrueGhibliStyleTransfer()