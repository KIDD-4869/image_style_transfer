#!/usr/bin/env python3
"""
优化的宫崎骏风格转换器 - 解决线条密集和模糊问题
保持物体识别性，创造精细的动漫效果
"""

import cv2
import numpy as np
from PIL import Image
import os
from typing import Optional

class OptimizedGhibliStyleTransfer:
    """优化的宫崎骏风格转换器"""
    
    def __init__(self):
        self.progress_callback = None
        self.task_id = None
        
    def set_progress_callback(self, callback, task_id):
        """设置进度回调"""
        self.progress_callback = callback
        self.task_id = task_id
        
    def apply_ghibli_style(self, image: Image.Image) -> Image.Image:
        """应用优化的宫崎骏风格转换"""
        print("🎨 开始优化的宫崎骏风格转换...")
        
        # 转换为OpenCV格式
        img_np = np.array(image)
        if img_np.ndim == 3 and img_np.shape[2] == 3:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
            
        self._update_progress(10)
        
        # 1. 智能平滑 - 保持重要结构
        smoothed = self._intelligent_smoothing(img_bgr)
        self._update_progress(25)
        
        # 2. 适度颜色简化 - 保持识别性
        simplified = self._moderate_color_simplification(smoothed)
        self._update_progress(50)
        
        # 3. 选择性边缘增强 - 只保留重要轮廓
        enhanced = self._selective_edge_enhancement(simplified, img_bgr)
        self._update_progress(75)
        
        # 4. 宫崎骏色彩风格
        final = self._apply_ghibli_colors(enhanced)
        self._update_progress(100)
        
        # 转换回RGB
        result_rgb = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result_rgb)
    
    def _intelligent_smoothing(self, img_bgr):
        """智能平滑 - 保持重要结构的同时移除纹理"""
        # 使用边缘保留滤波，保持重要边界
        smooth = cv2.edgePreservingFilter(img_bgr, flags=2, sigma_s=60, sigma_r=0.3)
        
        # 轻度双边滤波，进一步平滑但保持边缘
        smooth = cv2.bilateralFilter(smooth, 7, 50, 50)
        
        return smooth
    
    def _moderate_color_simplification(self, img_bgr):
        """适度的颜色简化 - 保持足够的颜色层次"""
        # 将图像重塑为像素列表
        data = img_bgr.reshape((-1, 3))
        data = np.float32(data)
        
        # K-means聚类，保留更多颜色以维持识别性
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        K = 20  # 增加颜色数量，保持更多细节
        
        _, labels, centers = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        quantized_data = centers[labels.flatten()]
        quantized = quantized_data.reshape(img_bgr.shape)
        
        # 轻微平滑色块边界，但不过度
        quantized = cv2.medianBlur(quantized, 3)
        
        # 与原图混合，保持一些原始细节
        result = cv2.addWeighted(quantized, 0.8, img_bgr, 0.2, 0)
        
        return result
    
    def _selective_edge_enhancement(self, img_bgr, original_bgr):
        """选择性边缘增强 - 只保留重要的轮廓线"""
        # 转换为灰度
        gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
        
        # 使用更保守的边缘检测参数
        # 高斯模糊，减少噪声
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny边缘检测 - 使用更高的阈值，只保留主要边缘
        edges = cv2.Canny(blurred, 80, 160)  # 提高阈值，减少细节线条
        
        # 形态学操作，去除小的噪声边缘
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)  # 开运算去噪
        
        # 只保留较长的边缘线条
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)  # 闭运算连接
        
        # 进一步过滤，只保留重要边缘
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_edges = np.zeros_like(edges)
        
        # 只保留长度足够的轮廓
        for contour in contours:
            if cv2.arcLength(contour, False) > 50:  # 只保留长轮廓
                cv2.drawContours(filtered_edges, [contour], -1, 255, 1)
        
        # 轻微模糊边缘，让线条更柔和
        filtered_edges = cv2.GaussianBlur(filtered_edges, (3, 3), 0)
        
        # 将边缘叠加到图像上，使用更轻的权重
        result = img_bgr.copy()
        mask = filtered_edges > 100  # 只在强边缘处绘制线条
        
        # 绘制深灰色线条而不是纯黑色，更自然
        result[mask] = [40, 40, 40]  # 深灰色轮廓线
        
        return result
    
    def _apply_ghibli_colors(self, img_bgr):
        """应用宫崎骏色彩风格 - 温暖明亮但不过度"""
        # 转换到HSV空间
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # 适度增强饱和度
        s = np.clip(s * 1.2, 0, 255).astype(np.uint8)
        
        # 轻微增强亮度
        v = np.clip(v * 1.05, 0, 255).astype(np.uint8)
        
        # 色调偏向温暖，但不过度
        h = np.where((h >= 10) & (h <= 40), np.clip(h + 3, 0, 179), h)
        
        # 合并通道
        hsv_enhanced = cv2.merge([h, s, v])
        enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
        
        # LAB空间微调
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 轻微增强色彩鲜艳度
        a = np.clip(a + 5, 0, 255).astype(np.uint8)
        b = np.clip(b + 8, 0, 255).astype(np.uint8)
        
        # 增强对比度，但保持自然
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        lab_enhanced = cv2.merge([l, a, b])
        result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        # 轻微锐化，增强清晰度
        kernel = np.array([[-0.5,-0.5,-0.5], [-0.5,5,-0.5], [-0.5,-0.5,-0.5]])
        sharpened = cv2.filter2D(result, -1, kernel)
        result = cv2.addWeighted(result, 0.7, sharpened, 0.3, 0)
        
        return result
    
    def _update_progress(self, progress):
        """更新进度"""
        if self.progress_callback and self.task_id:
            self.progress_callback(self.task_id, progress, progress//10, 10, 0)

# 全局实例
optimized_ghibli_processor = OptimizedGhibliStyleTransfer()