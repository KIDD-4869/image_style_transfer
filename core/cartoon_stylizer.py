#!/usr/bin/env python3
"""
卡通风格化器 - 让真实事物变成"画出来的"动漫效果
"""

import cv2
import numpy as np
from PIL import Image

class CartoonStylizer:
    def __init__(self):
        self.progress_callback = None
        self.task_id = None
        
    def set_progress_callback(self, callback, task_id):
        self.progress_callback = callback
        self.task_id = task_id
        
    def apply_ghibli_style(self, image: Image.Image) -> Image.Image:
        img_np = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) if img_np.ndim == 3 else cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        
        self._update_progress(10)
        
        # 1. 极度平滑 - 完全移除真实纹理
        smooth = self._extreme_smoothing(img_bgr)
        self._update_progress(30)
        
        # 2. 激进颜色量化 - 创造扁平色块
        quantized = self._aggressive_quantization(smooth)
        self._update_progress(60)
        
        # 3. 卡通轮廓线
        cartoon = self._add_cartoon_edges(quantized, img_bgr)
        self._update_progress(80)
        
        # 4. 动漫色彩
        final = self._cartoon_colors(cartoon)
        self._update_progress(100)
        
        return Image.fromarray(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
    
    def _extreme_smoothing(self, img):
        # 圆滑精细的平滑
        smooth = cv2.bilateralFilter(img, 9, 75, 75)
        return cv2.edgePreservingFilter(smooth, flags=2, sigma_s=50, sigma_r=0.4)
    
    def _aggressive_quantization(self, img):
        # 保持更多颜色层次
        data = img.reshape((-1, 3)).astype(np.float32)
        _, labels, centers = cv2.kmeans(data, 16, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0), 10, cv2.KMEANS_RANDOM_CENTERS)
        quantized = centers[labels.flatten()].reshape(img.shape).astype(np.uint8)
        # 与原图混合保持色彩
        return cv2.addWeighted(quantized, 0.7, img, 0.3, 0)
    
    def _add_cartoon_edges(self, img, original):
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        # 精细的边缘检测
        edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 120)
        # 细化线条
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        # 柔和叠加
        result = img.copy()
        mask = edges > 0
        result[mask] = [30, 30, 30]  # 深灰色线条
        return result
    
    def _cartoon_colors(self, img):
        # 宫崎骏色彩风格
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        # 适度增强
        s = np.clip(s * 1.2, 0, 255).astype(np.uint8)
        v = np.clip(v * 1.1, 0, 255).astype(np.uint8)
        # 温暖色调
        h = np.where((h >= 10) & (h <= 40), np.clip(h + 5, 0, 179), h)
        return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
    
    def _update_progress(self, progress):
        if self.progress_callback and self.task_id:
            self.progress_callback(self.task_id, progress, progress//10, 10, 0)

cartoon_stylizer = CartoonStylizer()