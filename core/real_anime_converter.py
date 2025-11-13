#!/usr/bin/env python3
"""
真正的动漫化转换器 - 基于你提供的参考图片分析
实现扁平化色块、清晰轮廓线、简化细节的真正动漫效果
"""

import cv2
import numpy as np
from PIL import Image
import os
from typing import Optional

class RealAnimeConverter:
    """真正的动漫化转换器"""
    
    def __init__(self):
        self.progress_callback = None
        self.task_id = None
        
    def set_progress_callback(self, callback, task_id):
        """设置进度回调"""
        self.progress_callback = callback
        self.task_id = task_id
        
    def convert_to_anime(self, image: Image.Image) -> Image.Image:
        """转换为真正的动漫风格"""
        print("🎨 开始真正的动漫化转换...")
        
        # 转换为OpenCV格式
        img_np = np.array(image)
        if img_np.ndim == 3 and img_np.shape[2] == 3:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
            
        self._update_progress(10)
        
        # 1. 强力平滑 - 移除所有纹理细节
        smoothed = self._aggressive_smoothing(img_bgr)
        self._update_progress(25)
        
        # 2. 颜色量化 - 创建扁平色块
        quantized = self._color_quantization(smoothed)
        self._update_progress(50)
        
        # 3. 生成清晰轮廓线
        edges = self._generate_clean_edges(img_bgr)
        self._update_progress(75)
        
        # 4. 合成最终动漫效果
        final = self._compose_anime_style(quantized, edges)
        self._update_progress(100)
        
        # 转换回RGB
        result_rgb = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result_rgb)
    
    def _aggressive_smoothing(self, img_bgr):
        """强力平滑处理 - 移除所有纹理"""
        # 多次双边滤波，彻底移除纹理
        smooth = img_bgr.copy()
        
        # 第一次：大范围平滑
        smooth = cv2.bilateralFilter(smooth, 15, 100, 100)
        
        # 第二次：中等范围平滑
        smooth = cv2.bilateralFilter(smooth, 9, 80, 80)
        
        # 第三次：细节平滑
        smooth = cv2.bilateralFilter(smooth, 7, 60, 60)
        
        return smooth
    
    def _color_quantization(self, img_bgr):
        """颜色量化 - 创建扁平色块效果"""
        # 将图像重塑为像素列表
        data = img_bgr.reshape((-1, 3))
        data = np.float32(data)
        
        # K-means聚类，大幅减少颜色数量
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
        K = 8  # 只保留8种主要颜色，创造真正的扁平效果
        
        _, labels, centers = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # 将聚类中心转换为整数
        centers = np.uint8(centers)
        
        # 重建图像
        quantized_data = centers[labels.flatten()]
        quantized = quantized_data.reshape(img_bgr.shape)
        
        # 进一步平滑色块边界
        quantized = cv2.medianBlur(quantized, 5)
        
        return quantized
    
    def _generate_clean_edges(self, img_bgr):
        """生成清晰的轮廓线"""
        # 转换为灰度
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # 高斯模糊，为边缘检测做准备
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Canny边缘检测 - 调整参数获得清晰边缘
        edges = cv2.Canny(blurred, 30, 80)
        
        # 形态学操作，连接断裂的边缘
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # 轻微膨胀，让线条更明显
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        return edges
    
    def _compose_anime_style(self, quantized, edges):
        """合成最终的动漫风格"""
        # 将边缘转换为3通道
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # 创建黑色轮廓线
        edges_colored = 255 - edges_colored  # 反转，让边缘变成黑色
        
        # 将轮廓线叠加到量化图像上
        # 使用加权混合，让轮廓线更突出
        result = cv2.addWeighted(quantized, 0.8, edges_colored, 0.2, 0)
        
        # 在边缘位置直接绘制黑线
        mask = edges > 0
        result[mask] = [0, 0, 0]  # 黑色轮廓线
        
        # 最后的色彩增强 - 让颜色更鲜艳
        result = self._enhance_anime_colors(result)
        
        return result
    
    def _enhance_anime_colors(self, img_bgr):
        """增强动漫色彩"""
        # 转换到HSV空间
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # 大幅增强饱和度
        s = np.clip(s * 1.4, 0, 255).astype(np.uint8)
        
        # 适度增强亮度
        v = np.clip(v * 1.1, 0, 255).astype(np.uint8)
        
        # 合并通道
        hsv_enhanced = cv2.merge([h, s, v])
        result = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
        
        return result
    
    def _update_progress(self, progress):
        """更新进度"""
        if self.progress_callback and self.task_id:
            self.progress_callback(self.task_id, progress, progress//10, 10, 0)

# 全局实例
real_anime_converter = RealAnimeConverter()