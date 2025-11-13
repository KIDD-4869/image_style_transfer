#!/usr/bin/env python3
"""
真正的动漫风格处理器 - 实现二次元动漫化效果
"""

import cv2
import numpy as np
from PIL import Image
import torch
from .image_processor_interface import ImageProcessorInterface, ProcessingResult, ProcessingStyle

class AnimeStyleProcessor(ImageProcessorInterface):
    """真正的动漫风格处理器 - 实现二次元效果"""
    
    def __init__(self):
        super().__init__(ProcessingStyle.GHIBLI_ENHANCED)
        self.progress_callback = None
        self.task_id = None
    
    def process(self, image: Image.Image, **kwargs) -> ProcessingResult:
        """处理图像为动漫风格"""
        try:
            result_image = self._apply_anime_style(image)
            return ProcessingResult(success=True, image=result_image)
        except Exception as e:
            return ProcessingResult(success=False, error_message=str(e))
    
    def _apply_anime_style(self, image: Image.Image) -> Image.Image:
        """应用真正的动漫风格转换"""
        img_np = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # 1. 强力边缘保留平滑 - 去除真实世界纹理
        img_bgr = self._strong_bilateral_filter(img_bgr)
        
        # 2. 动漫风格颜色量化 - 创造扁平色块
        img_bgr = self._anime_color_quantization(img_bgr)
        
        # 3. 生成动漫线稿
        img_bgr = self._generate_anime_lines(img_bgr)
        
        # 4. 宫崎骏色彩风格
        img_bgr = self._ghibli_color_style(img_bgr)
        
        # 5. 动漫光影效果
        img_bgr = self._anime_lighting(img_bgr)
        
        result_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result_rgb)
    
    def _strong_bilateral_filter(self, img_bgr):
        """强力双边滤波 - 去除真实世界细节"""
        # 多次双边滤波，彻底去除纹理
        filtered = img_bgr.copy()
        
        # 第一次：大范围平滑
        filtered = cv2.bilateralFilter(filtered, 15, 100, 100)
        
        # 第二次：进一步平滑
        filtered = cv2.bilateralFilter(filtered, 15, 100, 100)
        
        # 第三次：最终平滑
        filtered = cv2.bilateralFilter(filtered, 9, 80, 80)
        
        return filtered
    
    def _anime_color_quantization(self, img_bgr):
        """动漫风格颜色量化 - 创造扁平色块效果"""
        # 转换为浮点数
        data = img_bgr.reshape((-1, 3)).astype(np.float32)
        
        # K-means聚类，大幅减少颜色数量
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
        K = 8  # 只保留8种主要颜色，创造动漫效果
        
        _, labels, centers = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        
        # 重建图像
        quantized = centers[labels.flatten()]
        quantized = quantized.reshape(img_bgr.shape)
        
        # 使用SLIC超像素进一步平滑色块
        try:
            from skimage.segmentation import slic
            from skimage.color import label2rgb
            
            img_rgb = cv2.cvtColor(quantized, cv2.COLOR_BGR2RGB)
            segments = slic(img_rgb, n_segments=150, compactness=20, sigma=1)
            flat_rgb = (label2rgb(segments, img_rgb, kind='avg') * 255).astype(np.uint8)
            quantized = cv2.cvtColor(flat_rgb, cv2.COLOR_RGB2BGR)
        except ImportError:
            # 如果没有scikit-image，使用均值漂移
            quantized = cv2.pyrMeanShiftFiltering(quantized, 20, 40)
        
        return quantized
    
    def _generate_anime_lines(self, img_bgr):
        """生成动漫风格线稿"""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # 使用自适应阈值生成清晰线条
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                     cv2.THRESH_BINARY, 9, 10)
        
        # 形态学操作优化线条
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # 轻微模糊线条使其更自然
        edges = cv2.GaussianBlur(edges, (3, 3), 0)
        
        # 将线条叠加到图像上
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # 强化线条效果
        result = cv2.addWeighted(img_bgr, 0.8, edges_colored, 0.2, 0)
        
        return result
    
    def _ghibli_color_style(self, img_bgr):
        """宫崎骏色彩风格调整"""
        # 转换到HSV空间
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # 宫崎骏风格：温暖、高饱和度
        # 增强饱和度
        s = cv2.add(s, 50)
        s = np.clip(s, 0, 255)
        
        # 调整色调偏向温暖
        h = np.where(h < 30, h + 10, h)  # 红色更暖
        h = np.where((h >= 30) & (h < 60), h + 5, h)  # 黄色微调
        h = np.clip(h, 0, 179)
        
        # 提升亮度
        v = cv2.add(v, 20)
        v = np.clip(v, 0, 255)
        
        hsv_enhanced = cv2.merge([h, s, v])
        enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
        
        # LAB空间进一步调整
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 增强色彩鲜艳度
        a = cv2.add(a, 20)
        b = cv2.add(b, 15)
        a = np.clip(a, 0, 255)
        b = np.clip(b, 0, 255)
        
        lab_enhanced = cv2.merge([l, a, b])
        result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        return result
    
    def _anime_lighting(self, img_bgr):
        """动漫风格光影效果"""
        h, w = img_bgr.shape[:2]
        
        # 创建径向光照效果
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h * 0.4, w * 0.5
        
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        
        # 动漫风格的光照 - 更明显的效果
        light_mask = 1.0 - (distance / max_distance) * 0.15
        light_mask = np.clip(light_mask, 0.85, 1.0)
        
        # 应用光照
        result = img_bgr.astype(np.float32) * light_mask[:,:,np.newaxis]
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        # 最终锐化增强动漫感
        kernel = np.array([[-0.5,-0.5,-0.5], [-0.5,5,-0.5], [-0.5,-0.5,-0.5]])
        sharpened = cv2.filter2D(result, -1, kernel)
        result = cv2.addWeighted(result, 0.7, sharpened, 0.3, 0)
        
        return result
    
    def set_progress_callback(self, callback, task_id):
        """设置进度回调"""
        self.progress_callback = callback
        self.task_id = task_id
    
    def get_processing_info(self) -> dict:
        """获取处理器信息"""
        return {
            "processor_type": "AnimeStyleProcessor",
            "style_type": self.style_type.value,
            "description": "真正的动漫风格处理器 - 实现二次元效果"
        }