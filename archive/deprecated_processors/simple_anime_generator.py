#!/usr/bin/env python3
"""
简化的动漫图像生成器 - 基于原图重新生成动漫图片
"""

import cv2
import numpy as np
from PIL import Image
from .image_processor_interface import ImageProcessorInterface, ProcessingResult, ProcessingStyle

class SimpleAnimeGenerator(ImageProcessorInterface):
    """简化的动漫图像生成器"""
    
    def __init__(self):
        super().__init__(ProcessingStyle.GHIBLI_ENHANCED)
    
    def process(self, image: Image.Image, **kwargs) -> ProcessingResult:
        try:
            img_np = np.array(image)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            # 重新生成动漫图像
            anime_img = self._generate_anime_image(img_bgr)
            
            result_rgb = cv2.cvtColor(anime_img, cv2.COLOR_BGR2RGB)
            return ProcessingResult(success=True, image=Image.fromarray(result_rgb))
        except Exception as e:
            return ProcessingResult(success=False, error_message=str(e))
    
    def _generate_anime_image(self, img):
        """基于原图生成全新的动漫图像"""
        h, w = img.shape[:2]
        
        # 1. 提取主要颜色区域
        segments = self._create_color_segments(img)
        
        # 2. 创建动漫画布
        canvas = np.zeros_like(img)
        
        # 3. 填充动漫色块
        canvas = self._fill_anime_colors(canvas, segments)
        
        # 4. 添加动漫线条
        canvas = self._add_anime_outlines(canvas, img)
        
        # 5. 宫崎骏风格调色
        canvas = self._apply_ghibli_style(canvas)
        
        return canvas
    
    def _create_color_segments(self, img):
        """创建颜色分段"""
        # 简单的颜色量化
        data = img.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, 8, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        segmented = centers[labels.flatten()].reshape(img.shape).astype(np.uint8)
        return segmented
    
    def _fill_anime_colors(self, canvas, segments):
        """填充动漫颜色"""
        # 直接使用分段结果，但转换为动漫风格颜色
        hsv_segments = cv2.cvtColor(segments, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # 动漫化调整
        hsv_segments[:,:,1] = np.clip(hsv_segments[:,:,1] * 1.8, 0, 255)  # 高饱和度
        hsv_segments[:,:,2] = np.clip(hsv_segments[:,:,2] * 1.4, 0, 255)  # 高亮度
        
        # 色调偏移到宫崎骏风格
        hsv_segments[:,:,0] = np.where(hsv_segments[:,:,0] < 60, 
                                      hsv_segments[:,:,0] + 15, 
                                      hsv_segments[:,:,0])
        
        canvas = cv2.cvtColor(hsv_segments.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return canvas
    
    def _add_anime_outlines(self, canvas, original):
        """添加动漫轮廓线"""
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        
        # 边缘检测
        edges = cv2.Canny(gray, 50, 150)
        
        # 膨胀边缘使其更明显
        kernel = np.ones((2,2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # 在画布上绘制黑色轮廓
        canvas[edges > 0] = [20, 20, 20]
        
        return canvas
    
    def _apply_ghibli_style(self, canvas):
        """应用宫崎骏风格"""
        # 柔和光影
        h, w = canvas.shape[:2]
        center_x, center_y = w // 2, h // 3
        
        y, x = np.ogrid[:h, :w]
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        
        light_factor = 1.0 - (distance / max_distance) * 0.1
        light_factor = np.clip(light_factor, 0.9, 1.0)
        
        canvas = canvas.astype(np.float32)
        canvas *= light_factor[:, :, np.newaxis]
        canvas = np.clip(canvas, 0, 255).astype(np.uint8)
        
        return canvas
    
    def get_processing_info(self) -> dict:
        return {
            "processor_type": "SimpleAnimeGenerator",
            "style_type": self.style_type.value,
            "description": "简化的动漫图像生成器"
        }