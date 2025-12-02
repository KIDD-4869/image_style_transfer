#!/usr/bin/env python3
"""
真正的宫崎骏动漫风格转换器
"""

import cv2
import numpy as np
from PIL import Image
from .image_processor_interface import ImageProcessorInterface, ProcessingResult, ProcessingStyle

class RealAnimeConverter(ImageProcessorInterface):
    """真正的动漫风格转换器"""
    
    def __init__(self):
        super().__init__(ProcessingStyle.GHIBLI_ENHANCED)
    
    def process(self, image: Image.Image, **kwargs) -> ProcessingResult:
        try:
            img_np = np.array(image)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            # 1. 强力动漫化处理
            anime_img = self._apply_anime_effect(img_bgr)
            
            # 2. 宫崎骏色彩风格
            ghibli_img = self._apply_ghibli_colors(anime_img)
            
            # 3. 动漫线稿
            final_img = self._add_anime_lines(ghibli_img)
            
            result_rgb = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
            return ProcessingResult(success=True, image=Image.fromarray(result_rgb))
        except Exception as e:
            return ProcessingResult(success=False, error_message=str(e))
    
    def _apply_anime_effect(self, img):
        """强力动漫化效果"""
        # 颜色量化 - 减少颜色数量
        data = img.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, 8, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        quantized = centers[labels.flatten()].reshape(img.shape)
        
        # 双边滤波平滑
        smooth = cv2.bilateralFilter(quantized, 15, 80, 80)
        smooth = cv2.bilateralFilter(smooth, 15, 80, 80)
        
        return smooth
    
    def _apply_ghibli_colors(self, img):
        """宫崎骏色彩风格"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # 宫崎骏特色：温暖、高饱和度、明亮
        hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.4, 0, 255)  # 提高饱和度
        hsv[:,:,2] = np.clip(hsv[:,:,2] * 1.2, 0, 255)  # 提高亮度
        
        # 色调偏移 - 偏向温暖色调
        hsv[:,:,0] = np.where(hsv[:,:,0] < 30, hsv[:,:,0] + 10, hsv[:,:,0])
        
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def _add_anime_lines(self, img):
        """添加动漫线稿"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 边缘检测
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 7)
        edges = cv2.medianBlur(edges, 5)
        
        # 转换为3通道
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # 与彩色图像混合
        result = cv2.bitwise_and(img, edges)
        
        return result
    
    def get_processing_info(self) -> dict:
        return {
            "processor_type": "RealAnimeConverter",
            "style_type": self.style_type.value,
            "description": "真正的宫崎骏动漫风格转换器"
        }