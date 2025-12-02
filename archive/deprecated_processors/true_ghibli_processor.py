#!/usr/bin/env python3
"""
真正的宫崎骏风格处理器 - 基于实际宫崎骏作品特征
"""

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from .image_processor_interface import ImageProcessorInterface, ProcessingResult, ProcessingStyle

class TrueGhibliProcessor(ImageProcessorInterface):
    """真正的宫崎骏风格处理器"""
    
    def __init__(self):
        super().__init__(ProcessingStyle.GHIBLI_ENHANCED)
        self.progress_callback = None
        self.task_id = None
    
    def process(self, image: Image.Image, **kwargs) -> ProcessingResult:
        """处理图像为宫崎骏风格"""
        try:
            result_image = self._apply_true_ghibli_style(image)
            return ProcessingResult(success=True, image=result_image)
        except Exception as e:
            return ProcessingResult(success=False, error_message=str(e))
    
    def _apply_true_ghibli_style(self, image: Image.Image) -> Image.Image:
        """应用真正的宫崎骏风格"""
        img_np = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # 1. 宫崎骏风格色彩调整
        img_bgr = self._ghibli_color_transform(img_bgr)
        
        # 2. 柔和处理（去除过度边缘化）
        img_bgr = self._soft_processing(img_bgr)
        
        # 3. 宫崎骏特色光影
        img_bgr = self._ghibli_lighting(img_bgr)
        
        # 4. 细节优化
        img_bgr = self._detail_enhancement(img_bgr)
        
        result_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result_rgb)
    
    def _ghibli_color_transform(self, img_bgr):
        """宫崎骏风格色彩变换"""
        # 转换到HSV空间进行精确调整
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # 宫崎骏风格特点：温暖、柔和、高饱和度
        # 1. 色调调整 - 偏向温暖色调
        h = np.where(h < 30, h + 5, h)  # 红色区域更暖
        h = np.where((h >= 30) & (h < 60), h + 3, h)  # 黄色区域微调
        h = np.where((h >= 90) & (h < 150), h - 5, h)  # 蓝绿色区域偏暖
        
        # 2. 饱和度提升但保持自然
        s = cv2.add(s, 25)
        s = np.clip(s, 0, 200)  # 避免过饱和
        
        # 3. 亮度优化 - 宫崎骏风格明亮但柔和
        v = cv2.add(v, 15)
        v = np.clip(v, 0, 245)
        
        hsv_enhanced = cv2.merge([h, s, v])
        return cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
    
    def _soft_processing(self, img_bgr):
        """柔和处理 - 去除过度边缘化"""
        # 使用双边滤波保持边缘但柔化细节
        soft = cv2.bilateralFilter(img_bgr, 15, 80, 80)
        
        # 轻微的高斯模糊增加柔和感
        blur = cv2.GaussianBlur(img_bgr, (3, 3), 0.5)
        
        # 混合原图、双边滤波和模糊
        result = cv2.addWeighted(soft, 0.7, blur, 0.3, 0)
        
        return result
    
    def _ghibli_lighting(self, img_bgr):
        """宫崎骏特色光影效果"""
        h, w = img_bgr.shape[:2]
        
        # 创建柔和的径向光照
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h * 0.4, w * 0.5  # 光源稍微偏上
        
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        
        # 非常柔和的光照效果
        light_mask = 1.0 - (distance / max_distance) * 0.08
        light_mask = np.clip(light_mask, 0.92, 1.0)
        
        # 应用光照
        result = img_bgr.astype(np.float32) * light_mask[:,:,np.newaxis]
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    def _detail_enhancement(self, img_bgr):
        """细节增强 - 宫崎骏风格的细腻处理"""
        # 轻微锐化
        kernel = np.array([[-0.1,-0.1,-0.1], [-0.1,1.8,-0.1], [-0.1,-0.1,-0.1]])
        sharpened = cv2.filter2D(img_bgr, -1, kernel)
        
        # 与原图混合
        result = cv2.addWeighted(img_bgr, 0.8, sharpened, 0.2, 0)
        
        # 最终色彩平衡
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 轻微增强对比度
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        lab_final = cv2.merge([l, a, b])
        result = cv2.cvtColor(lab_final, cv2.COLOR_LAB2BGR)
        
        return result
    
    def set_progress_callback(self, callback, task_id):
        """设置进度回调"""
        self.progress_callback = callback
        self.task_id = task_id
    
    def get_processing_info(self) -> dict:
        """获取处理器信息"""
        return {
            "processor_type": "TrueGhibliProcessor",
            "style_type": self.style_type.value,
            "description": "真正的宫崎骏风格处理器"
        }