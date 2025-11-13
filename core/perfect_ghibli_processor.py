#!/usr/bin/env python3
"""
完美宫崎骏风格处理器 - 基于真实宫崎骏作品特征
实现：真实照片 → 动漫化 + 宫崎骏色彩风格
"""

import cv2
import numpy as np
from PIL import Image
import torch
from .image_processor_interface import ImageProcessorInterface, ProcessingResult, ProcessingStyle

class PerfectGhibliProcessor(ImageProcessorInterface):
    """完美宫崎骏风格处理器"""
    
    def __init__(self):
        super().__init__(ProcessingStyle.GHIBLI_ENHANCED)
        self.progress_callback = None
        self.task_id = None
    
    def process(self, image: Image.Image, **kwargs) -> ProcessingResult:
        """处理图像为完美宫崎骏风格"""
        try:
            result_image = self._apply_perfect_ghibli_style(image)
            return ProcessingResult(success=True, image=result_image)
        except Exception as e:
            return ProcessingResult(success=False, error_message=str(e))
    
    def _apply_perfect_ghibli_style(self, image: Image.Image) -> Image.Image:
        """应用完美宫崎骏风格转换"""
        img_np = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # 阶段1: 保持结构的动漫化处理
        img_bgr = self._structure_preserving_anime(img_bgr)
        
        # 阶段2: 宫崎骏色彩风格
        img_bgr = self._ghibli_color_transformation(img_bgr)
        
        # 阶段3: 宫崎骏光影效果
        img_bgr = self._ghibli_lighting_effect(img_bgr)
        
        # 阶段4: 细节优化
        img_bgr = self._final_ghibli_polish(img_bgr)
        
        result_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result_rgb)
    
    def _structure_preserving_anime(self, img_bgr):
        """保持结构的动漫化处理"""
        # 1. 适度的边缘保留平滑 - 去除纹理但保持结构
        smooth = cv2.bilateralFilter(img_bgr, 9, 75, 75)
        smooth = cv2.bilateralFilter(smooth, 9, 75, 75)
        
        # 2. 智能颜色量化 - 基于宫崎骏作品的颜色数量
        quantized = self._intelligent_color_quantization(smooth)
        
        # 3. 保持边缘清晰度
        edges = self._extract_important_edges(img_bgr)
        
        # 4. 融合量化结果和边缘
        result = self._blend_quantized_with_edges(quantized, edges)
        
        return result
    
    def _intelligent_color_quantization(self, img_bgr):
        """智能颜色量化 - 基于宫崎骏作品特征"""
        # 转换为LAB色彩空间进行更精确的量化
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        data = lab.reshape((-1, 3)).astype(np.float32)
        
        # 使用适中的颜色数量 - 既有动漫感又保持细节
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        K = 16  # 16种颜色 - 平衡动漫化和细节保持
        
        _, labels, centers = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        
        quantized_lab = centers[labels.flatten()].reshape(lab.shape)
        quantized_bgr = cv2.cvtColor(quantized_lab, cv2.COLOR_LAB2BGR)
        
        # 使用超像素进一步平滑，但保持适度
        try:
            from skimage.segmentation import slic
            from skimage.color import label2rgb
            
            img_rgb = cv2.cvtColor(quantized_bgr, cv2.COLOR_BGR2RGB)
            segments = slic(img_rgb, n_segments=300, compactness=20, sigma=1)
            smooth_rgb = (label2rgb(segments, img_rgb, kind='avg') * 255).astype(np.uint8)
            result = cv2.cvtColor(smooth_rgb, cv2.COLOR_RGB2BGR)
        except ImportError:
            result = cv2.pyrMeanShiftFiltering(quantized_bgr, 15, 30)
        
        return result
    
    def _extract_important_edges(self, img_bgr):
        """提取重要边缘 - 保持结构清晰"""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # 使用Canny边缘检测
        edges = cv2.Canny(gray, 50, 150)
        
        # 轻微膨胀使边缘更连续
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # 高斯模糊使边缘更柔和
        edges = cv2.GaussianBlur(edges, (3, 3), 0)
        
        return edges
    
    def _blend_quantized_with_edges(self, quantized, edges):
        """融合量化结果和边缘"""
        # 将边缘转换为3通道
        edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # 轻微叠加边缘，保持动漫感但不过度
        result = cv2.addWeighted(quantized, 0.9, edges_3ch, 0.1, 0)
        
        return result
    
    def _ghibli_color_transformation(self, img_bgr):
        """宫崎骏色彩变换 - 基于真实作品分析"""
        # 转换到HSV空间进行精确调整
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # 宫崎骏色彩特征1: 温暖色调偏移
        h = self._apply_warm_tone_shift(h)
        
        # 宫崎骏色彩特征2: 适度饱和度提升
        s = cv2.add(s, 30)
        s = np.clip(s, 0, 220)  # 避免过饱和
        
        # 宫崎骏色彩特征3: 明亮但柔和的亮度
        v = cv2.add(v, 15)
        v = np.clip(v, 0, 240)
        
        hsv_enhanced = cv2.merge([h, s, v])
        bgr_enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
        
        # LAB空间进一步调整
        lab = cv2.cvtColor(bgr_enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 宫崎骏色彩特征4: 特定的色彩偏向
        a = cv2.add(a, 10)  # 轻微偏红/绿
        b = cv2.add(b, 15)  # 轻微偏黄
        a = np.clip(a, 0, 255)
        b = np.clip(b, 0, 255)
        
        lab_final = cv2.merge([l, a, b])
        result = cv2.cvtColor(lab_final, cv2.COLOR_LAB2BGR)
        
        return result
    
    def _apply_warm_tone_shift(self, h):
        """应用温暖色调偏移"""
        # 红色区域 (0-30) 更暖
        h = np.where((h >= 0) & (h <= 30), np.clip(h + 8, 0, 179), h)
        
        # 橙黄色区域 (30-60) 微调
        h = np.where((h > 30) & (h <= 60), np.clip(h + 5, 0, 179), h)
        
        # 蓝绿色区域 (90-150) 稍微偏暖
        h = np.where((h > 90) & (h <= 150), np.clip(h - 3, 0, 179), h)
        
        return h
    
    def _ghibli_lighting_effect(self, img_bgr):
        """宫崎骏光影效果 - 柔和梦幻"""
        h, w = img_bgr.shape[:2]
        
        # 创建柔和的径向光照
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h * 0.4, w * 0.5  # 光源稍微偏上
        
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        
        # 非常柔和的光照效果 - 宫崎骏特色
        light_mask = 1.0 - (distance / max_distance) * 0.12
        light_mask = np.clip(light_mask, 0.88, 1.0)
        
        # 应用光照
        result = img_bgr.astype(np.float32) * light_mask[:,:,np.newaxis]
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    def _final_ghibli_polish(self, img_bgr):
        """最终宫崎骏风格润色"""
        # 1. 轻微锐化增强细节
        kernel = np.array([[-0.2,-0.2,-0.2], [-0.2,2.6,-0.2], [-0.2,-0.2,-0.2]])
        sharpened = cv2.filter2D(img_bgr, -1, kernel)
        result = cv2.addWeighted(img_bgr, 0.8, sharpened, 0.2, 0)
        
        # 2. 最终色彩平衡
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 轻微对比度增强
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        lab_final = cv2.merge([l, a, b])
        result = cv2.cvtColor(lab_final, cv2.COLOR_LAB2BGR)
        
        # 3. 最终柔化处理
        soft = cv2.GaussianBlur(result, (3, 3), 0.5)
        result = cv2.addWeighted(result, 0.9, soft, 0.1, 0)
        
        return result
    
    def set_progress_callback(self, callback, task_id):
        """设置进度回调"""
        self.progress_callback = callback
        self.task_id = task_id
    
    def get_processing_info(self) -> dict:
        """获取处理器信息"""
        return {
            "processor_type": "PerfectGhibliProcessor",
            "style_type": self.style_type.value,
            "description": "完美宫崎骏风格处理器 - 动漫化+宫崎骏色彩"
        }