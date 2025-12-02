#!/usr/bin/env python3
"""
宫崎骏风格处理器 - 统一实现
"""

import cv2
import numpy as np
from PIL import Image
import time
import logging
from typing import Dict, Any

from .base import BaseProcessor, ProcessingStrategy, ProcessingResult

logger = logging.getLogger(__name__)


class GhibliProcessor(BaseProcessor):
    """宫崎骏风格处理器"""
    
    def __init__(self):
        super().__init__(
            name="GhibliProcessor",
            description="宫崎骏动画风格转换器 - 支持多种处理策略"
        )
    
    def process(
        self, 
        image: Image.Image,
        strategy: ProcessingStrategy = ProcessingStrategy.BALANCED,
        **kwargs
    ) -> ProcessingResult:
        """
        处理图像为宫崎骏风格
        
        Args:
            image: 输入图像
            strategy: 处理策略
            **kwargs: 额外参数
            
        Returns:
            ProcessingResult: 处理结果
        """
        start_time = time.time()
        
        try:
            logger.info(f"开始处理图像，策略: {strategy.value}")
            
            # 根据策略选择处理方法
            if strategy == ProcessingStrategy.FAST:
                result_image = self._fast_process(image)
            elif strategy == ProcessingStrategy.QUALITY:
                result_image = self._quality_process(image)
            else:  # BALANCED
                result_image = self._balanced_process(image)
            
            processing_time = time.time() - start_time
            logger.info(f"处理完成，耗时: {processing_time:.2f}秒")
            
            return ProcessingResult(
                success=True,
                image=result_image,
                processing_time=processing_time,
                metadata={'strategy': strategy.value}
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"处理失败: {e}")
            return ProcessingResult(
                success=False,
                error_message=str(e),
                processing_time=processing_time
            )
    
    def _fast_process(self, image: Image.Image) -> Image.Image:
        """
        快速处理模式 - 基础色彩调整
        
        Args:
            image: 输入图像
            
        Returns:
            处理后的图像
        """
        self.update_progress(10, 1, 5, 0)
        
        # 转换为OpenCV格式
        img_np = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        self.update_progress(30, 2, 5, 0)
        
        # 快速色彩调整
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.3, 0, 255)  # 提高饱和度
        hsv[:,:,2] = np.clip(hsv[:,:,2] * 1.1, 0, 255)  # 提高亮度
        img_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        self.update_progress(60, 3, 5, 0)
        
        # 轻微平滑
        img_bgr = cv2.bilateralFilter(img_bgr, 5, 50, 50)
        
        self.update_progress(90, 4, 5, 0)
        
        # 转换回RGB
        result_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        self.update_progress(100, 5, 5, 0)
        
        return Image.fromarray(result_rgb)
    
    def _balanced_process(self, image: Image.Image) -> Image.Image:
        """
        平衡处理模式 - 改进的宫崎骏风格（保持细节）
        
        Args:
            image: 输入图像
            
        Returns:
            处理后的图像
        """
        self.update_progress(5, 1, 12, 0)
        
        # 转换为OpenCV格式
        img_np = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        self.update_progress(10, 2, 12, 0)
        
        # 1. 适度平滑（保留细节）
        img_bgr = cv2.bilateralFilter(img_bgr, 9, 75, 75)
        
        self.update_progress(20, 3, 12, 0)
        
        # 2. 适度颜色量化（保留更多颜色）
        img_bgr = self._color_quantization(img_bgr, num_colors=32)
        
        self.update_progress(35, 5, 12, 0)
        
        # 3. 超像素平滑（保持结构）
        img_bgr = self._apply_superpixel_smoothing(img_bgr, n_segments=300)
        
        self.update_progress(50, 7, 12, 0)
        
        # 4. 宫崎骏色彩风格（适度）
        img_bgr = self._apply_ghibli_colors(img_bgr, intensity=1.0)
        
        self.update_progress(65, 8, 12, 0)
        
        # 5. 添加精细边缘
        img_bgr = self._add_detailed_edges(img_bgr)
        
        self.update_progress(80, 10, 12, 0)
        
        # 6. 梦幻光影
        img_bgr = self._add_dreamy_lighting(img_bgr)
        
        self.update_progress(90, 11, 12, 0)
        
        # 7. 最终优化
        img_bgr = self._final_polish(img_bgr)
        
        self.update_progress(100, 12, 12, 0)
        
        # 转换回RGB
        result_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result_rgb)
    
    def _quality_process(self, image: Image.Image) -> Image.Image:
        """
        高质量处理模式 - 保守的宫崎骏风格（保持原图色彩和细节）
        
        策略：只做风格化，不改变颜色
        
        Args:
            image: 输入图像
            
        Returns:
            处理后的图像
        """
        self.update_progress(5, 1, 10, 0)
        
        # 转换为OpenCV格式
        img_np = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        original_bgr = img_bgr.copy()  # 保存原图
        
        self.update_progress(15, 2, 10, 0)
        
        # 1. 轻微平滑（保留大部分细节）
        img_bgr = cv2.bilateralFilter(img_bgr, 5, 50, 50)
        
        self.update_progress(30, 3, 10, 0)
        
        # 2. 轻微增强色彩饱和度（不改变色调）
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.15, 0, 255)  # 轻微提高饱和度
        hsv[:,:,2] = np.clip(hsv[:,:,2] * 1.05, 0, 255)  # 轻微提高亮度
        img_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        self.update_progress(50, 5, 10, 0)
        
        # 3. 添加柔和的卡通边缘（不破坏原图）
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.GaussianBlur(edges, (3, 3), 0)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        # 轻微混合边缘
        img_bgr = cv2.addWeighted(img_bgr, 0.95, edges_colored, 0.05, 0)
        
        self.update_progress(70, 7, 10, 0)
        
        # 4. 轻微锐化（增强细节）
        kernel = np.array([[-0.5,-0.5,-0.5], [-0.5,5,-0.5], [-0.5,-0.5,-0.5]])
        sharpened = cv2.filter2D(img_bgr, -1, kernel)
        img_bgr = cv2.addWeighted(img_bgr, 0.8, sharpened, 0.2, 0)
        
        self.update_progress(85, 8, 10, 0)
        
        # 5. 与原图混合（保持原图特征）
        img_bgr = cv2.addWeighted(original_bgr, 0.3, img_bgr, 0.7, 0)
        
        self.update_progress(100, 10, 10, 0)
        
        # 转换回RGB
        result_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result_rgb)
    
    def _color_quantization(self, img_bgr: np.ndarray, num_colors: int = 16) -> np.ndarray:
        """
        颜色量化 - 减少颜色数量，创建动漫色块效果
        
        Args:
            img_bgr: 输入图像
            num_colors: 目标颜色数量（越少越像动漫）
            
        Returns:
            量化后的图像
        """
        data = img_bgr.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        quantized = centers[labels.flatten()].reshape(img_bgr.shape)
        return quantized
    
    def _apply_superpixel_smoothing(self, img_bgr: np.ndarray, n_segments: int = 200) -> np.ndarray:
        """
        超像素平滑 - 创建动漫风格的色块效果
        
        Args:
            img_bgr: 输入图像
            n_segments: 超像素数量
            
        Returns:
            平滑后的图像
        """
        try:
            from skimage.segmentation import slic
            from skimage.color import label2rgb
            
            # 转换为RGB
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # SLIC超像素分割
            segments = slic(img_rgb, n_segments=n_segments, compactness=20, sigma=1, start_label=1)
            
            # 用平均颜色填充每个超像素
            smoothed = label2rgb(segments, img_rgb, kind='avg')
            smoothed = (smoothed * 255).astype(np.uint8)
            
            # 转换回BGR
            return cv2.cvtColor(smoothed, cv2.COLOR_RGB2BGR)
            
        except Exception as e:
            logger.warning(f"超像素平滑失败: {e}，使用备用方法")
            # 备用方法：使用均值滤波
            return cv2.blur(img_bgr, (5, 5))
    
    def _add_cartoon_edges(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        添加卡通边缘线条 - 明显的黑色轮廓线
        
        Args:
            img_bgr: 输入图像
            
        Returns:
            添加边缘后的图像
        """
        # 转换为灰度
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # 使用自适应阈值检测边缘
        edges = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            blockSize=9,
            C=2
        )
        
        # 反转边缘（黑色线条）
        edges = cv2.bitwise_not(edges)
        
        # 细化边缘
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.erode(edges, kernel, iterations=1)
        
        # 转换为BGR
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # 将边缘叠加到图像上（使用multiply混合）
        result = cv2.multiply(img_bgr.astype(np.float32), edges_bgr.astype(np.float32) / 255.0)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _enhance_anime_style(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        增强动漫风格 - 提高对比度和饱和度
        
        Args:
            img_bgr: 输入图像
            
        Returns:
            增强后的图像
        """
        # 转换到LAB色彩空间
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        # L通道：增强对比度
        l_channel = lab[:, :, 0]
        l_mean = np.mean(l_channel)
        lab[:, :, 0] = np.clip((l_channel - l_mean) * 1.3 + l_mean, 0, 255)
        
        # A和B通道：增强饱和度
        lab[:, :, 1] = np.clip(lab[:, :, 1] * 1.2, 0, 255)
        lab[:, :, 2] = np.clip(lab[:, :, 2] * 1.2, 0, 255)
        
        # 转换回BGR
        img_bgr = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        
        # HSV调整：进一步增强饱和度
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.4, 0, 255)  # 饱和度
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.15, 0, 255)  # 亮度
        
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def _apply_ghibli_colors(self, img_bgr: np.ndarray, intensity: float = 1.0) -> np.ndarray:
        """应用宫崎骏色彩风格"""
        # HSV调整
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,1] = np.clip(hsv[:,:,1] * (1.3 * intensity), 0, 255)  # 饱和度
        hsv[:,:,2] = np.clip(hsv[:,:,2] * (1.1 * intensity), 0, 255)  # 亮度
        
        # 色调偏移（温暖色调）
        h = hsv[:,:,0]
        h = np.where((h > 10) & (h < 40), np.clip(h + 8, 0, 179), h)
        hsv[:,:,0] = h
        
        img_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        # LAB调整
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[:,:,1] = np.clip(lab[:,:,1] + 10, 0, 255)  # a通道
        lab[:,:,2] = np.clip(lab[:,:,2] + 15, 0, 255)  # b通道
        
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    
    def _add_soft_edges(self, img_bgr: np.ndarray) -> np.ndarray:
        """添加柔和边缘"""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.GaussianBlur(edges, (3, 3), 0)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return cv2.addWeighted(img_bgr, 0.9, edges_colored, 0.1, 0)
    
    def _add_detailed_edges(self, img_bgr: np.ndarray) -> np.ndarray:
        """添加精细边缘"""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # 多种边缘检测
        edges_canny = cv2.Canny(gray, 50, 150)
        edges_adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 9, 3
        )
        
        # 合并边缘
        edges = cv2.bitwise_or(edges_canny, edges_adaptive)
        edges = cv2.GaussianBlur(edges, (3, 3), 0.5)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        return cv2.addWeighted(img_bgr, 0.85, edges_colored, 0.15, 0)
    
    def _add_dreamy_lighting(self, img_bgr: np.ndarray) -> np.ndarray:
        """添加梦幻光影"""
        h, w = img_bgr.shape[:2]
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        
        light_mask = 1.0 - (distance / max_distance) * 0.08
        light_mask = np.clip(light_mask, 0.92, 1.0)
        
        result = img_bgr.astype(np.float32) * light_mask[:,:,np.newaxis]
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _add_strong_cartoon_edges(self, img_bgr: np.ndarray) -> np.ndarray:
        """添加更粗的卡通边缘"""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # 使用更强的边缘检测
        edges = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,  # 更大的块
            C=3
        )
        
        # 反转并加粗边缘
        edges = cv2.bitwise_not(edges)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # 转换为BGR
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # 更强的混合
        result = cv2.multiply(img_bgr.astype(np.float32), edges_bgr.astype(np.float32) / 255.0)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _enhance_anime_style_extreme(self, img_bgr: np.ndarray) -> np.ndarray:
        """极致动漫风格增强"""
        # HSV超强调整
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # 极高饱和度
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 2.0, 0, 255)
        
        # 提高亮度
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.3, 0, 255)
        
        img_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        # LAB空间调整
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        # 增强对比度
        l_channel = lab[:, :, 0]
        l_mean = np.mean(l_channel)
        lab[:, :, 0] = np.clip((l_channel - l_mean) * 1.5 + l_mean, 0, 255)
        
        # 增强色彩
        lab[:, :, 1] = np.clip(lab[:, :, 1] * 1.5, 0, 255)
        lab[:, :, 2] = np.clip(lab[:, :, 2] * 1.5, 0, 255)
        
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    
    def _add_anime_highlights(self, img_bgr: np.ndarray) -> np.ndarray:
        """添加动漫高光效果"""
        # 找到亮区域
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # 扩展高光区域
        kernel = np.ones((5, 5), np.uint8)
        bright_mask = cv2.dilate(bright_mask, kernel, iterations=1)
        bright_mask = cv2.GaussianBlur(bright_mask, (15, 15), 0)
        
        # 创建高光层
        highlight = np.ones_like(img_bgr) * 255
        
        # 混合高光
        bright_mask_3ch = cv2.cvtColor(bright_mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
        result = img_bgr.astype(np.float32) * (1 - bright_mask_3ch * 0.3) + highlight * bright_mask_3ch * 0.3
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _final_polish(self, img_bgr: np.ndarray) -> np.ndarray:
        """最终润色"""
        # 强锐化
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(img_bgr, -1, kernel)
        result = cv2.addWeighted(img_bgr, 0.6, sharpened, 0.4, 0)
        
        # 轻微柔化边缘
        soft = cv2.GaussianBlur(result, (3, 3), 0.5)
        result = cv2.addWeighted(result, 0.85, soft, 0.15, 0)
        
        return result
