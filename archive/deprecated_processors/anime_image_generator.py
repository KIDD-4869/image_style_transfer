#!/usr/bin/env python3
"""
动漫图像生成器 - 基于原图重新生成动漫风格图片
"""

import cv2
import numpy as np
from PIL import Image
from .image_processor_interface import ImageProcessorInterface, ProcessingResult, ProcessingStyle

class AnimeImageGenerator(ImageProcessorInterface):
    """动漫图像生成器 - 重新生成而非滤镜处理"""
    
    def __init__(self):
        super().__init__(ProcessingStyle.GHIBLI_ENHANCED)
    
    def process(self, image: Image.Image, **kwargs) -> ProcessingResult:
        try:
            img_np = np.array(image)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            # 1. 提取图像结构信息
            structure = self._extract_structure(img_bgr)
            
            # 2. 基于结构重新生成动漫图像
            anime_img = self._generate_anime_image(img_bgr, structure)
            
            result_rgb = cv2.cvtColor(anime_img, cv2.COLOR_BGR2RGB)
            return ProcessingResult(success=True, image=Image.fromarray(result_rgb))
        except Exception as e:
            return ProcessingResult(success=False, error_message=str(e))
    
    def _extract_structure(self, img):
        """提取图像结构信息"""
        # 边缘检测
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # 轮廓检测
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 区域分割
        segments = self._segment_regions(img)
        
        return {
            'edges': edges,
            'contours': contours,
            'segments': segments
        }
    
    def _segment_regions(self, img):
        """区域分割"""
        # 使用K-means进行区域分割
        data = img.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, 6, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        segmented = centers[labels.flatten()].reshape(img.shape).astype(np.uint8)
        return segmented
    
    def _generate_anime_image(self, original, structure):
        """基于结构重新生成动漫图像"""
        h, w = original.shape[:2]
        
        # 创建空白画布
        canvas = np.zeros_like(original)
        
        # 1. 生成动漫色块
        anime_colors = self._generate_anime_colors(structure['segments'])
        
        # 2. 绘制基础色块
        canvas = self._draw_color_blocks(canvas, anime_colors, structure['segments'])
        
        # 3. 添加动漫线条
        canvas = self._draw_anime_lines(canvas, structure['edges'], structure['contours'])
        
        # 4. 添加宫崎骏风格渲染
        canvas = self._apply_ghibli_rendering(canvas)
        
        return canvas
    
    def _generate_anime_colors(self, segments):
        """生成动漫风格色彩"""
        # 提取主要颜色
        unique_colors = np.unique(segments.reshape(-1, 3), axis=0)
        
        # 转换为动漫风格色彩
        anime_colors = []
        for color in unique_colors:
            # 转换到HSV
            hsv_color = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)[0][0]
            
            # 动漫化调整：高饱和度、明亮、纯净
            hsv_color[1] = min(255, int(hsv_color[1] * 1.8))  # 高饱和度
            hsv_color[2] = min(255, int(hsv_color[2] * 1.4))  # 高亮度
            
            # 色调调整为宫崎骏风格
            if hsv_color[0] < 30:  # 红色系
                hsv_color[0] = min(179, hsv_color[0] + 10)
            elif hsv_color[0] < 90:  # 黄绿色系
                hsv_color[0] = min(179, hsv_color[0] + 5)
            
            # 转换回BGR
            bgr_color = cv2.cvtColor(np.uint8([[hsv_color]]), cv2.COLOR_HSV2BGR)[0][0]
            anime_colors.append(bgr_color)
        
        return np.array(anime_colors)
    
    def _draw_color_blocks(self, canvas, anime_colors, segments):
        """绘制动漫色块"""
        # 为每个区域填充对应的动漫颜色
        unique_segments = np.unique(segments.reshape(-1, 3), axis=0)
        
        for i, segment_color in enumerate(unique_segments):
            if i < len(anime_colors):
                mask = np.all(segments == segment_color, axis=2)
                canvas[mask] = anime_colors[i]
        
        return canvas
    
    def _draw_anime_lines(self, canvas, edges, contours):
        """绘制动漫线条"""
        # 绘制边缘线条
        line_color = (20, 20, 20)  # 深色线条
        
        # 绘制轮廓
        cv2.drawContours(canvas, contours, -1, line_color, 2)
        
        # 添加细节线条
        edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        line_mask = edges_3ch > 128
        canvas[line_mask] = line_color
        
        return canvas
    
    def _apply_ghibli_rendering(self, canvas):
        """应用宫崎骏风格渲染"""
        # 柔和光影效果
        h, w = canvas.shape[:2]
        
        # 创建径向光照
        center_x, center_y = w // 2, h // 3
        y, x = np.ogrid[:h, :w]
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        
        # 柔和的光照效果
        light_factor = 1.0 - (distance / max_distance) * 0.15
        light_factor = np.clip(light_factor, 0.85, 1.0)
        
        # 应用光照
        canvas = canvas.astype(np.float32)
        canvas *= light_factor[:, :, np.newaxis]
        canvas = np.clip(canvas, 0, 255).astype(np.uint8)
        
        # 最终色彩增强
        hsv = cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.2, 0, 255)  # 饱和度
        hsv[:,:,2] = np.clip(hsv[:,:,2] * 1.1, 0, 255)  # 亮度
        
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def get_processing_info(self) -> dict:
        return {
            "processor_type": "AnimeImageGenerator",
            "style_type": self.style_type.value,
            "description": "动漫图像生成器 - 重新生成动漫风格图片"
        }