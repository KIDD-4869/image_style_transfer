#!/usr/bin/env python3
"""
高级动漫风格处理模块
实现SLIC超像素分割、XDoG线条提取、多尺度融合等关键技术
为端到端GAN架构奠定基础
"""

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.segmentation import slic
from skimage.color import label2rgb
from skimage.util import img_as_float
import warnings
warnings.filterwarnings('ignore')

class AdvancedAnimeProcessor:
    """高级动漫风格处理器 - 长期目标的核心模块"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化参数 - 优化动漫色块效果
        self.slic_segments = 300  # 超像素数量，减少以获得更大的色块
        self.slic_compactness = 15  # 紧凑度参数，降低以获得更自然的边界
        self.slic_sigma = 0.8  # 高斯核标准差，降低以保持更多细节
        
        # XDoG参数 - 优化线条清晰度
        self.xdog_k = 1.6  # 高斯模糊比例，降低以增强线条细节
        self.xdog_sigma = 0.8  # 基础高斯模糊标准差，降低以保持细节
        self.xdog_epsilon = -0.15  # 阈值偏移，调整以获得更好的线条对比度
        self.xdog_phi = 15  # 对比度增强参数，增强以获得更清晰的线条
        
        # 多尺度参数
        self.pyramid_levels = 4  # 金字塔层数
        self.scale_factor = 0.8  # 缩放因子
        
        print("🎨 高级动漫风格处理器初始化完成")
    
    def slic_superpixel_segmentation(self, img_bgr):
        """
        SLIC超像素分割 - 创建自然的动漫色块边界
        
        Args:
            img_bgr: BGR格式图像
            
        Returns:
            segmented_img: 分割后的图像
            segments: 分割标签
        """
        try:
            # 转换为RGB格式（skimage使用RGB）
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_float = img_as_float(img_rgb)
            
            # SLIC超像素分割
            print(f"🔧 执行SLIC超像素分割，目标分割数: {self.slic_segments}")
            segments = slic(
                img_float, 
                n_segments=self.slic_segments,
                compactness=self.slic_compactness,
                sigma=self.slic_sigma,
                start_label=1
            )
            
            # 生成平均颜色的分割图像
            segmented_rgb = label2rgb(segments, img_rgb, kind='avg')
            segmented_bgr = cv2.cvtColor((segmented_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            
            print(f"✅ SLIC分割完成，实际分割数: {len(np.unique(segments))}")
            return segmented_bgr, segments
            
        except Exception as e:
            print(f"❌ SLIC分割失败: {e}")
            # 回退到均值漂移滤波
            try:
                fallback = cv2.pyrMeanShiftFiltering(img_bgr, 15, 30)
                print("⚠️ 使用均值漂移滤波作为回退方案")
                return fallback, None
            except Exception as e2:
                print(f"❌ 回退方案也失败: {e2}")
                return img_bgr, None
    
    def xdog_line_extraction(self, gray_img):
        """
        XDoG线条提取 - 生成手绘感的动漫线条，优化清晰度
        
        Args:
            gray_img: 灰度图像
            
        Returns:
            xdog_edges: XDoG边缘图像
        """
        try:
            print(f"🔧 执行XDoG线条提取")
            
            # 预处理：轻微锐化增强边缘
            kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(gray_img.astype(np.uint8), -1, kernel_sharpen)
            gray_enhanced = cv2.addWeighted(gray_img, 0.7, sharpened, 0.3, 0)
            
            # 第一个高斯模糊
            g1 = cv2.GaussianBlur(gray_enhanced.astype(np.float32), (0, 0), self.xdog_sigma)
            
            # 第二个高斯模糊（更大的sigma）
            g2 = cv2.GaussianBlur(gray_enhanced.astype(np.float32), (0, 0), self.xdog_sigma * self.xdog_k)
            
            # 计算差分
            difference = g1 - g2
            
            # 归一化
            if np.max(np.abs(difference)) > 0:
                difference = difference / np.max(np.abs(difference))
            
            # XDoG处理 - 优化线条对比度
            edges = np.ones_like(difference)
            
            # 应用阈值和增强，使用更精确的阈值
            mask = difference < self.xdog_epsilon
            edges[mask] = 1 + np.tanh(self.xdog_phi * (difference[mask] - self.xdog_epsilon))
            
            # 转换为0-255范围
            edges = (edges * 255).astype(np.uint8)
            
            # 反转边缘（线条为黑色，背景为白色）
            xdog_edges = 255 - edges
            
            # 后处理：增强线条清晰度
            # 轻微膨胀让线条更连贯
            kernel = np.ones((2,2), np.uint8)
            xdog_edges = cv2.dilate(xdog_edges, kernel, iterations=1)
            
            # 轻微腐蚀保持线条精细度
            xdog_edges = cv2.erode(xdog_edges, kernel, iterations=1)
            
            # 应用自适应阈值增强线条对比度
            xdog_edges = cv2.adaptiveThreshold(xdog_edges, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                              cv2.THRESH_BINARY, 3, 2)
            
            print("✅ XDoG线条提取完成")
            return xdog_edges
            
        except Exception as e:
            print(f"❌ XDoG线条提取失败: {e}")
            # 回退到增强的边缘检测
            try:
                # 使用更敏感的Canny参数
                edges = cv2.Canny(gray_img, 30, 100)
                # 轻微膨胀连接断开的线条
                kernel = np.ones((2,2), np.uint8)
                edges = cv2.dilate(edges, kernel, iterations=1)
                print("⚠️ 使用增强Canny边缘检测作为回退方案")
                return edges
            except Exception as e2:
                print(f"❌ 回退方案也失败: {e2}")
                return np.zeros_like(gray_img)
    
    def multi_scale_fusion(self, img_bgr, processing_func):
        """
        多尺度融合处理 - 金字塔处理保持全局和局部特征
        
        Args:
            img_bgr: 输入图像
            processing_func: 处理函数
            
        Returns:
            fused_result: 融合结果
        """
        try:
            print(f"🔧 执行多尺度融合处理，层数: {self.pyramid_levels}")
            
            # 构建高斯金字塔
            pyramid = [img_bgr.copy()]
            current_img = img_bgr.copy()
            
            # 下采样构建金字塔
            for i in range(1, self.pyramid_levels):
                h, w = current_img.shape[:2]
                new_h, new_w = int(h * self.scale_factor), int(w * self.scale_factor)
                current_img = cv2.resize(current_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                pyramid.append(current_img)
            
            # 在每一层进行处理
            processed_pyramid = []
            for i, level_img in enumerate(pyramid):
                print(f"   处理第 {i+1} 层，尺寸: {level_img.shape[:2]}")
                processed_level = processing_func(level_img)
                processed_pyramid.append(processed_level)
            
            # 上采样并融合
            result = processed_pyramid[-1].copy().astype(np.float32)
            
            for i in range(len(pyramid) - 2, -1, -1):
                h, w = pyramid[i].shape[:2]
                result_upsampled = cv2.resize(result, (w, h), interpolation=cv2.INTER_LINEAR)
                
                # 加权融合 - 增强细节保持
                weight = 0.6  # 降低上一层权重，增强当前层细节
                result = result_upsampled * weight + processed_pyramid[i].astype(np.float32) * (1 - weight)
            
            fused_result = np.clip(result, 0, 255).astype(np.uint8)
            
            print("✅ 多尺度融合完成")
            return fused_result
            
        except Exception as e:
            print(f"❌ 多尺度融合失败: {e}")
            # 回退到单层处理
            print("⚠️ 使用单层处理作为回退方案")
            return processing_func(img_bgr)
    
    def intelligent_color_mapping(self, img_bgr, reference_images=None):
        """
        智能色彩映射 - 基于宫崎骏参考图片的专业调色
        
        Args:
            img_bgr: 输入图像
            reference_images: 参考图像列表
            
        Returns:
            color_mapped_img: 色彩映射后的图像
        """
        try:
            print("🔧 执行智能色彩映射")
            
            # 如果没有提供参考图像，使用默认的宫崎骏色彩风格
            if reference_images is None or (isinstance(reference_images, (list, tuple)) and len(reference_images) == 0):
                return self._apply_default_ghibli_palette(img_bgr)
            
            # 分析参考图像的色彩分布
            ref_hists = []
            for ref_img in reference_images:
                ref_hist = self._analyze_color_distribution(ref_img)
                ref_hists.append(ref_hist)
            
            # 平均参考图像的色彩分布
            avg_ref_hist = np.mean(ref_hists, axis=0)
            
            # 应用色彩映射
            color_mapped_img = self._apply_color_transfer(img_bgr, avg_ref_hist)
            
            print("✅ 智能色彩映射完成")
            return color_mapped_img
            
        except Exception as e:
            print(f"❌ 智能色彩映射失败: {e}")
            # 回退到默认宫崎骏调色板
            print("⚠️ 使用默认宫崎骏调色板作为回退方案")
            return self._apply_default_ghibli_palette(img_bgr)
    
    def _analyze_color_distribution(self, img_bgr):
        """分析图像的色彩分布"""
        # 转换到LAB色彩空间进行分析
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        
        # 计算每个通道的直方图
        hist_l = cv2.calcHist([lab], [0], None, [256], [0, 256])
        hist_a = cv2.calcHist([lab], [1], None, [256], [0, 256])
        hist_b = cv2.calcHist([lab], [2], None, [256], [0, 256])
        
        # 归一化直方图
        hist_l = hist_l / (img_bgr.shape[0] * img_bgr.shape[1])
        hist_a = hist_a / (img_bgr.shape[0] * img_bgr.shape[1])
        hist_b = hist_b / (img_bgr.shape[0] * img_bgr.shape[1])
        
        return np.array([hist_l.flatten(), hist_a.flatten(), hist_b.flatten()])
    
    def _apply_color_transfer(self, img_bgr, target_hist):
        """应用色彩传递"""
        # 简化的色彩传递实现
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 应用直方图匹配
        l_matched = self._histogram_match(l, target_hist[0])
        a_matched = self._histogram_match(a, target_hist[1])
        b_matched = self._histogram_match(b, target_hist[2])
        
        # 重新组合
        lab_matched = cv2.merge([l_matched, a_matched, b_matched])
        result = cv2.cvtColor(lab_matched, cv2.COLOR_LAB2BGR)
        
        return result
    
    def _histogram_match(self, source, target_hist):
        """直方图匹配"""
        source_flat = source.flatten()
        source_hist = cv2.calcHist([source], [0], None, [256], [0, 256])
        source_hist = source_hist / source_flat.size
        
        # 计算累积分布函数
        source_cdf = np.cumsum(source_hist)
        target_cdf = np.cumsum(target_hist)
        
        # 创建映射表
        lut = np.zeros(256, dtype=source.dtype)
        for i in range(256):
            diff = np.abs(target_cdf - source_cdf[i])
            lut[i] = np.argmin(diff)
        
        # 应用映射
        matched = cv2.LUT(source, lut)
        return matched
    
    def _apply_default_ghibli_palette(self, img_bgr):
        """应用默认的宫崎骏调色板 - 保持原图色彩特征"""
        # 转换到LAB色彩空间进行更精细的色彩控制
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 保持原有亮度分布，仅做轻微增强
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        # 混合原始和增强的亮度，保持自然感
        l_final = cv2.addWeighted(l, 0.7, l_enhanced, 0.3, 0)
        
        # 轻微调整色彩通道，增强宫崎骏风格但不过度改变
        # a通道（绿-红）轻微偏暖
        a_adjusted = cv2.add(a, 5)
        a_adjusted = np.clip(a_adjusted, 0, 255)
        
        # b通道（蓝-黄）轻微调整
        b_adjusted = cv2.add(b, 3)
        b_adjusted = np.clip(b_adjusted, 0, 255)
        
        # 重新组合LAB图像
        lab_enhanced = cv2.merge([l_final, a_adjusted, b_adjusted])
        
        # 转换回BGR
        result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        # 转换到HSV进行最终调色
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # 轻微增强饱和度，保持自然
        s_enhanced = cv2.add(s, 10)
        s_enhanced = np.clip(s_enhanced, 0, 220)
        
        # 轻微调整亮度，避免过度曝光
        v_enhanced = cv2.add(v, 8)
        v_enhanced = np.clip(v_enhanced, 0, 245)
        
        # 混合原始和调整后的饱和度、亮度
        s_final = cv2.addWeighted(s, 0.8, s_enhanced, 0.2, 0)
        v_final = cv2.addWeighted(v, 0.9, v_enhanced, 0.1, 0)
        
        # 重新组合HSV
        hsv_final = cv2.merge([h, s_final, v_final])
        final_result = cv2.cvtColor(hsv_final, cv2.COLOR_HSV2BGR)
        
        # 与原图进行色彩保持混合
        final_result = cv2.addWeighted(img_bgr, 0.6, final_result, 0.4, 0)
        
        return final_result
    
    def process_anime_style(self, img_bgr, use_slic=True, use_xdog=True, use_multiscale=True, use_color_mapping=True):
        """
        完整的动漫风格处理流程
        
        Args:
            img_bgr: 输入图像
            use_slic: 是否使用SLIC超像素分割
            use_xdog: 是否使用XDoG线条提取
            use_multiscale: 是否使用多尺度融合
            use_color_mapping: 是否使用智能色彩映射
            
        Returns:
            result_img: 处理后的图像
        """
        print("🎨 开始高级动漫风格处理...")
        
        result = img_bgr.copy()
        
        try:
            # 1. SLIC超像素分割
            if use_slic:
                print("\n📐 步骤1: SLIC超像素分割")
                result, segments = self.slic_superpixel_segmentation(result)
            else:
                segments = None
            
            # 2. 定义多尺度处理函数
            def multiscale_process(img):
                # XDoG线条提取
                if use_xdog:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    xdog_edges = self.xdog_line_extraction(gray)
                    edges_colored = cv2.cvtColor(xdog_edges, cv2.COLOR_GRAY2BGR)
                    
                    # 线条叠加 - 增强线条清晰度
                    img_processed = cv2.addWeighted(img, 0.9, edges_colored, 0.1, 0)
                    return img_processed
                else:
                    return img
            
            # 3. 多尺度融合处理
            if use_multiscale:
                print("\n🔀 步骤2: 多尺度融合处理")
                result = self.multi_scale_fusion(result, multiscale_process)
            elif use_xdog:
                print("\n✏️ 步骤2: XDoG线条提取")
                gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
                xdog_edges = self.xdog_line_extraction(gray)
                edges_colored = cv2.cvtColor(xdog_edges, cv2.COLOR_GRAY2BGR)
                result = cv2.addWeighted(result, 0.9, edges_colored, 0.1, 0)
            
            # 4. 智能色彩映射
            if use_color_mapping:
                print("\n🎨 步骤3: 智能色彩映射")
                result = self.intelligent_color_mapping(result)
            
            # 5. 最终优化
            print("\n✨ 步骤4: 最终优化")
            result = self._final_optimization(result)
            
            print("✅ 高级动漫风格处理完成")
            return result
            
        except Exception as e:
            print(f"❌ 高级处理失败: {e}")
            return img_bgr
    
    def _final_optimization(self, img_bgr):
        """最终优化处理"""
        # 轻微锐化
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(img_bgr, -1, kernel)
        
        # 混合锐化结果
        result = cv2.addWeighted(img_bgr, 0.9, sharpened, 0.1, 0)
        
        # 轻微降噪
        denoised = cv2.fastNlMeansDenoisingColored(result, None, 3, 3, 7, 21)
        
        return denoised

# 创建全局处理器实例
advanced_processor = AdvancedAnimeProcessor()

def process_with_advanced_techniques(img_bgr, **kwargs):
    """
    使用高级技术处理图像的便捷函数
    
    Args:
        img_bgr: 输入图像
        **kwargs: 处理参数
        
    Returns:
        处理后的图像
    """
    return advanced_processor.process_anime_style(img_bgr, **kwargs)