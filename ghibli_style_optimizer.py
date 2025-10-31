#!/usr/bin/env python3
"""
宫崎骏风格优化器 - 基于真实宫崎骏图片训练和改进模型
"""

import cv2
import numpy as np
from PIL import Image
import os
import glob
import json
from collections import defaultdict
import matplotlib.pyplot as plt

class GhibliStyleOptimizer:
    """宫崎骏风格优化器"""
    
    def __init__(self):
        self.reference_folder = 'temp'
        self.analysis_results = {}
        
    def collect_ghibli_references(self):
        """收集宫崎骏参考图片并分析特征"""
        print("🔍 收集和分析宫崎骏风格参考图片...")
        
        # 支持的图片格式
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
        reference_images = []
        
        for ext in image_extensions:
            pattern = os.path.join(self.reference_folder, ext)
            reference_images.extend(glob.glob(pattern))
        
        if not reference_images:
            print("⚠️  没有找到宫崎骏参考图片，请将图片放入temp文件夹")
            return False
        
        print(f"📊 找到 {len(reference_images)} 张参考图片")
        
        # 分析每张图片的特征
        self.analysis_results = self._analyze_ghibli_features(reference_images)
        
        # 保存分析结果
        self._save_analysis_results()
        
        return True
    
    def _analyze_ghibli_features(self, image_paths):
        """分析宫崎骏风格特征"""
        analysis = {
            'color_analysis': defaultdict(list),
            'texture_analysis': defaultdict(list),
            'character_analysis': defaultdict(list),
            'lighting_analysis': defaultdict(list),
            'composition_analysis': defaultdict(list)
        }
        
        for img_path in image_paths:
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                print(f"📖 分析: {os.path.basename(img_path)}")
                
                # 色彩分析
                color_features = self._analyze_colors(img)
                for key, value in color_features.items():
                    analysis['color_analysis'][key].append(value)
                
                # 纹理分析
                texture_features = self._analyze_textures(img)
                for key, value in texture_features.items():
                    analysis['texture_analysis'][key].append(value)
                
                # 人物分析（重点）
                character_features = self._analyze_characters(img)
                for key, value in character_features.items():
                    analysis['character_analysis'][key].append(value)
                
                # 光影分析
                lighting_features = self._analyze_lighting(img)
                for key, value in lighting_features.items():
                    analysis['lighting_analysis'][key].append(value)
                
                # 构图分析
                composition_features = self._analyze_composition(img)
                for key, value in composition_features.items():
                    analysis['composition_analysis'][key].append(value)
                    
            except Exception as e:
                print(f"❌ 分析 {img_path} 时出错: {e}")
        
        return analysis
    
    def _analyze_colors(self, img):
        """分析色彩特征"""
        # 转换为HSV色彩空间
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 分析色调分布
        hue_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        
        # 分析饱和度和亮度
        saturation = np.mean(hsv[:,:,1])
        brightness = np.mean(hsv[:,:,2])
        
        # 提取主色调
        pixels = img.reshape(-1, 3)
        dominant_colors = self._extract_dominant_colors(pixels, 5)
        
        return {
            'hue_distribution': hue_hist.flatten().tolist(),
            'saturation': float(saturation),
            'brightness': float(brightness),
            'dominant_colors': dominant_colors
        }
    
    def _analyze_textures(self, img):
        """分析纹理特征"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 计算纹理特征
        # 使用局部二值模式(LBP)分析纹理
        lbp = self._compute_lbp(gray)
        
        # 计算梯度特征
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        return {
            'texture_variance': float(np.var(gray)),
            'gradient_strength': float(np.mean(gradient_magnitude))
        }
    
    def _analyze_characters(self, img):
        """分析人物特征（重点优化区域）"""
        # 使用人脸检测（如果可用）
        face_features = self._detect_faces(img)
        
        # 分析皮肤区域
        skin_features = self._analyze_skin_regions(img)
        
        # 分析头发区域
        hair_features = self._analyze_hair_regions(img)
        
        return {
            'face_detected': len(face_features) > 0,
            'skin_tone': skin_features.get('average_skin_tone', [0, 0, 0]),
            'hair_color': hair_features.get('average_hair_color', [0, 0, 0]),
            'character_sharpness': float(self._calculate_sharpness(img))
        }
    
    def _analyze_lighting(self, img):
        """分析光影特征"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 分析光照分布
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        # 计算对比度
        contrast = np.std(gray)
        
        # 分析阴影和高光区域
        shadow_threshold = 50
        highlight_threshold = 200
        
        shadow_pixels = np.sum(gray < shadow_threshold)
        highlight_pixels = np.sum(gray > highlight_threshold)
        total_pixels = gray.size
        
        return {
            'contrast': float(contrast),
            'shadow_ratio': float(shadow_pixels / total_pixels),
            'highlight_ratio': float(highlight_pixels / total_pixels)
        }
    
    def _analyze_composition(self, img):
        """分析构图特征"""
        h, w = img.shape[:2]
        
        # 分析图像中心区域
        center_region = img[h//4:3*h//4, w//4:3*w//4]
        center_brightness = np.mean(cv2.cvtColor(center_region, cv2.COLOR_BGR2GRAY))
        
        return {
            'aspect_ratio': float(w / h),
            'center_brightness': float(center_brightness)
        }
    
    def _extract_dominant_colors(self, pixels, n_colors):
        """提取主色调"""
        # 使用K-means聚类提取主色调
        pixels_float = np.float32(pixels)
        
        # 简化版本：直接计算平均颜色
        if len(pixels) > 0:
            avg_color = np.mean(pixels, axis=0)
            return [avg_color.tolist()]
        
        return [[0, 0, 0]]
    
    def _compute_lbp(self, gray):
        """计算局部二值模式"""
        # 简化的LBP计算
        radius = 1
        n_points = 8 * radius
        
        # 使用简单的纹理方差代替复杂的LBP计算
        return np.var(gray)
    
    def _detect_faces(self, img):
        """检测人脸"""
        # 使用OpenCV的人脸检测（如果可用）
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 尝试加载人脸检测器
        face_cascade = cv2.CascadeClassifier()
        
        # 简化的面部区域检测
        # 在实际应用中，这里应该使用真正的人脸检测
        return []
    
    def _analyze_skin_regions(self, img):
        """分析皮肤区域"""
        # 简化的皮肤颜色检测
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 定义皮肤颜色范围（HSV空间）
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # 创建皮肤掩码
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # 提取皮肤区域
        skin_pixels = img[skin_mask > 0]
        
        if len(skin_pixels) > 0:
            avg_skin_tone = np.mean(skin_pixels, axis=0)
        else:
            avg_skin_tone = [0, 0, 0]
        
        return {'average_skin_tone': avg_skin_tone.tolist()}
    
    def _analyze_hair_regions(self, img):
        """分析头发区域"""
        # 简化的头发颜色检测
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 假设暗色区域可能是头发
        hair_mask = gray < 80
        hair_pixels = img[hair_mask]
        
        if len(hair_pixels) > 0:
            avg_hair_color = np.mean(hair_pixels, axis=0)
        else:
            avg_hair_color = [0, 0, 0]
        
        return {'average_hair_color': avg_hair_color.tolist()}
    
    def _calculate_sharpness(self, img):
        """计算图像清晰度"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 使用拉普拉斯算子计算清晰度
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = np.var(laplacian)
        
        return sharpness
    
    def _save_analysis_results(self):
        """保存分析结果"""
        # 转换为可JSON序列化的格式
        serializable_results = {}
        
        for category, features in self.analysis_results.items():
            serializable_results[category] = {}
            for feature_name, values in features.items():
                # 处理numpy数组和标量
                if isinstance(values, list):
                    serializable_results[category][feature_name] = [
                        float(v) if isinstance(v, (np.floating, float)) else v 
                        for v in values
                    ]
                else:
                    serializable_results[category][feature_name] = values
        
        # 保存到文件
        output_file = os.path.join(self.reference_folder, 'ghibli_analysis_results.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 分析结果已保存到: {output_file}")
    
    def generate_optimization_parameters(self):
        """生成优化参数"""
        if not self.analysis_results:
            print("❌ 请先运行collect_ghibli_references()收集参考图片")
            return None
        
        print("🎯 生成宫崎骏风格优化参数...")
        
        # 基于分析结果生成优化参数
        optimization_params = {
            'color_optimization': self._generate_color_params(),
            'texture_optimization': self._generate_texture_params(),
            'character_optimization': self._generate_character_params(),
            'lighting_optimization': self._generate_lighting_params()
        }
        
        # 保存优化参数
        params_file = os.path.join(self.reference_folder, 'optimization_parameters.json')
        with open(params_file, 'w', encoding='utf-8') as f:
            json.dump(optimization_params, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 优化参数已保存到: {params_file}")
        return optimization_params
    
    def _generate_color_params(self):
        """生成色彩优化参数"""
        color_data = self.analysis_results.get('color_analysis', {})
        
        # 计算平均饱和度和亮度
        avg_saturation = np.mean(color_data.get('saturation', [0.5]))
        avg_brightness = np.mean(color_data.get('brightness', [0.5]))
        
        return {
            'saturation_boost': max(1.0, 1.5 - avg_saturation / 128),
            'brightness_adjust': max(1.0, 1.3 - avg_brightness / 128),
            'contrast_enhance': 1.2
        }
    
    def _generate_texture_params(self):
        """生成纹理优化参数"""
        texture_data = self.analysis_results.get('texture_analysis', {})
        
        avg_gradient = np.mean(texture_data.get('gradient_strength', [10]))
        
        return {
            'smoothing_strength': min(15, max(5, avg_gradient / 5)),
            'edge_preservation': 0.8,
            'detail_enhancement': 1.1
        }
    
    def _generate_character_params(self):
        """生成人物优化参数"""
        character_data = self.analysis_results.get('character_analysis', {})
        
        avg_sharpness = np.mean(character_data.get('character_sharpness', [100]))
        
        return {
            'face_enhancement': True,
            'skin_smoothing': 0.7,
            'eye_enhancement': 1.3,
            'sharpness_boost': max(1.0, 200 / avg_sharpness)
        }
    
    def _generate_lighting_params(self):
        """生成光影优化参数"""
        lighting_data = self.analysis_results.get('lighting_analysis', {})
        
        avg_contrast = np.mean(lighting_data.get('contrast', [40]))
        
        return {
            'soft_lighting': True,
            'shadow_reduction': 0.3,
            'highlight_enhancement': 1.1,
            'contrast_adjust': max(1.0, 60 / avg_contrast)
        }

def main():
    """主函数"""
    print("=" * 60)
    print("🎨 宫崎骏风格优化器")
    print("=" * 60)
    
    optimizer = GhibliStyleOptimizer()
    
    # 确保temp目录存在
    os.makedirs('temp', exist_ok=True)
    
    # 收集和分析参考图片
    if optimizer.collect_ghibli_references():
        # 生成优化参数
        optimizer.generate_optimization_parameters()
        
        print("\n📋 使用说明:")
        print("1. 将更多的宫崎骏风格图片放入temp文件夹")
        print("2. 重新运行此脚本来更新分析结果")
        print("3. 使用生成的优化参数改进风格转换模型")
    
    print("\n" + "=" * 60)
    print("✅ 优化完成")
    print("=" * 60)

if __name__ == '__main__':
    main()