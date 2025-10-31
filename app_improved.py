#!/usr/bin/env python3
"""
改进的宫崎骏风格转换模型 - 基于temp文件夹中的参考图片学习
"""

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import io
import base64
from flask import Flask, render_template, request, jsonify
import os
import json
import glob

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# 创建上传目录
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class ImprovedGhibliStyleTransfer:
    """改进的宫崎骏风格转换模型 - 基于参考图片学习"""
    
    def __init__(self):
        self.temp_folder = 'temp'
        self.ghibli_features = self._load_ghibli_features()
        self.ghibli_params = self._load_processing_params()
        
    def _load_ghibli_features(self):
        """加载宫崎骏风格特征"""
        features_file = os.path.join(self.temp_folder, 'ghibli_style_features.json')
        if os.path.exists(features_file):
            with open(features_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # 默认宫崎骏风格特征
            return {
                "color_palette": {
                    "sky_blue": [135, 206, 235],
                    "grass_green": [144, 238, 144],
                    "character_skin": [255, 218, 185],
                    "hair_brown": [165, 42, 42],
                    "dress_pink": [255, 192, 203],
                },
                "lighting_characteristics": {
                    "soft_shadows": True,
                    "warm_tones": True,
                    "dreamy_atmosphere": True
                }
            }
    
    def _load_processing_params(self):
        """加载处理参数"""
        params_file = os.path.join(self.temp_folder, 'ghibli_processing_params.json')
        if os.path.exists(params_file):
            with open(params_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # 默认处理参数
            return {
                "bilateral_filter": {"d": 9, "sigmaColor": 75, "sigmaSpace": 75},
                "edge_preservation": {"strength": 0.8},
                "color_enhancement": {
                    "saturation_boost": 1.3,
                    "brightness_adjust": 1.1,
                    "contrast_enhance": 1.2
                }
            }
    
    def _analyze_reference_images(self):
        """分析temp文件夹中的参考图片"""
        reference_images = []
        
        # 支持的图片格式
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        
        for ext in image_extensions:
            pattern = os.path.join(self.temp_folder, ext)
            reference_images.extend(glob.glob(pattern))
        
        if not reference_images:
            print("⚠️  temp文件夹中没有找到参考图片，使用默认宫崎骏风格")
            return None
        
        print(f"📊 找到 {len(reference_images)} 张参考图片")
        
        # 分析参考图片的色彩特征
        color_features = self._extract_color_features(reference_images)
        
        return color_features
    
    def _extract_color_features(self, image_paths):
        """从参考图片中提取色彩特征"""
        color_features = {
            'hue_distribution': [],
            'saturation_levels': [],
            'brightness_levels': [],
            'dominant_colors': []
        }
        
        for img_path in image_paths:
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                    
                # 转换为HSV色彩空间
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                
                # 分析色调分布
                hue_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
                color_features['hue_distribution'].append(hue_hist)
                
                # 分析饱和度和亮度
                saturation = np.mean(hsv[:,:,1])
                brightness = np.mean(hsv[:,:,2])
                color_features['saturation_levels'].append(saturation)
                color_features['brightness_levels'].append(brightness)
                
                # 提取主色调
                pixels = img.reshape(-1, 3)
                dominant_color = np.mean(pixels, axis=0)
                color_features['dominant_colors'].append(dominant_color)
                
            except Exception as e:
                print(f"❌ 分析图片 {img_path} 时出错: {e}")
        
        return color_features
    
    def apply_ghibli_style(self, image):
        """应用改进的宫崎骏风格"""
        
        # 分析参考图片（如果存在）
        reference_features = self._analyze_reference_images()
        
        # 将PIL图像转换为numpy数组
        img_np = np.array(image)
        
        # 转换为BGR格式
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        
        # 1. 高质量预处理
        processed = self._high_quality_preprocess(img_bgr)
        
        # 2. 基于参考图片的色彩调整
        if reference_features:
            processed = self._adjust_colors_based_on_reference(processed, reference_features)
        
        # 3. 智能边缘保留平滑
        smoothed = self._smart_edge_preserving_smooth(processed)
        
        # 4. 宫崎骏风格色彩映射
        ghibli_colors = self._ghibli_style_color_mapping(smoothed)
        
        # 5. 细节增强和恢复
        detailed = self._enhance_and_preserve_details(ghibli_colors, processed)
        
        # 6. 梦幻光影效果
        final = self._apply_dreamy_lighting(detailed)
        
        # 转换回RGB格式
        result_rgb = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
        
        return result_rgb
    
    def _high_quality_preprocess(self, img):
        """高质量预处理"""
        h, w = img.shape[:2]
        
        # 保持原始分辨率，仅在过大时调整
        max_size = 2000
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        return img
    
    def _adjust_colors_based_on_reference(self, img, reference_features):
        """基于参考图片调整色彩"""
        # 转换为LAB色彩空间进行更精确的色彩调整
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        
        # 调整亮度和对比度
        l, a, b = cv2.split(lab)
        
        # 基于参考图片的亮度特征调整
        if reference_features['brightness_levels']:
            target_brightness = np.mean(reference_features['brightness_levels'])
            current_brightness = np.mean(l)
            brightness_ratio = target_brightness / current_brightness if current_brightness > 0 else 1.0
            l = cv2.multiply(l, brightness_ratio)
        
        # 合并通道
        lab_adjusted = cv2.merge([l, a, b])
        
        # 转换回BGR
        result = cv2.cvtColor(lab_adjusted, cv2.COLOR_LAB2BGR)
        
        return result
    
    def _smart_edge_preserving_smooth(self, img):
        """智能边缘保留平滑"""
        params = self.ghibli_params['bilateral_filter']
        
        # 双边滤波，保留边缘
        smoothed = cv2.bilateralFilter(
            img, 
            params['d'], 
            params['sigmaColor'], 
            params['sigmaSpace']
        )
        
        return smoothed
    
    def _ghibli_style_color_mapping(self, img):
        """宫崎骏风格色彩映射"""
        # 转换为HSV色彩空间
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 增强饱和度（宫崎骏风格通常色彩鲜艳）
        h, s, v = cv2.split(hsv)
        
        params = self.ghibli_params['color_enhancement']
        s = cv2.multiply(s, params['saturation_boost'])
        v = cv2.multiply(v, params['brightness_adjust'])
        
        # 限制值范围
        s = np.clip(s, 0, 255)
        v = np.clip(v, 0, 255)
        
        # 合并通道
        hsv_enhanced = cv2.merge([h, s, v])
        
        # 转换回BGR
        result = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
        
        return result
    
    def _enhance_and_preserve_details(self, img, original):
        """增强和保留细节"""
        # 提取原图的高频细节
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 计算细节差异
        detail_diff = cv2.subtract(original_gray, img_gray)
        
        # 增强细节
        enhanced_detail = cv2.multiply(detail_diff, 0.3)
        
        # 将细节添加回结果
        result_gray = cv2.add(img_gray, enhanced_detail)
        
        # 将灰度细节应用到彩色图像
        result = img.copy()
        for i in range(3):
            result[:,:,i] = cv2.addWeighted(
                result[:,:,i], 0.7, 
                result_gray, 0.3, 0
            )
        
        return result
    
    def _apply_dreamy_lighting(self, img):
        """应用梦幻光影效果"""
        # 创建柔和的光照效果
        h, w = img.shape[:2]
        
        # 创建中心亮四周暗的光照遮罩
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h / 2, w / 2
        
        # 计算距离中心的距离
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        
        # 创建光照遮罩（中心亮，四周暗）
        light_mask = 1.0 - (distance / max_distance) * 0.3
        light_mask = np.clip(light_mask, 0.7, 1.0)
        
        # 应用光照效果
        result = img.astype(np.float32) * light_mask[:,:,np.newaxis]
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result

# 创建改进的模型实例
improved_model = ImprovedGhibliStyleTransfer()

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """处理文件上传和风格转换"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': '没有选择文件'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': '没有选择文件'})
        
        # 读取图片
        image = Image.open(file.stream)
        
        # 应用改进的宫崎骏风格
        result_image = improved_model.apply_ghibli_style(image)
        
        # 转换为PIL图像
        result_pil = Image.fromarray(result_image)
        
        # 转换为base64
        buffered = io.BytesIO()
        result_pil.save(buffered, format="JPEG", quality=95)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'image': f"data:image/jpeg;base64,{img_str}"
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5003)