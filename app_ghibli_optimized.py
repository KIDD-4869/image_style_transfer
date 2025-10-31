#!/usr/bin/env python3
"""
基于宫崎骏参考图片优化的风格转换模型
"""

import cv2
import numpy as np
from PIL import Image
import os
import json
import io
import base64
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# 创建上传目录
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class GhibliOptimizedStyleTransfer:
    """基于宫崎骏参考图片优化的风格转换模型"""
    
    def __init__(self):
        self.optimization_params = self._load_optimization_params()
        self.ghibli_features = self._load_ghibli_features()
        
    def _load_optimization_params(self):
        """加载优化参数"""
        params_file = 'temp/optimization_parameters.json'
        if os.path.exists(params_file):
            with open(params_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # 默认参数
            return {
                'color_optimization': {
                    'saturation_boost': 1.3,
                    'brightness_adjust': 1.1,
                    'contrast_enhance': 1.2
                },
                'texture_optimization': {
                    'smoothing_strength': 9,
                    'edge_preservation': 0.8,
                    'detail_enhancement': 1.1
                },
                'character_optimization': {
                    'face_enhancement': True,
                    'skin_smoothing': 0.7,
                    'eye_enhancement': 1.3,
                    'sharpness_boost': 1.2
                },
                'lighting_optimization': {
                    'soft_lighting': True,
                    'shadow_reduction': 0.3,
                    'highlight_enhancement': 1.1,
                    'contrast_adjust': 1.1
                }
            }
    
    def _load_ghibli_features(self):
        """加载宫崎骏风格特征"""
        features_file = 'temp/ghibli_analysis_results.json'
        if os.path.exists(features_file):
            with open(features_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return {}
    
    def apply_ghibli_style(self, image):
        """应用基于真实宫崎骏图片优化的风格"""
        # 将PIL图像转换为numpy数组
        img_np = np.array(image)
        
        # 转换为BGR格式
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        
        # 1. 高质量预处理（保持原始细节）
        processed = self._high_quality_preprocess(img_bgr)
        
        # 2. 智能人物区域检测和优化（重点）
        character_optimized = self._optimize_character_regions(processed)
        
        # 3. 基于宫崎骏风格的色彩映射
        ghibli_colors = self._ghibli_style_color_mapping(character_optimized)
        
        # 4. 纹理优化和细节增强
        texture_enhanced = self._enhance_textures_and_details(ghibli_colors)
        
        # 5. 光影效果优化
        final = self._apply_optimized_lighting(texture_enhanced)
        
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
    
    def _optimize_character_regions(self, img):
        """智能人物区域优化（重点改进）"""
        params = self.optimization_params['character_optimization']
        
        # 检测人物区域
        character_mask = self._detect_character_regions(img)
        
        # 对人物区域进行特殊处理
        if np.sum(character_mask) > 0:
            # 人物区域：保持清晰度，轻微平滑
            character_region = cv2.bitwise_and(img, img, mask=character_mask)
            
            # 对人物区域应用优化
            optimized_character = self._optimize_character_features(character_region)
            
            # 背景区域：应用更强的动漫化效果
            background_mask = cv2.bitwise_not(character_mask)
            background_region = cv2.bitwise_and(img, img, mask=background_mask)
            
            # 对背景应用更强的平滑效果
            background_smoothed = cv2.bilateralFilter(
                background_region, 
                d=15, sigmaColor=80, sigmaSpace=80
            )
            
            # 合并人物和背景
            result = cv2.add(optimized_character, background_smoothed)
        else:
            # 如果没有检测到人物区域，对整个图像应用统一处理
            result = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
        
        return result
    
    def _detect_character_regions(self, img):
        """检测人物区域"""
        h, w = img.shape[:2]
        
        # 使用多种方法检测人物区域
        
        # 方法1：皮肤颜色检测
        skin_mask = self._detect_skin_regions(img)
        
        # 方法2：人脸区域检测（如果可用）
        face_mask = self._detect_face_regions(img)
        
        # 方法3：基于亮度和对比度的区域检测
        brightness_mask = self._detect_bright_regions(img)
        
        # 合并多个检测结果
        combined_mask = cv2.bitwise_or(skin_mask, face_mask)
        combined_mask = cv2.bitwise_or(combined_mask, brightness_mask)
        
        # 形态学操作，填充空洞，平滑边缘
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        return combined_mask
    
    def _detect_skin_regions(self, img):
        """检测皮肤区域"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 皮肤颜色范围（HSV空间）
        lower_skin1 = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin1 = np.array([20, 255, 255], dtype=np.uint8)
        lower_skin2 = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin2 = np.array([20, 255, 255], dtype=np.uint8)
        
        skin_mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
        skin_mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
        
        skin_mask = cv2.bitwise_or(skin_mask1, skin_mask2)
        
        return skin_mask
    
    def _detect_face_regions(self, img):
        """检测人脸区域"""
        # 创建空掩码（在实际应用中应该使用真正的人脸检测）
        h, w = img.shape[:2]
        return np.zeros((h, w), dtype=np.uint8)
    
    def _detect_bright_regions(self, img):
        """检测明亮区域（通常包含人物）"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 自适应阈值，检测明亮区域
        bright_mask = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        return bright_mask
    
    def _optimize_character_features(self, character_region):
        """优化人物特征"""
        params = self.optimization_params['character_optimization']
        
        # 对人物区域应用轻微平滑，保持细节
        smoothed = cv2.bilateralFilter(
            character_region, 
            d=5,  # 较小的d值，保持细节
            sigmaColor=30, 
            sigmaSpace=30
        )
        
        # 增强清晰度
        if params.get('sharpness_boost', 1.0) > 1.0:
            sharpened = self._sharpen_image(smoothed, params['sharpness_boost'])
        else:
            sharpened = smoothed
        
        return sharpened
    
    def _ghibli_style_color_mapping(self, img):
        """宫崎骏风格色彩映射"""
        params = self.optimization_params['color_optimization']
        
        # 转换为HSV色彩空间
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # 增强饱和度
        s = cv2.multiply(s, params['saturation_boost'])
        s = np.clip(s, 0, 255)
        
        # 调整亮度
        v = cv2.multiply(v, params['brightness_adjust'])
        v = np.clip(v, 0, 255)
        
        # 增强对比度
        v = cv2.convertScaleAbs(v, alpha=params['contrast_enhance'])
        
        # 合并通道
        hsv_enhanced = cv2.merge([h, s, v])
        
        # 转换回BGR
        result = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
        
        return result
    
    def _enhance_textures_and_details(self, img):
        """增强纹理和细节"""
        params = self.optimization_params['texture_optimization']
        
        # 边缘保留平滑
        smoothed = cv2.bilateralFilter(
            img, 
            d=int(params['smoothing_strength']),
            sigmaColor=75, 
            sigmaSpace=75
        )
        
        # 细节增强
        if params.get('detail_enhancement', 1.0) > 1.0:
            # 提取高频细节
            detail = cv2.subtract(img, smoothed)
            enhanced_detail = cv2.multiply(detail, params['detail_enhancement'])
            
            # 合并细节
            result = cv2.add(smoothed, enhanced_detail)
        else:
            result = smoothed
        
        return result
    
    def _apply_optimized_lighting(self, img):
        """应用优化的光影效果"""
        params = self.optimization_params['lighting_optimization']
        
        if params.get('soft_lighting', True):
            # 创建柔和的光照效果
            h, w = img.shape[:2]
            
            # 创建中心亮四周暗的光照遮罩
            y, x = np.ogrid[:h, :w]
            center_y, center_x = h / 2, w / 2
            
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_distance = np.sqrt(center_x**2 + center_y**2)
            
            # 创建光照遮罩
            light_mask = 1.0 - (distance / max_distance) * 0.2
            light_mask = np.clip(light_mask, 0.8, 1.0)
            
            # 应用光照效果
            result = img.astype(np.float32) * light_mask[:,:,np.newaxis]
            result = np.clip(result, 0, 255).astype(np.uint8)
        else:
            result = img
        
        return result
    
    def _sharpen_image(self, img, strength=1.0):
        """锐化图像"""
        # 创建锐化核
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]]) * strength
        
        sharpened = cv2.filter2D(img, -1, kernel)
        
        # 与原图混合，保持自然感
        result = cv2.addWeighted(img, 0.7, sharpened, 0.3, 0)
        
        return result

# 创建优化后的模型实例
optimized_model = GhibliOptimizedStyleTransfer()

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
        
        # 应用优化的宫崎骏风格
        result_image = optimized_model.apply_ghibli_style(image)
        
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
    app.run(debug=True, host='0.0.0.0', port=5004)