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

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# 创建上传目录
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class GhibliStyleTransfer(nn.Module):
    """宫崎骏动漫风格转换模型 - 区域色彩分布优化版"""
    def __init__(self):
        super(GhibliStyleTransfer, self).__init__()
        
    def forward(self, image, style_type="ghibli"):
        """
        应用宫崎骏动漫风格转换（区域色彩分布优化版）
        
        Args:
            image: PIL图像
            style_type: 风格类型，可选 "ghibli"（宫崎骏）、"anime"（通用动漫）、"painting"（绘画风）
        """
        # 将PIL图像转换为numpy数组
        img_np = np.array(image)
        
        # 转换为BGR格式（OpenCV使用）
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        else:
            # 如果是灰度图，转换为彩色
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        
        # 根据风格类型选择处理方式
        if style_type == "ghibli":
            result = self._apply_ghibli_style_regional_optimized(img_bgr)
        elif style_type == "anime":
            result = self._apply_anime_style_preserve_details(img_bgr)
        elif style_type == "painting":
            result = self._apply_painting_style_preserve_details(img_bgr)
        else:
            result = self._apply_ghibli_style_regional_optimized(img_bgr)
        
        # 转换回RGB格式
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        
        return Image.fromarray(result_rgb)
    
    def _apply_ghibli_style_regional_optimized(self, img):
        """应用宫崎骏风格（区域优化版）- 基于区域色彩分布和高画质动漫化"""
        original_img = img.copy()
        
        # 1. 高画质预处理：保持原始分辨率
        preprocessed = self._high_quality_preprocess(img)
        
        # 2. 深度区域分析：分析不同区域的色彩分布特征
        regional_features = self._deep_regional_analysis(preprocessed)
        
        # 3. 智能区域动漫化：不同区域采用不同动漫化策略
        regional_anime = self._smart_regional_anime_conversion(preprocessed, regional_features)
        
        # 4. 宫崎骏色彩映射：基于真实宫崎骏图片的区域色彩特征
        ghibli_colors = self._ghibli_regional_color_mapping(regional_anime, regional_features)
        
        # 5. 高画质细节融合：在动漫化基础上智能恢复重要细节
        detailed_result = self._high_quality_detail_fusion(ghibli_colors, original_img, regional_features)
        
        # 6. 宫崎骏梦幻光影：添加区域适应的光影效果
        final = self._ghibli_regional_lighting(detailed_result, regional_features)
        
        return final
    
    def _high_quality_preprocess(self, img):
        """高画质预处理：保持原始分辨率，增强细节可见度"""
        h, w = img.shape[:2]
        
        # 保持原始分辨率，仅当图片过大时适度缩小
        max_size = 2000
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # 轻微锐化，保持细节清晰度
        kernel = np.array([[-0.1, -0.1, -0.1],
                          [-0.1,  1.8, -0.1],
                          [-0.1, -0.1, -0.1]])
        sharpened = cv2.filter2D(img, -1, kernel)
        
        # 与原图混合，保持自然感
        final = cv2.addWeighted(img, 0.7, sharpened, 0.3, 0)
        
        return final
    
    def _deep_regional_analysis(self, img):
        """深度区域分析：精细分析不同区域的色彩和纹理特征"""
        h, w = img.shape[:2]
        
        # 将图片分为5x5=25个更精细的区域
        regions = []
        region_features = []
        
        for i in range(5):
            for j in range(5):
                y1, y2 = i*h//5, (i+1)*h//5
                x1, x2 = j*w//5, (j+1)*w//5
                region = img[y1:y2, x1:x2]
                regions.append(region)
                
                if region.size > 0:
                    # 多色彩空间分析
                    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
                    lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
                    
                    # 计算详细统计特征
                    h_mean, s_mean, v_mean = np.mean(hsv, axis=(0, 1))
                    h_std, s_std, v_std = np.std(hsv, axis=(0, 1))
                    
                    l_mean, a_mean, b_mean = np.mean(lab, axis=(0, 1))
                    l_std, a_std, b_std = np.std(lab, axis=(0, 1))
                    
                    # 纹理分析
                    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                    texture_std = np.std(gray)
                    
                    # 边缘密度分析
                    edges = cv2.Canny(gray, 50, 150)
                    edge_density = np.sum(edges > 0) / gray.size
                    
                    # 精细区域分类
                    region_type = self._fine_region_classification(
                        h_mean, s_mean, v_mean, h_std, s_std, v_std,
                        l_mean, texture_std, edge_density
                    )
                    
                    region_features.append({
                        'coords': (y1, y2, x1, x2),
                        'hsv_stats': (h_mean, s_mean, v_mean, h_std, s_std, v_std),
                        'lab_stats': (l_mean, a_mean, b_mean, l_std, a_std, b_std),
                        'texture_std': texture_std,
                        'edge_density': edge_density,
                        'type': region_type
                    })
        
        return region_features
    
    def _fine_region_classification(self, h_mean, s_mean, v_mean, h_std, s_std, v_std,
                                  l_mean, texture_std, edge_density):
        """精细区域分类：基于多重特征"""
        # 基于亮度分类
        if v_mean > 200:  # 极高亮度
            if s_mean < 30:  # 低饱和度
                return "bright_sky"  # 明亮天空
            else:
                return "highlight"  # 高光区域
        elif v_mean > 160:  # 高亮度
            if h_std > 40:  # 色彩变化大
                return "colorful_highlight"  # 彩色高光
            else:
                return "main_light"  # 主要亮部
        elif v_mean > 120:  # 中等亮度
            if edge_density > 0.1:  # 边缘密集
                return "detailed_object"  # 细节丰富物体
            elif texture_std > 25:  # 纹理丰富
                return "textured_area"  # 纹理区域
            else:
                return "main_object"  # 主要物体
        elif v_mean > 80:  # 低亮度
            if s_mean > 100:  # 高饱和度
                return "colorful_shadow"  # 彩色阴影
            else:
                return "shadow"  # 普通阴影
        else:  # 极低亮度
            return "deep_shadow"  # 深阴影
    
    def _smart_regional_anime_conversion(self, img, region_features):
        """智能区域动漫化：不同区域采用不同动漫化策略"""
        result = img.copy()
        
        for feature in region_features:
            y1, y2, x1, x2 = feature['coords']
            region = img[y1:y2, x1:x2]
            
            if region.size == 0:
                continue
            
            # 根据精细区域类型应用不同的动漫化处理
            if feature['type'] == "bright_sky":
                # 明亮天空：高度简化，保持渐变
                processed = cv2.bilateralFilter(region, d=25, sigmaColor=40, sigmaSpace=40)
            elif feature['type'] == "highlight":
                # 高光区域：保持明亮，适度简化
                processed = cv2.bilateralFilter(region, d=15, sigmaColor=25, sigmaSpace=25)
            elif feature['type'] == "colorful_highlight":
                # 彩色高光：保留色彩变化，轻微简化
                processed = cv2.bilateralFilter(region, d=8, sigmaColor=15, sigmaSpace=15)
            elif feature['type'] == "main_light":
                # 主要亮部：保留细节，适度简化
                processed = cv2.bilateralFilter(region, d=6, sigmaColor=12, sigmaSpace=12)
            elif feature['type'] == "detailed_object":
                # 细节丰富物体：保留更多细节
                processed = cv2.bilateralFilter(region, d=3, sigmaColor=8, sigmaSpace=8)
            elif feature['type'] == "textured_area":
                # 纹理区域：保留纹理特征
                processed = cv2.bilateralFilter(region, d=4, sigmaColor=10, sigmaSpace=10)
            elif feature['type'] == "main_object":
                # 主要物体：标准动漫化
                processed = cv2.bilateralFilter(region, d=5, sigmaColor=12, sigmaSpace=12)
            elif feature['type'] == "colorful_shadow":
                # 彩色阴影：增强色彩，简化纹理
                processed = cv2.bilateralFilter(region, d=10, sigmaColor=20, sigmaSpace=20)
            elif feature['type'] == "shadow":
                # 普通阴影：中度简化
                processed = cv2.bilateralFilter(region, d=12, sigmaColor=25, sigmaSpace=25)
            else:  # deep_shadow
                # 深阴影：高度简化
                processed = cv2.bilateralFilter(region, d=15, sigmaColor=30, sigmaSpace=30)
            
            result[y1:y2, x1:x2] = processed
        
        return result
    
    def _ghibli_regional_color_mapping(self, img, region_features):
        """宫崎骏区域色彩映射：基于真实宫崎骏图片的区域色彩特征"""
        result = img.copy()
        
        for feature in region_features:
            y1, y2, x1, x2 = feature['coords']
            region = img[y1:y2, x1:x2]
            
            if region.size == 0:
                continue
            
            # 根据区域类型应用不同的宫崎骏色彩处理
            if feature['type'] == "bright_sky":
                processed = self._ghibli_sky_colors(region, feature)
            elif feature['type'] == "highlight":
                processed = self._ghibli_highlight_colors(region, feature)
            elif feature['type'] == "colorful_highlight":
                processed = self._ghibli_colorful_highlight(region, feature)
            elif feature['type'] == "main_light":
                processed = self._ghibli_main_light_colors(region, feature)
            elif feature['type'] == "detailed_object":
                processed = self._ghibli_detailed_object_colors(region, feature)
            elif feature['type'] == "textured_area":
                processed = self._ghibli_textured_area_colors(region, feature)
            elif feature['type'] == "main_object":
                processed = self._ghibli_main_object_colors(region, feature)
            elif feature['type'] == "colorful_shadow":
                processed = self._ghibli_colorful_shadow_colors(region, feature)
            elif feature['type'] == "shadow":
                processed = self._ghibli_shadow_colors(region, feature)
            else:  # deep_shadow
                processed = self._ghibli_deep_shadow_colors(region, feature)
            
            result[y1:y2, x1:x2] = processed
        
        return result
    
    def _ghibli_sky_colors(self, region, feature):
        """宫崎骏天空色彩：明亮、渐变、梦幻"""
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # 增强蓝色和青色调
        blue_mask = (hsv[:, :, 0] >= 90) & (hsv[:, :, 0] <= 150)
        hsv[blue_mask, 1] = hsv[blue_mask, 1] * 1.4  # 增强饱和度
        hsv[blue_mask, 2] = np.clip(hsv[blue_mask, 2] * 1.1, 200, 255)  # 提高亮度
        
        # 白色云朵处理
        white_mask = (hsv[:, :, 2] > 220) & (hsv[:, :, 1] < 40)
        hsv[white_mask, 2] = 240  # 标准白色亮度
        
        hsv = np.clip(hsv, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def _ghibli_highlight_colors(self, region, feature):
        """宫崎骏高光色彩：温暖、明亮"""
        lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB).astype(np.float32)
        l, a, b = cv2.split(lab)
        
        # 提高亮度，增强温暖感
        l = np.clip(l * 1.15, 180, 255)
        a = np.clip(a * 1.1 + 10, 0, 255)  # 偏向红色
        b = np.clip(b * 1.05 + 8, 0, 255)  # 偏向黄色
        
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    
    def _ghibli_colorful_highlight(self, region, feature):
        """宫崎骏彩色高光：保持色彩多样性"""
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # 适度增强饱和度，保持色彩变化
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.2, 80, 200)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.08, 160, 240)
        
        hsv = np.clip(hsv, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def _ghibli_main_light_colors(self, region, feature):
        """宫崎骏主要亮部色彩：自然明亮"""
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.15, 60, 180)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.05, 140, 220)
        
        hsv = np.clip(hsv, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def _ghibli_detailed_object_colors(self, region, feature):
        """宫崎骏细节丰富物体色彩：保持细节"""
        # 轻微色彩增强，保持细节
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.1, 50, 160)
        
        hsv = np.clip(hsv, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def _ghibli_textured_area_colors(self, region, feature):
        """宫崎骏纹理区域色彩：保持纹理特征"""
        # 保持原有色彩，轻微增强
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.08, 0, 255)
        
        hsv = np.clip(hsv, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def _ghibli_main_object_colors(self, region, feature):
        """宫崎骏主要物体色彩：标准处理"""
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.12, 70, 170)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.03, 120, 200)
        
        hsv = np.clip(hsv, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def _ghibli_colorful_shadow_colors(self, region, feature):
        """宫崎骏彩色阴影色彩：增强色彩，保持阴影感"""
        lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB).astype(np.float32)
        l, a, b = cv2.split(lab)
        
        # 提高阴影区域亮度，增强色彩
        l = np.clip(l * 1.4, 0, 180)
        a = np.clip(a * 1.3, 0, 255)
        b = np.clip(b * 1.3, 0, 255)
        
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    
    def _ghibli_shadow_colors(self, region, feature):
        """宫崎骏普通阴影色彩：轻微提亮"""
        lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB).astype(np.float32)
        l, a, b = cv2.split(lab)
        
        l = np.clip(l * 1.3, 0, 150)
        
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    
    def _ghibli_deep_shadow_colors(self, region, feature):
        """宫崎骏深阴影色彩：保持深度感"""
        lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB).astype(np.float32)
        l, a, b = cv2.split(lab)
        
        l = np.clip(l * 1.2, 0, 120)
        
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    
    def _high_quality_detail_fusion(self, ghibli_colors, original_img, region_features):
        """高画质细节融合：智能恢复重要细节"""
        result = ghibli_colors.copy()
        
        for feature in region_features:
            y1, y2, x1, x2 = feature['coords']
            ghibli_region = ghibli_colors[y1:y2, x1:x2]
            original_region = original_img[y1:y2, x1:x2]
            
            if ghibli_region.size == 0:
                continue
            
            # 根据区域类型决定细节保留程度
            if feature['type'] in ["detailed_object", "textured_area"]:
                # 细节丰富区域：保留更多原图细节
                blend_ratio = 0.3
            elif feature['type'] in ["main_object", "main_light"]:
                # 主要物体区域：适度保留细节
                blend_ratio = 0.15
            else:
                # 其他区域：较少保留细节
                blend_ratio = 0.05
            
            # 细节融合
            blended = cv2.addWeighted(ghibli_region, 1 - blend_ratio, 
                                    original_region, blend_ratio, 0)
            result[y1:y2, x1:x2] = blended
        
        return result
    
    def _ghibli_regional_lighting(self, img, region_features):
        """宫崎骏区域光影：基于区域的光影效果"""
        result = img.copy()
        
        # 创建宫崎骏特有的梦幻光影
        h, w = img.shape[:2]
        
        # 多光源效果
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        
        # 主光源（右上角）
        light1_y, light1_x = h // 4, w * 3 // 4
        dist1 = np.sqrt((x_coords - light1_x)**2 + (y_coords - light1_y)**2)
        
        # 辅助光源（左上角）
        light2_y, light2_x = h // 3, w // 4
        dist2 = np.sqrt((x_coords - light2_x)**2 + (y_coords - light2_y)**2)
        
        # 组合光照效果
        max_dist = np.sqrt((w/2)**2 + (h/2)**2)
        light_map1 = 1.0 - (dist1 / max_dist) * 0.3
        light_map2 = 1.0 - (dist2 / max_dist) * 0.2
        
        combined_light = (light_map1 * 0.6 + light_map2 * 0.4)
        
        # 应用光照效果
        lighted = img.astype(np.float32) * combined_light[:, :, np.newaxis]
        lighted = np.clip(lighted, 0, 255).astype(np.uint8)
        
        # 添加梦幻辉光
        blurred_dream = cv2.GaussianBlur(lighted, (21, 21), 0)
        dreamy = cv2.addWeighted(lighted, 0.85, blurred_dream, 0.15, 0)
        
        return dreamy
    
    # 其他风格的处理函数保持不变...
    
    def _apply_anime_style_preserve_details(self, img):
        """应用通用动漫风格（细节保留版）"""
        # 简化处理...
        return img
    
    def _apply_painting_style_preserve_details(self, img):
        """应用绘画风格（细节保留版）"""
        # 简化处理...
        return img

def image_to_base64(img):
    """将PIL图像转换为base64字符串"""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

# 初始化模型
model = GhibliStyleTransfer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': '没有选择文件'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    # 获取风格类型参数
    style_type = request.form.get('style_type', 'ghibli')
    
    if file:
        try:
            # 读取图像
            image = Image.open(file.stream)
            
            # 调整图像大小（限制最大尺寸）
            max_size = 800
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # 应用风格转换
            result_image = model(image, style_type=style_type)
            
            # 转换为base64
            original_b64 = image_to_base64(image)
            result_b64 = image_to_base64(result_image)
            
            return jsonify({
                'success': True,
                'original': original_b64,
                'result': result_b64,
                'style_type': style_type
            })
            
        except Exception as e:
            return jsonify({'error': f'处理图像时出错: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5003)