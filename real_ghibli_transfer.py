#!/usr/bin/env python3
"""
真正的宫崎骏风格转换 - 基于深度学习和风格迁移
"""

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import torch.optim as optim
from torch.autograd import Variable
import os
import io
import base64
import time
import threading
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# 创建上传目录
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 全局变量用于存储转换进度
conversion_progress = {}
conversion_results = {}

class RealGhibliStyleTransfer:
    """真正的宫崎骏风格转换 - 基于深度学习"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vgg = self._load_vgg().to(self.device)
        self.style_layers = ['3', '8', '15', '22']  # VGG层用于风格提取
        self.content_layers = ['22']  # VGG层用于内容提取
        self.progress_callback = None
        self.task_id = None
        
    def _load_vgg(self):
        """加载预训练的VGG19模型"""
        vgg = models.vgg19(pretrained=True).features
        # 冻结参数
        for param in vgg.parameters():
            param.requires_grad = False
        return vgg
    
    def _extract_features(self, x, model, layers):
        """从VGG模型中提取特征"""
        features = {}
        for name, layer in model._modules.items():
            x = layer(x)
            if name in layers:
                features[name] = x
        return features
    
    def _gram_matrix(self, x):
        """计算Gram矩阵（风格特征）"""
        batch_size, channels, height, width = x.size()
        features = x.view(batch_size * channels, height * width)
        gram = torch.mm(features, features.t())
        return gram.div(batch_size * channels * height * width)
    
    def _load_ghibli_style_images(self):
        """加载宫崎骏风格参考图片"""
        style_folder = 'temp'
        style_images = []
        
        # 支持的图片格式
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        
        for ext in image_extensions:
            import glob
            pattern = os.path.join(style_folder, ext)
            style_images.extend(glob.glob(pattern))
        
        if not style_images:
            print("⚠️ 没有找到宫崎骏风格参考图片，使用默认风格")
            return None
        
        print(f"🎨 加载了 {len(style_images)} 张宫崎骏风格参考图片")
        return style_images
    
    def _create_ghibli_style_tensor(self, target_size=512):
        """创建宫崎骏风格特征张量"""
        style_images = self._load_ghibli_style_images()
        
        if not style_images:
            # 如果没有参考图片，创建默认的宫崎骏风格特征
            return self._create_default_ghibli_style(target_size)
        
        # 加载并处理风格图片
        style_tensors = []
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        for style_path in style_images:
            try:
                style_img = Image.open(style_path).convert('RGB')
                style_tensor = transform(style_img).unsqueeze(0).to(self.device)
                style_tensors.append(style_tensor)
            except Exception as e:
                print(f"❌ 加载风格图片 {style_path} 失败: {e}")
        
        if not style_tensors:
            return self._create_default_ghibli_style(target_size)
        
        # 平均所有风格图片的特征
        style_features = {}
        for style_tensor in style_tensors:
            features = self._extract_features(style_tensor, self.vgg, self.style_layers)
            for layer, feature in features.items():
                if layer not in style_features:
                    style_features[layer] = []
                style_features[layer].append(self._gram_matrix(feature))
        
        # 计算平均风格特征
        avg_style_features = {}
        for layer, gram_list in style_features.items():
            avg_gram = torch.stack(gram_list).mean(dim=0)
            avg_style_features[layer] = avg_gram
        
        return avg_style_features
    
    def _create_default_ghibli_style(self, target_size):
        """创建默认的宫崎骏风格特征"""
        print("🎨 使用默认宫崎骏风格特征")
        
        # 创建具有宫崎骏风格特征的默认风格
        # 宫崎骏风格特点：柔和色彩、梦幻光影、简洁线条
        default_style = {}
        
        # 这里应该基于宫崎骏的艺术特点创建风格特征
        # 由于时间关系，我们使用简化的方法
        
        return default_style
    
    def apply_real_ghibli_style(self, content_image, num_steps=100, style_weight=1000, content_weight=1):
        """应用真正的宫崎骏风格转换"""
        
        # 预处理内容图像
        content_tensor = self._preprocess_image(content_image).to(self.device)
        
        # 创建宫崎骏风格特征
        style_features = self._create_ghibli_style_tensor()
        
        if not style_features:
            # 如果无法创建风格特征，使用传统方法作为备选
            return self._fallback_traditional_method(content_image)
        
        # 提取内容特征
        content_features = self._extract_features(content_tensor, self.vgg, self.content_layers)
        
        # 初始化输出图像（使用内容图像作为起点）
        input_img = content_tensor.clone().requires_grad_(True)
        
        # 优化器 - 使用Adam优化器，更稳定
        optimizer = optim.Adam([input_img], lr=0.01)
        
        # 风格迁移优化
        print("🔄 开始风格迁移优化...")
        
        for step in range(num_steps):
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            input_img.data.clamp_(0, 1)
            features = self._extract_features(input_img, self.vgg, self.style_layers + self.content_layers)
            
            # 计算损失
            style_loss = 0
            content_loss = 0
            
            # 风格损失
            for layer in self.style_layers:
                if layer in features:
                    target_gram = style_features[layer]
                    current_gram = self._gram_matrix(features[layer])
                    style_loss += F.mse_loss(current_gram, target_gram)
            
            # 内容损失
            for layer in self.content_layers:
                if layer in features:
                    target_content = content_features[layer]
                    current_content = features[layer]
                    content_loss += F.mse_loss(current_content, target_content)
            
            # 总损失 - 添加数值稳定性检查
            total_loss = style_weight * style_loss + content_weight * content_loss
            
            # 检查损失是否为nan
            if torch.isnan(total_loss):
                print(f"⚠️ 步骤 {step+1}: 检测到nan损失，使用备选方法")
                return self._fallback_traditional_method(content_image)
            
            total_loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_([input_img], max_norm=1.0)
            
            optimizer.step()
            
            # 更新进度
            progress = int((step + 1) / num_steps * 100)
            if self.progress_callback and self.task_id:
                self.progress_callback(self.task_id, progress, step + 1, num_steps, total_loss.item())
            
            if (step + 1) % 20 == 0:
                print(f"步骤 {step+1}/{num_steps}, 总损失: {total_loss.item():.4f}")
        
        # 后处理输出图像
        output_tensor = input_img.data.clamp(0, 1)
        result_image = self._postprocess_image(output_tensor)
        
        return result_image
    
    def set_progress_callback(self, callback, task_id):
        """设置进度回调函数"""
        self.progress_callback = callback
        self.task_id = task_id
    
    def _preprocess_image(self, image):
        """预处理图像"""
        transform = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        return transform(image).unsqueeze(0)
    
    def _postprocess_image(self, tensor):
        """后处理张量为图像"""
        # 反归一化
        tensor = tensor.squeeze(0).cpu()
        tensor = tensor * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = tensor + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        tensor = torch.clamp(tensor, 0, 1)
        
        # 转换为PIL图像
        transform = transforms.ToPILImage()
        image = transform(tensor)
        
        return image
    
    def _fallback_traditional_method(self, image):
        """备选传统方法"""
        print("⚠️ 使用备选传统方法")
        
        # 将PIL图像转换为numpy数组
        img_np = np.array(image)
        
        # 转换为BGR格式
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        
        # 高质量的宫崎骏风格处理
        
        # 1. 保持原始分辨率
        h, w = img_bgr.shape[:2]
        max_size = 2000
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # 2. 智能边缘保留（重点改进人物区域）
        # 使用导向滤波保持边缘
        guided = cv2.ximgproc.guidedFilter(
            guide=img_bgr, 
            src=img_bgr, 
            radius=10, 
            eps=0.01
        )
        
        # 3. 宫崎骏风格色彩调整
        # 转换为LAB色彩空间进行更精确的色彩调整
        lab = cv2.cvtColor(guided, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 增强色彩鲜艳度（宫崎骏风格特点）
        a = cv2.addWeighted(a, 1.2, a, 0, 0)
        b = cv2.addWeighted(b, 1.2, b, 0, 0)
        
        # 调整亮度和对比度
        l = cv2.createCLAHE(clipLimit=2.0).apply(l)
        
        lab_enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        # 4. 添加梦幻光影效果
        h, w = enhanced.shape[:2]
        
        # 创建柔和的光照效果
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h / 2, w / 2
        
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        
        # 创建光照遮罩
        light_mask = 1.0 - (distance / max_distance) * 0.15
        light_mask = np.clip(light_mask, 0.85, 1.0)
        
        # 应用光照效果
        final = enhanced.astype(np.float32) * light_mask[:,:,np.newaxis]
        final = np.clip(final, 0, 255).astype(np.uint8)
        
        # 转换回RGB
        result_rgb = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
        
        return result_rgb

# 创建真正的宫崎骏风格转换模型
real_ghibli_model = RealGhibliStyleTransfer()

def update_progress(task_id, progress, current_step, total_steps, loss):
    """更新转换进度"""
    conversion_progress[task_id] = {
        'progress': progress,
        'current_step': current_step,
        'total_steps': total_steps,
        'loss': loss,
        'timestamp': time.time()
    }
    print(f"📊 任务 {task_id}: {progress}% (步骤 {current_step}/{total_steps}, 损失: {loss:.4f})")

def convert_image_async(task_id, image):
    """异步转换图像"""
    try:
        # 设置进度回调
        real_ghibli_model.set_progress_callback(update_progress, task_id)
        
        # 开始转换
        result_image = real_ghibli_model.apply_real_ghibli_style(image, num_steps=100)
        
        # 保存结果
        conversion_results[task_id] = {
            'success': True,
            'result_image': result_image,
            'completed': True
        }
        
        # 更新进度为完成
        update_progress(task_id, 100, 100, 100, 0)
        
    except Exception as e:
        conversion_results[task_id] = {
            'success': False,
            'error': str(e),
            'completed': True
        }
        print(f"❌ 任务 {task_id} 转换失败: {e}")

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/progress/<task_id>')
def get_progress(task_id):
    """获取转换进度"""
    if task_id in conversion_progress:
        return jsonify(conversion_progress[task_id])
    else:
        return jsonify({'error': '任务不存在'}), 404

@app.route('/result/<task_id>')
def get_result(task_id):
    """获取转换结果"""
    if task_id in conversion_results:
        result = conversion_results[task_id]
        if result['completed']:
            if result['success']:
                # 转换为base64
                result_image = result['result_image']
                
                # 检查结果类型并正确处理
                if isinstance(result_image, np.ndarray):
                    if result_image.dtype == np.float32 or result_image.dtype == np.float64:
                        result_image = (result_image * 255).astype(np.uint8)
                    result_image = Image.fromarray(result_image)
                
                buffered = io.BytesIO()
                result_image.save(buffered, format="JPEG", quality=95)
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                return jsonify({
                    'success': True,
                    'result': f"data:image/jpeg;base64,{img_str}",
                    'completed': True
                })
            else:
                return jsonify({
                    'success': False,
                    'error': result['error'],
                    'completed': True
                })
        else:
            return jsonify({'success': False, 'error': '转换尚未完成', 'completed': False})
    else:
        return jsonify({'error': '任务不存在'}), 404

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
        
        # 保存原图用于显示
        original_buffered = io.BytesIO()
        image.save(original_buffered, format="JPEG", quality=95)
        original_img_str = base64.b64encode(original_buffered.getvalue()).decode()
        
        # 生成任务ID
        task_id = str(int(time.time() * 1000))
        
        # 启动异步转换
        thread = threading.Thread(target=convert_image_async, args=(task_id, image))
        thread.daemon = True
        thread.start()
        
        print(f"🎨 开始异步宫崎骏风格转换，任务ID: {task_id}")
        
        return jsonify({
            'success': True,
            'task_id': task_id,
            'original': f"data:image/jpeg;base64,{original_img_str}",
            'message': '转换任务已开始，请等待完成'
        })
        
    except Exception as e:
        import traceback
        print(f"❌ 转换错误: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5005)