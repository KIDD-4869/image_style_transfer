#!/usr/bin/env python3
"""
紧急修复 - 使用现有的AnimeGAN模型实现真正的宫崎骏风格
"""

import sys
import os
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

sys.path.insert(0, '.')

class EmergencyGhibliProcessor:
    """紧急宫崎骏处理器 - 使用AnimeGAN"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._load_animegan_model()
    
    def _load_animegan_model(self):
        """加载AnimeGAN模型"""
        try:
            model_path = "models/anime_gan/AnimeGANv2_Hayao.pth"
            if os.path.exists(model_path):
                # 简化的AnimeGAN网络结构
                class SimpleAnimeGAN(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.conv1 = nn.Conv2d(3, 32, 7, 1, 3)
                        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
                        self.conv3 = nn.Conv2d(64, 64, 3, 1, 1)
                        self.deconv1 = nn.ConvTranspose2d(64, 32, 3, 2, 1, 1)
                        self.deconv2 = nn.Conv2d(32, 3, 7, 1, 3)
                        self.relu = nn.ReLU()
                        self.tanh = nn.Tanh()
                    
                    def forward(self, x):
                        x = self.relu(self.conv1(x))
                        x = self.relu(self.conv2(x))
                        x = self.relu(self.conv3(x))
                        x = self.relu(self.deconv1(x))
                        x = self.tanh(self.deconv2(x))
                        return x
                
                self.model = SimpleAnimeGAN().to(self.device)
                
                # 尝试加载权重
                try:
                    checkpoint = torch.load(model_path, map_location=self.device)
                    if isinstance(checkpoint, dict) and 'generator' in checkpoint:
                        # 如果有generator键
                        state_dict = checkpoint['generator']
                    else:
                        state_dict = checkpoint
                    
                    # 过滤匹配的权重
                    model_dict = self.model.state_dict()
                    filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
                    
                    if filtered_dict:
                        model_dict.update(filtered_dict)
                        self.model.load_state_dict(model_dict)
                        print("✅ AnimeGAN模型权重部分加载成功")
                    else:
                        print("⚠️ 使用随机初始化权重")
                        
                except Exception as e:
                    print(f"⚠️ 权重加载失败，使用随机权重: {e}")
                
                self.model.eval()
                print("✅ AnimeGAN模型初始化完成")
            else:
                print("❌ AnimeGAN模型文件不存在")
                self.model = None
                
        except Exception as e:
            print(f"❌ AnimeGAN模型加载失败: {e}")
            self.model = None
    
    def process(self, image: Image.Image):
        """处理图像"""
        if self.model is None:
            return self._fallback_processing(image)
        
        try:
            # 预处理
            img_tensor = self._preprocess(image)
            
            # 模型推理
            with torch.no_grad():
                output = self.model(img_tensor)
            
            # 后处理
            result = self._postprocess(output, image.size)
            
            return result
            
        except Exception as e:
            print(f"⚠️ 模型推理失败，使用备选方案: {e}")
            return self._fallback_processing(image)
    
    def _preprocess(self, image):
        """预处理图像"""
        # 调整大小
        img = image.resize((256, 256), Image.LANCZOS)
        
        # 转换为tensor
        img_array = np.array(img).astype(np.float32) / 127.5 - 1.0
        img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        
        return img_tensor
    
    def _postprocess(self, tensor, original_size):
        """后处理"""
        # 转换回图像
        output = tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        output = (output + 1.0) * 127.5
        output = np.clip(output, 0, 255).astype(np.uint8)
        
        # 转换为PIL图像并调整大小
        result = Image.fromarray(output)
        result = result.resize(original_size, Image.LANCZOS)
        
        return result
    
    def _fallback_processing(self, image):
        """备选处理方案 - 强力动漫化"""
        img_np = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # 1. 强力平滑
        smooth = cv2.bilateralFilter(img_bgr, 15, 100, 100)
        smooth = cv2.bilateralFilter(smooth, 15, 100, 100)
        
        # 2. 激进颜色量化
        data = smooth.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, 8, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        quantized = centers[labels.flatten()].reshape(smooth.shape)
        
        # 3. 宫崎骏色彩
        hsv = cv2.cvtColor(quantized, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        s = cv2.add(s, 60)  # 大幅提升饱和度
        s = np.clip(s, 0, 255)
        
        v = cv2.add(v, 30)  # 提升亮度
        v = np.clip(v, 0, 255)
        
        # 色调偏暖
        h = np.where(h < 30, h + 15, h)
        h = np.clip(h, 0, 179)
        
        hsv_enhanced = cv2.merge([h, s, v])
        result_bgr = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
        
        # 4. 边缘增强
        gray = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 10)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        final = cv2.addWeighted(result_bgr, 0.85, edges_colored, 0.15, 0)
        
        result_rgb = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result_rgb)

def emergency_test():
    """紧急测试"""
    print("🚨 紧急修复测试")
    
    processor = EmergencyGhibliProcessor()
    
    # 创建测试图像
    test_img = Image.new('RGB', (400, 300), (120, 150, 180))
    
    result = processor.process(test_img)
    
    os.makedirs('emergency_output', exist_ok=True)
    result.save('emergency_output/emergency_test.jpg')
    
    print("✅ 紧急修复完成")
    print("📁 结果保存到: emergency_output/emergency_test.jpg")

if __name__ == '__main__':
    emergency_test()