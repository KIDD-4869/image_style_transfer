#!/usr/bin/env python3
"""
增强版宫崎骏风格转换配置
"""

import os
import cv2

# 模型路径配置
MODEL_PATHS = {
    # 人脸检测模型
    'face_detector': {
        'dnn_model': 'models/face_detector/opencv_face_detector_uint8.pb',
        'dnn_config': 'models/face_detector/opencv_face_detector.pbtxt',
        'haar_cascade': cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    },
    
    # 语义分割模型
    'segmentation': {
        'model_name': 'deeplabv3_resnet50',
        'weights': 'DEFAULT'
    }
}

# 处理参数配置
PROCESSING_PARAMS = {
    # 人脸检测参数
    'face_detection': {
        'confidence_threshold': 0.5,
        'min_face_size': (30, 30),
        'scale_factor': 1.1,
        'min_neighbors': 5
    },
    
    # 语义分割参数
    'segmentation': {
        'target_size': 520,
        'person_class': 15,  # COCO数据集人物类别
        'mask_smooth_kernel': (9, 9)
    },
    
    # 人脸美化参数
    'face_enhancement': {
        'skin_smooth_d': 9,
        'skin_smooth_sigma_color': 75,
        'skin_smooth_sigma_space': 75,
        'eye_brightness_boost': 20,
        'lip_saturation_boost': 30,
        'face_brightness_boost': 10,
        'face_red_tone_boost': 5,
        'face_yellow_tone_boost': 3
    },
    
    # 背景处理参数
    'background_processing': {
        'classic_intensity': 0.5,
        'fantasy_intensity': 0.6,
        'nostalgic_intensity': 0.7,
        'vibrant_intensity': 0.8,
        'lighting_intensity': 1.0
    }
}

# 宫崎骏风格模板配置
GHIBLI_STYLES = {
    'classic': {
        'name': '经典宫崎骏',
        'description': '柔和色彩，梦幻光影，最接近宫崎骏原作的风格',
        'saturation_boost': 30,
        'hue_shift': 5,
        'lighting_intensity': 0.5
    },
    'fantasy': {
        'name': '梦幻宫崎骏',
        'description': '更强烈的色彩和光影效果，创造梦幻般的氛围',
        'saturation_boost': 40,
        'hue_shift': 8,
        'lighting_intensity': 0.6
    },
    'nostalgic': {
        'name': '怀旧宫崎骏',
        'description': '柔和的怀旧色调，带有轻微的胶片颗粒感',
        'saturation_boost': -20,
        'hue_shift': 10,
        'lighting_intensity': 0.7
    },
    'vibrant': {
        'name': '鲜艳宫崎骏',
        'description': '高饱和度鲜艳色彩，适合风景和色彩丰富的场景',
        'saturation_boost': 50,
        'hue_shift': 3,
        'lighting_intensity': 0.8
    }
}

# 性能优化配置
PERFORMANCE_CONFIG = {
    # 图像尺寸限制
    'max_image_size': 2048,
    'target_processing_size': 800,
    
    # 批处理设置
    'batch_size': 1,
    'parallel_processing': False,
    
    # 内存优化
    'memory_limit_mb': 2048,
    'gpu_memory_fraction': 0.5
}

# 错误处理和回退配置
ERROR_HANDLING = {
    # 模型加载失败时的回退策略
    'fallback_strategies': {
        'face_detection': 'haar_cascade',  # DNN失败时使用Haar级联
        'segmentation': 'full_image',      # 分割失败时使用全图处理
        'neural_style': 'traditional_cv'   # 神经网络失败时使用传统CV
    },
    
    # 错误重试配置
    'max_retries': 3,
    'retry_delay': 2,
    
    # 超时设置
    'timeout_seconds': 300
}

def get_model_path(model_type, model_name):
    """获取模型路径"""
    if model_type in MODEL_PATHS and model_name in MODEL_PATHS[model_type]:
        path = MODEL_PATHS[model_type][model_name]
        
        # 检查模型文件是否存在
        if os.path.exists(path):
            return path
        else:
            print(f"⚠️ 模型文件不存在: {path}")
            return None
    
    return None

def get_processing_param(category, param_name):
    """获取处理参数"""
    if category in PROCESSING_PARAMS and param_name in PROCESSING_PARAMS[category]:
        return PROCESSING_PARAMS[category][param_name]
    return None

def get_style_config(style_name):
    """获取风格配置"""
    if style_name in GHIBLI_STYLES:
        return GHIBLI_STYLES[style_name]
    # 默认返回经典风格
    return GHIBLI_STYLES['classic']