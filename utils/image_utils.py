"""
图像处理工具模块
"""

import io
import base64
from PIL import Image
import numpy as np

def image_to_base64(img: Image.Image) -> str:
    """将PIL图像转换为base64字符串"""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def base64_to_image(base64_str: str) -> Image.Image:
    """将base64字符串转换为PIL图像"""
    if base64_str.startswith('data:image'):
        base64_str = base64_str.split(',')[1]
    
    img_data = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_data))
    return img

def resize_image(image: Image.Image, max_size: int) -> Image.Image:
    """调整图像大小，保持宽高比"""
    if max(image.size) <= max_size:
        return image
    
    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return image

def numpy_to_pil(img_np: np.ndarray) -> Image.Image:
    """将numpy数组转换为PIL图像"""
    if img_np.dtype == np.float32 or img_np.dtype == np.float64:
        img_np = (img_np * 255).astype(np.uint8)
    
    return Image.fromarray(img_np)

def pil_to_numpy(img: Image.Image) -> np.ndarray:
    """将PIL图像转换为numpy数组"""
    return np.array(img)