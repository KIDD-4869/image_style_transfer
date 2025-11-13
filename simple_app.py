#!/usr/bin/env python3
"""
简化的宫崎骏风格转换应用 - 使用新的真正动漫化算法
"""

import io
import base64
from flask import Flask, render_template, request, jsonify
import os
import time
import threading
from PIL import Image
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 导入卡通风格化器
from core.cartoon_stylizer import cartoon_stylizer

app = Flask(__name__)

# 基本配置
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# 创建上传目录
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 全局变量存储转换结果
conversion_results = {}
conversion_progress = {}

def convert_image_simple(task_id, image):
    """简化的图像转换函数"""
    try:
        print(f"🎨 开始转换任务: {task_id}")
        
        # 设置进度回调
        def progress_callback(tid, progress, current_step, total_steps, loss):
            conversion_progress[tid] = {
                'progress': progress,
                'current_step': current_step,
                'total_steps': total_steps,
                'timestamp': time.time()
            }
            print(f"📊 任务 {tid}: {progress}%")
        
        cartoon_stylizer.set_progress_callback(progress_callback, task_id)
        
        # 应用卡通风格化
        result_image = cartoon_stylizer.apply_ghibli_style(image)
        
        # 保存结果
        conversion_results[task_id] = {
            'success': True,
            'result_image': result_image,
            'original_image': image,
            'completed': True
        }
        
        print(f"✅ 任务 {task_id} 转换完成")
        
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

@app.route('/upload', methods=['POST'])
def upload_file():
    """处理文件上传"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': '没有选择文件'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': '没有选择文件'})
        
        # 检查文件类型
        allowed_extensions = {'jpg', 'jpeg', 'png', 'bmp'}
        file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        if file_ext not in allowed_extensions:
            return jsonify({'success': False, 'error': f'不支持的文件格式，请上传 {allowed_extensions} 格式的图片'})
        
        # 读取图片
        try:
            image = Image.open(file.stream).convert('RGB')
        except Exception as e:
            return jsonify({'success': False, 'error': f'图片文件损坏: {str(e)}'})
        
        print(f"📊 图片尺寸: {image.size[0]}x{image.size[1]}")
        
        # 生成任务ID
        task_id = str(int(time.time() * 1000))
        
        # 保存原图用于显示
        original_buffered = io.BytesIO()
        image.save(original_buffered, format="JPEG", quality=95)
        original_img_str = base64.b64encode(original_buffered.getvalue()).decode()
        
        # 启动异步转换
        thread = threading.Thread(target=convert_image_simple, args=(task_id, image))
        thread.daemon = True
        thread.start()
        
        print(f"🎨 开始宫崎骏风格转换，任务ID: {task_id}")
        
        return jsonify({
            'success': True,
            'task_id': task_id,
            'original': f"data:image/jpeg;base64,{original_img_str}",
            'message': '宫崎骏风格转换已开始'
        })
        
    except Exception as e:
        logger.error(f"上传错误: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/progress/<task_id>')
def get_progress(task_id):
    """获取转换进度"""
    if task_id in conversion_progress:
        return jsonify(conversion_progress[task_id])
    elif task_id in conversion_results:
        if conversion_results[task_id]['completed']:
            return jsonify({'progress': 100, 'completed': True})
    
    return jsonify({'progress': 0, 'error': '任务不存在'})

@app.route('/result/<task_id>')
def get_result(task_id):
    """获取转换结果"""
    if task_id not in conversion_results:
        return jsonify({'success': False, 'error': '任务不存在'})
    
    result_data = conversion_results[task_id]
    
    if not result_data['completed']:
        return jsonify({'success': False, 'error': '任务仍在处理中'})
    
    if not result_data['success']:
        return jsonify({'success': False, 'error': result_data['error']})
    
    try:
        # 转换结果为base64
        result_image = result_data['result_image']
        buffered = io.BytesIO()
        result_image.save(buffered, format="JPEG", quality=95)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # 原图
        original_image = result_data['original_image']
        original_buffered = io.BytesIO()
        original_image.save(original_buffered, format="JPEG", quality=95)
        original_img_str = base64.b64encode(original_buffered.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'result': f"data:image/jpeg;base64,{img_str}",
            'original': f"data:image/jpeg;base64,{original_img_str}",
            'completed': True
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': f'结果处理失败: {str(e)}'})

if __name__ == '__main__':
    print("🚀 启动宫崎骏风格转换应用...")
    print("🎨 使用卡通风格化算法 - 让真实变成动漫")
    print("🌐 访问地址: http://localhost:5005")
    app.run(debug=True, host='0.0.0.0', port=5005)