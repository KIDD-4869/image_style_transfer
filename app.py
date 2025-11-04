import io
import base64
from flask import Flask, render_template, request, jsonify
import os
import time
import threading
from PIL import Image
import numpy as np
import logging

# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from config.settings import config
from core.real_ghibli_transfer import RealGhibliStyleTransfer
from core.ghibli_enhanced import GhibliEnhancedTransfer
from auto_learning import RealGhibliStyleTransferWithLearning

app = Flask(__name__)

# 加载配置
app.config.from_object(config['default'])

# 创建上传目录
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 初始化宫崎骏风格转换模型
real_ghibli_model = RealGhibliStyleTransfer(use_neural_network=True)
ghibli_enhanced_model = GhibliEnhancedTransfer()

# 任务管理
task_progress = {}
task_results = {}

def update_progress(task_id, progress, current_step, total_steps, loss):
    """更新转换进度"""
    task_progress[task_id] = {
        'progress': progress,
        'current_step': current_step,
        'total_steps': total_steps,
        'loss': loss,
        'timestamp': time.time()
    }
    print(f"📊 任务 {task_id}: {progress}% (步骤 {current_step}/{total_steps}, 损失: {loss:.4f})")

def convert_image_async(task_id, image, use_neural=True, style_intensity=1.0, use_enhanced=False):
    """异步转换图像
    
    Args:
        task_id: 任务ID
        image: 输入图像
        use_neural: 是否使用神经网络风格迁移
        style_intensity: 风格强度 (0.5-2.0)
        use_enhanced: 是否使用增强版功能
    """
    try:
        if use_enhanced:
            # 使用增强版功能
            print("🎨 使用增强版宫崎骏风格转换")
            
            # 设置进度回调
            ghibli_enhanced_model.set_progress_callback(update_progress, task_id)
            
            # 开始转换
            result_image = ghibli_enhanced_model.apply_enhanced_ghibli_style(image)
        else:
            # 使用基础版功能
            print("🎨 使用基础版宫崎骏风格转换")
            
            # 设置进度回调
            real_ghibli_model.set_progress_callback(update_progress, task_id)
            
            # 根据风格强度调整参数
            style_weight = int(300000 * style_intensity)
            num_steps = max(50, min(200, int(100 * style_intensity)))
            
            print(f"🎯 转换参数: 神经网络={use_neural}, 风格强度={style_intensity}, 步数={num_steps}")
            
            # 开始转换
            result_image = real_ghibli_model.apply_real_ghibli_style(
                image, 
                num_steps=num_steps, 
                style_weight=style_weight,
                use_neural=use_neural
            )
        
        # 保存结果（包括原图）
        task_results[task_id] = {
            'success': True,
            'result_image': result_image,
            'original_image': image,  # 保存原图
            'completed': True
        }
        
        # 更新进度为完成
        update_progress(task_id, 100, 100, 100, 0)
        
    except Exception as e:
        task_results[task_id] = {
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
    if task_id in task_progress:
        return jsonify(task_progress[task_id])
    elif task_id in task_results:
        # 如果任务已完成，返回完成状态
        result = task_results[task_id]
        if result['completed']:
            return jsonify({
                'progress': 100,
                'current_step': 100,
                'total_steps': 100,
                'loss': 0,
                'timestamp': time.time()
            })
    
    # 任务不存在或尚未开始
    return jsonify({'error': '任务不存在或尚未开始'}), 404

@app.route('/result/<task_id>')
def get_result(task_id):
    """获取转换结果"""
    # 首先检查任务是否在结果中
    if task_id in task_results:
        result = task_results[task_id]
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
                
                # 同时返回原图
                original_image = result.get('original_image')
                if original_image:
                    original_buffered = io.BytesIO()
                    original_image.save(original_buffered, format="JPEG", quality=95)
                    original_img_str = base64.b64encode(original_buffered.getvalue()).decode()
                else:
                    # 如果没有保存原图，返回默认值
                    original_img_str = ""
                
                return jsonify({
                    'success': True,
                    'result': f"data:image/jpeg;base64,{img_str}",
                    'original': f"data:image/jpeg;base64,{original_img_str}",
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
    
    # 检查任务是否在进度中但尚未完成
    if task_id in task_progress:
        progress = task_progress[task_id]
        return jsonify({
            'success': False, 
            'error': '任务仍在处理中', 
            'completed': False,
            'progress': progress.get('progress', 0),
            'current_step': progress.get('current_step', 0),
            'total_steps': progress.get('total_steps', 100)
        })
    
    # 任务不存在或已完成但结果已过期
    return jsonify({'success': False, 'error': '任务不存在或已完成', 'completed': True})

@app.route('/upload', methods=['POST'])
def upload_file():
    """处理文件上传和风格转换"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': '没有选择文件'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': '没有选择文件'})
        
        # 检查文件类型
        allowed_extensions = {'jpg', 'jpeg', 'png', 'bmp', 'gif'}
        file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        if file_ext not in allowed_extensions:
            return jsonify({'success': False, 'error': f'不支持的文件格式: {file_ext}，请上传 {allowed_extensions} 格式的图片'})
        
        # 检查文件大小
        file.seek(0, 2)  # 移动到文件末尾
        file_size = file.tell()
        file.seek(0)  # 重置文件指针
        
        if file_size > app.config['MAX_CONTENT_LENGTH']:
            return jsonify({'success': False, 'error': f'文件太大，最大支持 {app.config["MAX_CONTENT_LENGTH"] // (1024*1024)}MB'})
        
        # 读取图片
        image = Image.open(file.stream)
        
        # 检查图片尺寸 - 移除尺寸限制，支持任意尺寸图片
        max_size = app.config.get('MAX_IMAGE_SIZE', 0)
        if max_size > 0 and max(image.size) > max_size * 2:  # 如果设置了最大尺寸才检查
            return jsonify({'success': False, 'error': f'图片尺寸过大，最大支持 {max_size}x{max_size} 像素'})
        
        # 记录图片尺寸信息
        print(f"📊 图片尺寸: {image.size[0]}x{image.size[1]}, 格式: {image.format}")
        
        # 保存原图用于显示
        original_buffered = io.BytesIO()
        image.save(original_buffered, format="JPEG", quality=95)
        original_img_str = base64.b64encode(original_buffered.getvalue()).decode()
        
        # 生成任务ID
        task_id = str(int(time.time() * 1000))
        
        # 在启动异步任务之前，先创建进度记录
        update_progress(task_id, 0, 0, 100, 0)
        
        # 获取处理参数
        use_neural = request.form.get('use_neural', 'true').lower() == 'true'
        style_intensity = float(request.form.get('style_intensity', '1.0'))
        use_enhanced = request.form.get('use_enhanced', 'false').lower() == 'true'
        
        # 启动异步转换
        thread = threading.Thread(target=convert_image_async, args=(task_id, image, use_neural, style_intensity, use_enhanced))
        thread.daemon = True
        thread.start()
        
        print(f"🎨 开始异步宫崎骏风格转换，任务ID: {task_id}")
        print(f"📊 图片信息: {image.size[0]}x{image.size[1]}, 格式: {image.format}")
        
        return jsonify({
            'success': True,
            'task_id': task_id,
            'original': f"data:image/jpeg;base64,{original_img_str}",
            'message': '转换任务已开始，请等待完成',
            'estimated_time': '预计处理时间: 30-60秒'
        })
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        
        # 分类错误信息
        if 'image file is truncated' in error_msg.lower():
            error_msg = '图片文件损坏，请重新上传'
        elif 'cannot identify image file' in error_msg.lower():
            error_msg = '无法识别图片格式，请上传有效的图片文件'
        
        logger.error(f"❌ 转换错误: {error_msg}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': error_msg})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5006)