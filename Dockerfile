# 多阶段构建的Docker文件
FROM python:3.9-slim as base

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    && rm -rf /var/lib/apt/lists/*

# 创建非root用户
RUN groupadd -r appuser && useradd -r -g appuser appuser

# 复制requirements文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 生产阶段
FROM base as production

# 复制应用代码
COPY --chown=appuser:appuser . .

# 创建必要的目录
RUN mkdir -p logs cache static/uploads models && \
    chown -R appuser:appuser logs cache static/uploads models

# 切换到非root用户
USER appuser

# 设置环境变量
ENV PYTHONPATH=/app
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5003/health', timeout=5)"

# 暴露端口
EXPOSE 5003

# 启动命令
CMD ["python", "app.py"]

# 开发阶段
FROM base as development

# 安装开发依赖
RUN pip install --no-cache-dir pytest pytest-cov black flake8 mypy

# 复制应用代码
COPY --chown=appuser:appuser . .

# 创建必要的目录
RUN mkdir -p logs cache static/uploads models && \
    chown -R appuser:appuser logs cache static/uploads models

# 切换到非root用户
USER appuser

# 设置环境变量
ENV PYTHONPATH=/app
ENV FLASK_APP=app.py
ENV FLASK_ENV=development

# 暴露端口
EXPOSE 5003

# 启动命令
CMD ["python", "app.py"]