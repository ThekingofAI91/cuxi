# ============================================================
# 促膝 · 生产镜像
# 构建时先装 CPU 版 torch（PyPI 默认 wheel 带 CUDA，体积大且服务器用不上），
# 再装锁定版本的其余依赖。
# 运行时需要挂载：chroma_db/（向量库）、data/（会话库）、模型缓存卷。
# ============================================================
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/models \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# libgomp1: torch/onnx 运行时需要；curl: 健康检查
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# 1) CPU 版 torch（约 200MB，比 CUDA 版小一个数量级）
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 2) 锁定版本的其余依赖（torch 已满足，pip 会跳过）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 非 root 运行；预建挂载点
RUN useradd -m appuser \
    && mkdir -p /app/chroma_db /app/data /app/models /app/logs \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# 应用启动有 20-30s 模型预热，start-period 必须给够，否则会被判失败反复重启
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -fs http://localhost:8000/health || exit 1

# --proxy-headers: nginx 反代后信任 X-Forwarded-For（否则限流按反代 IP 计数，形同虚设）
# --no-server-header: 抹掉 server: uvicorn 指纹
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--no-server-header"]
