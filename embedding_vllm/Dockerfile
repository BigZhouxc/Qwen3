FROM qwenllm/qwenvl:qwen3vl-cu128

WORKDIR /app

# 安装 Conda（如系统 Python < 3.11）
ENV CONDA_DIR=/opt/conda
RUN bash -c 'python3 -c "import sys; exit(0 if sys.version_info >= (3,11) else 1)" || ( \
    echo "Installing Miniconda (py311)..." && \
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm -f /tmp/miniconda.sh && \
    /opt/conda/bin/conda create -y -n py311 python=3.11 && \
    /opt/conda/bin/conda clean -a -y )'

# 使用 conda 的 Python 3.11
ENV PATH=/opt/conda/envs/py311/bin:$PATH

# 设置 pip 国内源、安装依赖
RUN pip install -U "pip<25" "setuptools>=77.0.3,<80" wheel && \
    pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ && \
    pip config set global.timeout 120 && \
    pip install --no-cache-dir \
      fastapi uvicorn[standard] pillow numpy httpx accelerate safetensors \
      "transformers>=4.51.0" "qwen-vl-utils==0.0.14"

# 拷贝 Qwen3-VL-Embedding 并安装（用vllm框架的时候使用）
COPY Qwen3-VL-Embedding /app/Qwen3-VL-Embedding
RUN pip install --no-cache-dir -e /app/Qwen3-VL-Embedding


# 关键：只装 vLLM，并尽量避免它升级/替换 torch（如果失败再走下面方案B）
RUN python -m pip install --no-cache-dir "vllm>=0.14.0" --no-deps

# server.py 由 docker-compose 挂载，不 COPY
ENV PYTHONPATH=/app/Qwen3-VL-Embedding:/app

EXPOSE 38042

CMD ["bash", "-lc", "uvicorn server:app --host 0.0.0.0 --port 38042"]