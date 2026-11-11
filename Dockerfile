# syntax=docker/dockerfile:1.7
# 用于 Streamlit UI 的运行时镜像。CI 产物镜像里不跑训练，Streamlit 启动后由用户上传 CSV。
# 两阶段都用 python:3.12-slim：使用 /opt/venv 让 site-packages 路径与 python 版本解耦。
FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# xgboost 在较新 manylinux 上需要 libgomp；shap/matplotlib 依赖少量构建工具
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# 用独立虚拟环境方便跨阶段 COPY，不依赖具体的 python3.X site-packages 路径。
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /app

# 先拷贝元数据，利用 Docker layer cache；后面再拷贝源代码。
COPY pyproject.toml README.md requirements.txt ./
COPY src ./src

RUN pip install --upgrade pip && \
    pip install .

# ---------- 运行时阶段 ----------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/* && \
    useradd --create-home --uid 1000 appuser

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
COPY streamlit_app.py ./streamlit_app.py
COPY src ./src
COPY data ./data

USER appuser
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health').read()" || exit 1

CMD ["streamlit", "run", "streamlit_app.py"]
