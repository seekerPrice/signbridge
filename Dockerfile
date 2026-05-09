# HF Space Docker SDK runtime — pin python 3.11 (mediapipe + gradio happy)
FROM python:3.11-slim

# System deps for OpenCV / gradio video / pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# HF Spaces Docker SDK convention: app expects user "user" with UID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH HF_HOME=/home/user/.cache/huggingface

COPY --chown=user requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --user -r /app/requirements.txt

COPY --chown=user . /app

# HF Spaces Docker convention: app must listen on 0.0.0.0:7860
ENV GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    GRADIO_ANALYTICS_ENABLED=False
EXPOSE 7860

CMD ["python", "app.py"]
