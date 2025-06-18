FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Install core packages
RUN pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn==0.24.0 \
    numpy==1.24.4

# Install PyTorch CPU version
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Install compatible versions that work together
RUN pip install --no-cache-dir \
    huggingface_hub==0.20.0 \
    tokenizers==0.15.0 \
    safetensors==0.4.0 \
    regex>=2022.1.18 \
    accelerate==0.25.0 \
    transformers==4.35.2

# Create user
RUN useradd -m -u 1000 deepseek && \
    mkdir -p /app/cache/huggingface && \
    chown -R deepseek:deepseek /app

# Copy app
COPY . .
RUN chown -R deepseek:deepseek /app

USER deepseek

EXPOSE 8000
CMD ["python", "main.py"]