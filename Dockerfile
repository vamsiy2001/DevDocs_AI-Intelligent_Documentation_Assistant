FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch FIRST (200MB vs 2.6GB CUDA wheel from PyPI)
RUN pip install --no-cache-dir torch==2.3.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining requirements (torch already present, won't reinstall CUDA version)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# HF Spaces requires user with uid 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH

EXPOSE 7860

CMD ["python", "app.py"]
