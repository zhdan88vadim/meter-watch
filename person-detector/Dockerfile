FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip,id=permanent_pip_cache,sharing=locked \
    pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y git && apt-get clean

# Copy application
COPY . .

# Create directories
RUN mkdir -p recordings models static/css static/js templates

# Run
# CMD ["python", "run.py"]

# Keep container alive for testing
CMD ["tail", "-f", "/dev/null"]