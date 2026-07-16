FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-dri \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY cnn/backend/requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# Copy requirements and install with PERMANENT cache
RUN --mount=type=cache,target=/root/.cache/pip,id=permanent_pip_cache,sharing=locked \
    pip install -r requirements.txt

# RUN --mount=type=cache,target=/root/.cache/pip,id=permanent_pip_cache,sharing=locked \
#     pip install --no-cache-dir -r requirements.txt

# Copy application
# COPY . .

COPY cnn/backend /app

COPY meter-watch-shared /app/meter_watch_shared

RUN pip install -e /app/meter_watch_shared


CMD ["python", "app.py"]

# Keep container alive for testing
# CMD ["tail", "-f", "/dev/null"]