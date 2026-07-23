FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-dri \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY ./docker_base/base-requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# Copy requirements and install with PERMANENT cache
RUN --mount=type=cache,target=/root/.cache/pip,id=permanent_pip_cache,sharing=locked \
    pip install -r base-requirements.txt

CMD ["python", "-c", "print('Base image ready for service')"]
