# FROM python:3.10-slim
FROM person-tracker-base:latest

WORKDIR /app

# Install system dependencies
# RUN apt-get update && apt-get install -y \
#     libgl1 \
#     libglib2.0-0 \
#     libsm6 \
#     libxext6 \
#     libxrender-dev \
#     libgomp1 \
#     && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY person-detector/requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip,id=permanent_pip_cache,sharing=locked \
    pip install -r requirements.txt

RUN apt-get update && apt-get install -y git && apt-get clean

# Copy application
# COPY . .

COPY person-detector /app

COPY ./meter-watch-shared /app/meter_watch_shared

RUN pip install -e /app/meter_watch_shared

# Run
CMD ["python", "run.py"]

# Keep container alive for testing
# CMD ["tail", "-f", "/dev/null"]