FROM person-tracker-base:latest

WORKDIR /app

# Copy requirements and install
COPY ./services/person-detector/requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip,id=permanent_pip_cache,sharing=locked \
    pip install -r requirements.txt

RUN apt-get update && apt-get install -y git && apt-get clean

COPY ./services/person-detector /app

COPY ./meter-watch-shared /app/meter_watch_shared

COPY ./alembic /app/alembic

COPY ./alembic.ini /app/alembic.ini

RUN pip install -e /app/meter_watch_shared

CMD ["python", "run.py"]

# Keep container alive for testing
# CMD ["tail", "-f", "/dev/null"]