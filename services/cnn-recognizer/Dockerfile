FROM person-tracker-base:latest

WORKDIR /app

COPY ./services/cnn-recognizer /app

COPY ./meter-watch-shared /app/meter_watch_shared

RUN pip install -e /app/meter_watch_shared

CMD ["python", "app.py"]

# Keep container alive for testing
# CMD ["tail", "-f", "/dev/null"]