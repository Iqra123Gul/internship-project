FROM flickr8k-base

WORKDIR /app

# System libraries required by opencv-python at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir opencv-python-headless fastapi uvicorn python-multipart

COPY src/ ./src/
COPY app/ ./app/

ENV MODEL_DIR=/app/models/week2_baseline_blip

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]