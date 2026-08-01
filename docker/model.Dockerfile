FROM flickr8k-base

WORKDIR /app
COPY src/model.py .

ENV MODEL_DIR=/models/week2_baseline_blip

CMD ["python", "model.py"]