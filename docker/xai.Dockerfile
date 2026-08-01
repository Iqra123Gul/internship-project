FROM flickr8k-base

WORKDIR /app
COPY src/model.py src/xai.py .

ENV MODEL_DIR=/models/week2_baseline_blip
ENV IMAGE_PATH=/data/sample.jpg

CMD ["python", "xai.py"]