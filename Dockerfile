FROM python:3.11-slim

WORKDIR /app

COPY docker-requirements.txt .

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

RUN pip install --no-cache-dir -r docker-requirements.txt

COPY app/ ./app/
COPY src/ ./src/

ENV PYTHONUNBUFFERED=1
ENV MODEL_DIR=IqraGulAI/flickr8k-blip-baseline

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]