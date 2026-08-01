FROM python:3.11-slim

WORKDIR /app

COPY docker-requirements.txt requirements.txt
RUN pip install --no-cache-dir torch transformers pillow numpy --extra-index-url https://download.pytorch.org/whl/cpu