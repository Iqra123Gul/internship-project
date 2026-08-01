FROM python:3.11-slim

WORKDIR /app

COPY docker-requirements.txt requirements.txt
RUN pip install --no-cache-dir pandas scikit-learn

COPY src/data_loader.py .

ENV CAPTIONS_PATH=/data/Flickr8k/captions.txt

CMD ["python", "data_loader.py"]