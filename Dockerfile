FROM python:3.11-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
ENV PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gfortran && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip && \
    pip install numpy pandas scikit-learn ta boto3

CMD ["bash", "-lc", "\
python etl/feature_engineering.py && \
python modeling/ml_direction_classifier.py \
"]
# CMD ["bash", "-lc", "python etl/feature_engineering.py"]
    