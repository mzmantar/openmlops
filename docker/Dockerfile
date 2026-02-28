FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl unzip build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir --default-timeout=120 -r requirements.txt

COPY . /workspace
ENV PYTHONUNBUFFERED=1