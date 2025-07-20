# backend/Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies for lxml/readability speedups
RUN apt-get update && apt-get install -y build-essential libxml2-dev libxslt1-dev && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# HuggingFace cache (optional: pre-pull models to reduce first request latency)
# RUN python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('prhegde/t5-query-reformulation-RL');"

COPY . .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host","0.0.0.0", "--port","8000"] 