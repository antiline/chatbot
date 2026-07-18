FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# 세무 챗봇 웹 API (FastAPI). Ollama 주소는 OLLAMA_BASE_URL 로 주입.
CMD ["fastapi", "run", "api/index.py", "--host", "0.0.0.0", "--port", "8000"]
