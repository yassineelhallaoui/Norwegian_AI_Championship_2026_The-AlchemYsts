FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

COPY requirements.txt .
RUN pip install -r requirements.txt && \
    pip install fastapi uvicorn pydantic

COPY . .

# Expose the standard Cloud Run port
EXPOSE 8080

CMD ["python", "main.py"]
# Build timestamp: 1774152571
# Build timestamp: 1774153434
