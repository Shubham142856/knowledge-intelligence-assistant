FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for document parsing
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY dashboard.py .

EXPOSE 8000 8501

CMD ["sh", "-c", \
     "uvicorn src.api.app:app --host 0.0.0.0 --port 8000 & \
      streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0"]
