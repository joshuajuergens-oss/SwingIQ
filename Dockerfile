FROM python:3.13-slim

RUN apt-get update && apt-get install -y \
    libxcb1 \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

ENV PORT=8080
ENV PYTHONUNBUFFERED=1
CMD ["sh", "-c", "gunicorn app:app --workers 2 --timeout 300 --bind 0.0.0.0:8080"]
