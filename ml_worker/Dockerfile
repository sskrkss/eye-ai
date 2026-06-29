FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY .env /app/

CMD ["python", "main.py"]