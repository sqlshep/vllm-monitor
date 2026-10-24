# Slim base is fine: the NVIDIA Container Toolkit injects nvidia-smi and the
# driver libs at runtime when the container is started with GPU access
# (--gpus all / NVIDIA_VISIBLE_DEVICES). No CUDA image required just to query.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 9100

# Single worker: this is a lightweight read-only status endpoint.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9100"]
