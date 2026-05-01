FROM python:3.10-slim
WORKDIR /app
COPY ./model_api /app/model_api
COPY ./model /app/model
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
ENV PYTHONUNBUFFERED=1
ENV API_KEY=changeme
ENV MODEL_PATH=/app/model/emotion_detection_v1.keras
EXPOSE 8000
CMD ["uvicorn", "model_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
