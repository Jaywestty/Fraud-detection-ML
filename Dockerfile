# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY . .

# Expose FastAPI and MLflow ports
EXPOSE 8000
EXPOSE 5000

# Run FastAPI (api.app:app) and optional MLflow UI for local experiments
CMD ["sh", "-c", "mlflow ui --backend-store-uri sqlite:///api/mlflow.db --default-artifact-root ./mlruns & uvicorn api.app:app --host 0.0.0.0 --port 8000"]