FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV API_BASE_URL=""
ENV MODEL_NAME="digital-detox-coach"
ENV HF_TOKEN=""

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . .

# Create results directory
RUN mkdir -p results

# Make inference script executable
RUN chmod +x inference.py

# Expose port for API
EXPOSE 5000

# Run OpenEnv API server (for hackathon validation)
CMD python openenv_api.py