FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files
COPY . .

# Expose ports
EXPOSE 7860 8501

# Run both inference (OpenEnv) and Streamlit
CMD python inference.py & streamlit run app.py --server.port=8501 --server.address=0.0.0.0