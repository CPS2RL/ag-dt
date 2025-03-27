FROM python:3.10-slim
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create results directory
RUN mkdir -p /app/results

# Copy application files
COPY model/ model
COPY data/ data
COPY test_scalability.py .

CMD ["python", "test_scalability.py"]
