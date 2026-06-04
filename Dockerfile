FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY data/ ./data/
COPY hydra_frontend.html .


# Create logs directory
RUN touch hydra_queries.log

# Expose port
EXPOSE 7860

# Run the application
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "7860"]
