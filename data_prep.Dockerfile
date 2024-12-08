# data_prep.Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install only required packages
RUN pip install requests pandas

# Copy source
COPY src/data_prep.py .

# Create data directory
RUN mkdir -p data

# Set environment variables
ENV PYTHONPATH=/app

# Set entrypoint
ENTRYPOINT ["python", "data_prep.py"]