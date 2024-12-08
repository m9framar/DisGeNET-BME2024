# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app
# Install PyTorch CPU first
RUN pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu

# Install PyTorch Geometric and dependencies
RUN pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
RUN pip install torch-geometric
# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source code and data
COPY src/ ./src/
COPY data/ ./data/

# Create results directory
RUN mkdir -p results

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=""

# Run training script
ENTRYPOINT ["python", "src/model_train.py"]
CMD ["--csv_file", "data/finalized_data.csv"]