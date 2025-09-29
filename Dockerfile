FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV headless
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1-mesa-glx \
    libgthread-2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV FLASK_APP=backend/app.py
ENV PYTHONPATH=/app/backend
ENV OPENCV_IO_ENABLE_JASPER=1
ENV QT_X11_NO_MITSHM=1
ENV DEBIAN_FRONTEND=noninteractive

# Expose port
EXPOSE 5000

# Create storage directory
RUN mkdir -p /app/backend/storage

# Command to run the application
CMD ["gunicorn", "-b", "0.0.0.0:5000", "backend.app:app", "--workers", "2", "--timeout", "120"]