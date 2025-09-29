#!/bin/bash

# Build and deployment script for SIH Kolam Heritage Project

echo "🏗️  Building SIH Kolam Heritage Application..."

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

# Build the Docker image
echo "📦 Building Docker image..."
docker build -t kolam-heritage:latest .

if [ $? -eq 0 ]; then
    echo "✅ Docker image built successfully!"
    
    # Optional: Run the container locally for testing
    echo "🚀 Starting container on port 5000..."
    docker run -p 5000:5000 kolam-heritage:latest
else
    echo "❌ Docker build failed!"
    exit 1
fi