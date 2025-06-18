#!/bin/bash

# DeepSeek Local Setup Script

set -e

echo "Setting up DeepSeek Local Environment..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
mkdir -p logs ssl

# Check for GPU support
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected. Using GPU-enabled configuration."
    COMPOSE_FILE="docker-compose.gpu.yml"
else
    echo "No NVIDIA GPU detected. Using CPU-only configuration."
    COMPOSE_FILE="docker-compose.yml"
fi

# Build and start the services
echo "Building Docker images..."
docker-compose -f $COMPOSE_FILE build

echo "Starting services..."
docker-compose -f $COMPOSE_FILE up -d

echo "Waiting for services to be ready..."
sleep 30

# Test the service
echo "Testing the service..."
python3 -c "
import requests
import time
max_retries = 10
for i in range(max_retries):
    try:
        response = requests.get('http://localhost:8000/health')
        if response.status_code == 200:
            print('Service is ready!')
            break
    except:
        pass
    print(f'Waiting for service... ({i+1}/{max_retries})')
    time.sleep(10)
else:
    print('Service failed to start properly')
"

echo "Setup complete! Service is running on http://localhost:8000"
echo "You can test it with: python client.py"