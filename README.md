# DeepSeek Local Server

A complete Docker-based solution for running DeepSeek models locally with GPU/CPU support and low-end hardware optimization.

## Features

- 🚀 FastAPI-based server with OpenAI-compatible API
- 🐳 Docker and Docker Compose support
- 🎯 GPU acceleration with NVIDIA CUDA
- ⚡ 4-bit quantization for low-end hardware
- 🔄 Automatic model loading and caching
- 🏥 Health checks and monitoring
- 🌐 CORS support for web applications
- 📊 Memory optimization for resource-constrained environments

## Quick Start

### Prerequisites

- Docker and Docker Compose
- For GPU support: NVIDIA Docker runtime
- At least 4GB RAM (8GB recommended)
- Python 3.10+ (for client testing)

### Installation

1. Clone or create the project directory:
```bash
mkdir deepseek-local && cd deepseek-local
```

2. Copy all the provided files to the directory

3. Make the setup script executable:
```bash
chmod +x setup.sh
```

4. Run the setup script:
```bash
./setup.sh
```

### Manual Setup

#### For CPU-only systems:
```bash
docker-compose up -d
```

#### For GPU-enabled systems:
```bash
docker-compose -f docker-compose.gpu.yml up -d
```

## Usage

### Health Check
```bash
curl http://localhost:8000/health
```

### Simple Text Generation
```bash
curl -X POST "http://localhost:8000/generate?prompt=def%20fibonacci(n):&max_tokens=256&temperature=0.3"
```

### OpenAI-compatible Chat API
```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Write a Python function to sort a list."}
    ],
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

### Using the Python Client
```bash
python client.py
```

## Configuration

### Environment Variables

- `MODEL_NAME`: HuggingFace model name (default: `deepseek-ai/deepseek-coder-1.3b-base`)
- `USE_QUANTIZATION`: Enable 4-bit quantization (default: `true`)
- `CUDA_VISIBLE_DEVICES`: GPU devices to use

### Available Models

You can use different DeepSeek models by changing the `MODEL_NAME` environment variable:

- `deepseek-ai/deepseek-coder-1.3b-base` (recommended for low-end)
- `deepseek-ai/deepseek-coder-6.7b-base`
- `deepseek-ai/deepseek-llm-7b-base`

### Memory Requirements

| Model Size | RAM (CPU) | VRAM (GPU) | With Quantization |
|------------|-----------|------------|-------------------|
| 1.3B       | 4-6GB     | 2-3GB      | 1-2GB            |
| 6.7B       | 16-20GB   | 8-12GB     | 4-6GB            |
| 7B         | 18-24GB   | 10-14GB    | 5-7GB            |

## API Endpoints

### Health and Status
- `GET /` - Basic server info
- `GET /health` - Detailed health check

### Text Generation
- `POST /generate` - Simple text generation
- `POST /v1/chat/completions` - OpenAI-compatible chat API

## Docker Compose Profiles

### Default Profile
Basic DeepSeek server only:
```bash
docker-compose up -d
```

### With Caching
Include Redis for caching:
```bash
docker-compose --profile cache up -d
```

### With Reverse Proxy
Include Nginx reverse proxy:
```bash
docker-compose --profile proxy up -d
```

### All Services
```bash
docker-compose --profile cache --profile proxy up -d
```

## Monitoring and Logs

### View logs
```bash
docker-compose logs -f deepseek-server
```

### Monitor resources
```bash
docker stats
```

### Access logs directory
Logs are mounted to `./logs` directory in the host.

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce model size or enable quantization
2. **CUDA Out of Memory**: Enable quantization or use CPU-only mode
3. **Model Loading Slow**: First run downloads the model, subsequent runs use cache
4. **Port Conflicts**: Change port mapping in docker-compose.yml

### Performance Optimization

1. **Enable quantization** for low-end hardware
2. **Use GPU** if available for better performance
3. **Increase Docker memory limits** if needed
4. **Use SSD storage** for faster model loading

### Debug Mode

Run with debug logging:
```bash
docker-compose logs -f deepseek-server
```

## Development

### Local Development
```bash
pip install -r requirements.txt
python main.py
```

### Custom Models
Modify the `MODEL_NAME` environment variable to use different models.

### API Extensions
The FastAPI server can be extended with additional endpoints in `main.py`.

## License

This project is provided as-is for educational and development purposes.

## Contributing

Feel free to submit issues and enhancement requests!