#!/bin/bash

# DeepSeek Local Setup with Virtual Environment (macOS compatible)

set -e

echo "🚀 Setting up DeepSeek Local Environment with Virtual Environment..."

# Check Python version (macOS compatible)
if command -v python3 &> /dev/null; then
    python_version=$(python3 --version | sed 's/Python //')
    echo "📍 Python version: $python_version"
else
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    echo "💡 You can install it with: brew install python"
    exit 1
fi

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "❌ requirements.txt not found!"
    echo "💡 Make sure you've created the requirements.txt file in the current directory."
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "⚡ Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📈 Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt

echo "✅ Virtual environment setup complete!"
echo ""
echo "🎯 To activate the virtual environment in the future, run:"
echo "   source venv/bin/activate"
echo ""
echo "🧪 To test the setup locally, run:"
echo "   source venv/bin/activate"
echo "   python test_imports.py"
echo "   python main.py"
echo ""
echo "🐳 To run with Docker:"
echo "   docker-compose -f docker-compose.tinkr.yml up --build"
echo ""
echo "🔍 Next steps:"
echo "   1. source venv/bin/activate"
echo "   2. python test_imports.py"
echo "   3. python main.py"