#!/bin/bash
"""
Setup script for Ollama LLM-based retrieval

This script helps set up Ollama for use with the e-commerce search engine.
"""

echo "🚀 Setting up Ollama for LLM-based retrieval"
echo "=============================================="

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama is not installed"
    echo "Please install Ollama from: https://ollama.ai/"
    echo "Or run: curl -fsSL https://ollama.ai/install.sh | sh"
    exit 1
fi

echo "✅ Ollama is installed"

# Check if Ollama server is running
if ! curl -s http://localhost:11434/api/tags &> /dev/null; then
    echo "⚠️  Ollama server is not running"
    echo "Starting Ollama server..."
    ollama serve &
    sleep 5
fi

echo "✅ Ollama server is running"

# List of recommended models for retrieval
MODELS=("llama3" "mistral" "phi3" "gemma")

echo ""
echo "📦 Available models:"
ollama list

echo ""
echo "🔽 Pulling recommended models for retrieval..."

for model in "${MODELS[@]}"; do
    echo "Pulling $model..."
    if ollama pull "$model"; then
        echo "✅ $model pulled successfully"
    else
        echo "❌ Failed to pull $model"
    fi
done

echo ""
echo "🎯 Testing model availability..."

# Test llama3 (most recommended)
if ollama list | grep -q "llama3"; then
    echo "✅ llama3 is available"
    echo "Testing llama3..."
    echo "Hello, how are you?" | ollama run llama3
else
    echo "❌ llama3 not found"
fi

echo ""
echo "🎉 Ollama setup complete!"
echo ""
echo "Usage examples:"
echo "  python main.py --method ollama"
echo "  python examples/ollama_example.py"
echo ""
echo "Available models:"
ollama list
