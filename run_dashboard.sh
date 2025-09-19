#!/bin/bash

# Launch script for exploratory analysis dashboard
# Project 7 - MLE

echo "🏷️  StackOverflow Tags Analysis & Prediction Dashboard"
echo "====================================================="

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please install it with:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Check if dependencies are installed
if [ ! -f "uv.lock" ]; then
    echo "📦 Installing dependencies..."
    uv sync
else
    echo "✅ Dependencies already installed"
fi

# Check for data presence
if [ ! -f "data/train.csv" ] || [ ! -f "data/test.csv" ]; then
    echo "⚠️  No data detected. Please place data files in the data/ folder"
    exit 1
fi

# Environment variables
if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env file from template..."
    cp .env.example .env
    echo "📝 Please edit .env with your OpenAI API key to use tag prediction"
fi

echo "🚀 Launching dashboard..."
echo ""
echo "📍 Dashboard will be accessible at: http://localhost:8501"
echo "⏹️  To stop the dashboard: Ctrl+C"
echo ""

# Launch Streamlit
uv run streamlit run main.py
