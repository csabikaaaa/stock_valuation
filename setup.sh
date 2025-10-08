#!/bin/bash
# Stock Valuation Tool Setup Script

echo "🚀 Setting up Stock Valuation Tool..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo "❌ pip is not installed. Please install pip first."
    exit 1
fi

# Install requirements
echo "📦 Installing required packages..."
pip install -r requirements.txt

# Check if Streamlit is installed correctly
if command -v streamlit &> /dev/null; then
    echo "✅ Setup complete!"
    echo "🎯 To run the application, use: streamlit run main.py"
    echo "📖 Read README.md for detailed usage instructions"
else
    echo "❌ Setup failed. Please check error messages above."
    exit 1
fi