#!/bin/bash

# AI Legal Assistant Setup Script
# ================================

set -e  # Exit on any error

echo "🚀 Setting up AI Legal Assistant..."
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python installation
print_status "Checking Python installation..."
if command_exists python3; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    print_success "Python found: $PYTHON_VERSION"
else
    print_error "Python 3 is required but not found!"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

# Check pip
if command_exists pip3; then
    print_success "pip3 found"
    PIP_CMD="pip3"
elif command_exists pip; then
    print_success "pip found"
    PIP_CMD="pip"
else
    print_error "pip is required but not found!"
    exit 1
fi

# Install Python dependencies
print_status "Installing Python dependencies..."
$PIP_CMD install -r requirements.txt

if [ $? -eq 0 ]; then
    print_success "Python dependencies installed"
else
    print_error "Failed to install Python dependencies"
    exit 1
fi

# Check/Install Ollama
print_status "Checking Ollama installation..."
if command_exists ollama; then
    print_success "Ollama is already installed"
else
    print_warning "Ollama not found. Installing..."
    
    # Detect OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        print_status "Installing Ollama for Linux..."
        curl -fsSL https://ollama.ai/install.sh | sh
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        print_status "Installing Ollama for macOS..."
        brew install ollama || {
            print_warning "Homebrew not found. Please install Ollama manually:"
            echo "Visit: https://ollama.ai/download"
            exit 1
        }
    else
        print_warning "Please install Ollama manually for your OS:"
        echo "Visit: https://ollama.ai/download"
        exit 1
    fi
fi

# Start Ollama service (if not running)
print_status "Starting Ollama service..."
if pgrep -x "ollama" > /dev/null; then
    print_success "Ollama service is already running"
else
    print_status "Starting Ollama service..."
    ollama serve &
    sleep 5
fi

# Download recommended model
print_status "Checking for AI models..."
if ollama list | grep -q "llama3.2:3b"; then
    print_success "llama3.2:3b model is already available"
else
    print_status "Downloading llama3.2:3b model (this may take a few minutes)..."
    ollama pull llama3.2:3b
    
    if [ $? -eq 0 ]; then
        print_success "Model downloaded successfully"
    else
        print_warning "Failed to download llama3.2:3b. Trying alternative..."
        print_status "Downloading mistral:7b model..."
        ollama pull mistral:7b
    fi
fi

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p chroma_db
mkdir -p logs

# Set permissions
chmod +x setup.sh

print_success "Setup completed successfully!"
echo ""
echo "🎉 Your AI Legal Assistant is ready!"
echo "=================================="
echo ""
echo "To start the application:"
echo "  streamlit run streamlit_app.py"
echo ""
echo "Or run the CLI version:"
echo "  python agent/legal_assistant.py"
echo ""
echo "📋 What's installed:"
echo "  ✅ Python dependencies (LangChain, ChromaDB, etc.)"
echo "  ✅ Ollama (Local LLM server)"
echo "  ✅ AI Model (llama3.2:3b or mistral:7b)"
echo "  ✅ Vector database setup"
echo ""
echo "💡 Pro Tips:"
echo "  • The first run will take time to process all legal documents"
echo "  • Keep Ollama running in the background"
echo "  • Check logs/ directory for troubleshooting"
echo ""
echo "🚀 Ready to help people understand Indian laws!"
