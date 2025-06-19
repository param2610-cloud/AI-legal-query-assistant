# AI Legal Assistant Setup Script for Windows (PowerShell)
# ========================================================

param(
    [switch]$SkipOllamaInstall,
    [switch]$Help
)

if ($Help) {
    Write-Host "AI Legal Assistant Setup Script for Windows" -ForegroundColor Cyan
    Write-Host "Usage: .\setup_windows.ps1 [-SkipOllamaInstall] [-Help]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Parameters:"
    Write-Host "  -SkipOllamaInstall    Skip automatic Ollama installation"
    Write-Host "  -Help                 Show this help message"
    exit 0
}

# Set execution policy for current session
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force

# Color functions
function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Function to check if command exists
function Test-Command {
    param([string]$Command)
    $null = Get-Command $Command -ErrorAction SilentlyContinue
    return $?
}

Write-Host "🚀 Setting up AI Legal Assistant for Windows..." -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan

try {
    # Check Python installation
    Write-Status "Checking Python installation..."
    if (Test-Command "python") {
        $pythonVersion = python --version 2>&1
        Write-Success "Python found: $pythonVersion"
    } else {
        Write-Error "Python 3 is required but not found!"
        Write-Host "Please install Python 3.8 or higher from https://python.org" -ForegroundColor Yellow
        Write-Host "Make sure to check 'Add Python to PATH' during installation" -ForegroundColor Yellow
        Read-Host "Press Enter to exit"
        exit 1
    }

    # Check pip
    if (Test-Command "pip") {
        Write-Success "pip found"
        $pipCmd = "pip"
    } else {
        Write-Error "pip is required but not found!"
        Write-Host "Please reinstall Python with pip included" -ForegroundColor Yellow
        Read-Host "Press Enter to exit"
        exit 1
    }

    # Install Python dependencies
    Write-Status "Installing Python dependencies..."
    $installResult = & $pipCmd install -r requirements.txt 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Python dependencies installed"
    } else {
        Write-Error "Failed to install Python dependencies"
        Write-Host "Error: $installResult" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }

    # Check/Install Ollama
    Write-Status "Checking Ollama installation..."
    if (Test-Command "ollama") {
        Write-Success "Ollama is already installed"
    } elseif (-not $SkipOllamaInstall) {
        Write-Warning "Ollama not found. Attempting to install..."
        
        # Check if winget is available
        if (Test-Command "winget") {
            Write-Status "Installing Ollama using winget..."
            winget install Ollama.Ollama
            if ($LASTEXITCODE -eq 0) {
                Write-Success "Ollama installed successfully"
                # Refresh PATH
                $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
            } else {
                Write-Warning "Winget installation failed. Please install manually."
            }
        } else {
            Write-Warning "Winget not available. Please install Ollama manually:"
            Write-Host "1. Visit: https://ollama.ai/download" -ForegroundColor Yellow
            Write-Host "2. Download the Windows installer" -ForegroundColor Yellow
            Write-Host "3. Run the installer as Administrator" -ForegroundColor Yellow
            Read-Host "Press Enter after installing Ollama to continue"
        }
        
        # Check again
        if (-not (Test-Command "ollama")) {
            Write-Error "Ollama still not found. Please install it manually."
            Write-Host "Visit: https://ollama.ai/download" -ForegroundColor Yellow
            Read-Host "Press Enter to exit"
            exit 1
        }
    }

    # Check if Ollama service is running
    Write-Status "Checking Ollama service..."
    $ollamaProcess = Get-Process -Name "ollama" -ErrorAction SilentlyContinue
    if ($ollamaProcess) {
        Write-Success "Ollama service is already running"
    } else {
        Write-Status "Starting Ollama service..."
        Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
        Start-Sleep -Seconds 5
    }

    # Download recommended model
    Write-Status "Checking for AI models..."
    $modelList = ollama list 2>&1
    if ($modelList -match "llama3.2:3b") {
        Write-Success "llama3.2:3b model is already available"
    } else {
        Write-Status "Downloading llama3.2:3b model (this may take a few minutes)..."
        $pullResult = ollama pull llama3.2:3b 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Model downloaded successfully"
        } else {
            Write-Warning "Failed to download llama3.2:3b. Trying alternative..."
            Write-Status "Downloading mistral:7b model..."
            $pullResult = ollama pull mistral:7b 2>&1
            if ($LASTEXITCODE -ne 0) {
                Write-Error "Failed to download AI models. Please check your internet connection."
                Write-Host "You can try downloading models manually later using: ollama pull <model-name>" -ForegroundColor Yellow
            }
        }
    }

    # Create necessary directories
    Write-Status "Creating necessary directories..."
    @("chroma_db", "logs") | ForEach-Object {
        if (-not (Test-Path $_)) {
            New-Item -ItemType Directory -Path $_ -Force | Out-Null
        }
    }

    # Check if Streamlit is available
    if (-not (Test-Command "streamlit")) {
        Write-Status "Installing Streamlit..."
        & $pipCmd install streamlit | Out-Null
    }

    # Create batch files for easy launching
    Write-Status "Creating launch shortcuts..."
    
    # Create Streamlit launcher
    $streamlitLauncher = @"
@echo off
echo Starting AI Legal Assistant Web Interface...
streamlit run streamlit_app.py
pause
"@
    $streamlitLauncher | Out-File -FilePath "start_web_app.bat" -Encoding ASCII

    # Create CLI launcher
    $cliLauncher = @"
@echo off
echo Starting AI Legal Assistant CLI...
python agent/legal_assistant.py
pause
"@
    $cliLauncher | Out-File -FilePath "start_cli.bat" -Encoding ASCII

    Write-Success "Setup completed successfully!"
    Write-Host ""
    Write-Host "🎉 Your AI Legal Assistant is ready!" -ForegroundColor Green
    Write-Host "====================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "To start the application:" -ForegroundColor Cyan
    Write-Host "  Double-click: start_web_app.bat (Web Interface)" -ForegroundColor Yellow
    Write-Host "  Double-click: start_cli.bat (Command Line)" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Or run manually:" -ForegroundColor Cyan
    Write-Host "  streamlit run streamlit_app.py" -ForegroundColor Yellow
    Write-Host "  python agent/legal_assistant.py" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "📋 What's installed:" -ForegroundColor Cyan
    Write-Host "  ✅ Python dependencies (LangChain, ChromaDB, etc.)" -ForegroundColor Green
    Write-Host "  ✅ Ollama (Local LLM server)" -ForegroundColor Green
    Write-Host "  ✅ AI Model (llama3.2:3b or mistral:7b)" -ForegroundColor Green
    Write-Host "  ✅ Vector database setup" -ForegroundColor Green
    Write-Host "  ✅ Windows launch shortcuts" -ForegroundColor Green
    Write-Host ""
    Write-Host "💡 Pro Tips:" -ForegroundColor Cyan
    Write-Host "  • The first run will take time to process all legal documents" -ForegroundColor Yellow
    Write-Host "  • Keep Ollama running in the background" -ForegroundColor Yellow
    Write-Host "  • Check logs/ directory for troubleshooting" -ForegroundColor Yellow
    Write-Host "  • Run as Administrator if you encounter permission issues" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "🚀 Ready to help people understand Indian laws!" -ForegroundColor Green

} catch {
    Write-Error "An error occurred during setup: $($_.Exception.Message)"
    Write-Host "Please check the error message above and try again." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Read-Host "Press Enter to exit"
