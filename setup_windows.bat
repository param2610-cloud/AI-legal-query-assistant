@echo off
REM AI Legal Assistant Setup Script for Windows
REM ==========================================

setlocal enabledelayedexpansion

echo 🚀 Setting up AI Legal Assistant for Windows...
echo ============================================

REM Colors are limited in Windows CMD, using simple text formatting

:print_status
echo [INFO] %~1
goto :eof

:print_success
echo [SUCCESS] %~1
goto :eof

:print_warning
echo [WARNING] %~1
goto :eof

:print_error
echo [ERROR] %~1
goto :eof

REM Check Python installation
call :print_status "Checking Python installation..."
python --version >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
    call :print_success "Python found: !PYTHON_VERSION!"
) else (
    call :print_error "Python 3 is required but not found!"
    echo Please install Python 3.8 or higher from https://python.org
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

REM Check pip
pip --version >nul 2>&1
if %errorlevel% equ 0 (
    call :print_success "pip found"
    set PIP_CMD=pip
) else (
    call :print_error "pip is required but not found!"
    echo Please reinstall Python with pip included
    pause
    exit /b 1
)

REM Install Python dependencies
call :print_status "Installing Python dependencies..."
%PIP_CMD% install -r requirements.txt
if %errorlevel% equ 0 (
    call :print_success "Python dependencies installed"
) else (
    call :print_error "Failed to install Python dependencies"
    echo Please check your internet connection and Python installation
    pause
    exit /b 1
)

REM Check/Install Ollama
call :print_status "Checking Ollama installation..."
where ollama >nul 2>&1
if %errorlevel% equ 0 (
    call :print_success "Ollama is already installed"
) else (
    call :print_warning "Ollama not found. Please install manually..."
    echo.
    echo Please download and install Ollama for Windows:
    echo 1. Visit: https://ollama.ai/download
    echo 2. Download the Windows installer
    echo 3. Run the installer as Administrator
    echo 4. Restart this script after installation
    echo.
    pause
    echo Checking again for Ollama...
    where ollama >nul 2>&1
    if %errorlevel% neq 0 (
        call :print_error "Ollama still not found. Please install it first."
        pause
        exit /b 1
    )
)

REM Check if Ollama service is running
call :print_status "Checking Ollama service..."
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe">NUL
if %errorlevel% equ 0 (
    call :print_success "Ollama service is already running"
) else (
    call :print_status "Starting Ollama service..."
    start /B ollama serve
    timeout /t 5 /nobreak >nul
)

REM Download recommended model
call :print_status "Checking for AI models..."
ollama list | findstr "llama3.2:3b" >nul 2>&1
if %errorlevel% equ 0 (
    call :print_success "llama3.2:3b model is already available"
) else (
    call :print_status "Downloading llama3.2:3b model (this may take a few minutes)..."
    ollama pull llama3.2:3b
    if %errorlevel% equ 0 (
        call :print_success "Model downloaded successfully"
    ) else (
        call :print_warning "Failed to download llama3.2:3b. Trying alternative..."
        call :print_status "Downloading mistral:7b model..."
        ollama pull mistral:7b
        if %errorlevel% neq 0 (
            call :print_error "Failed to download AI models. Please check your internet connection."
        )
    )
)

REM Create necessary directories
call :print_status "Creating necessary directories..."
if not exist "chroma_db" mkdir chroma_db
if not exist "logs" mkdir logs

REM Check if Streamlit is available
streamlit --version >nul 2>&1
if %errorlevel% neq 0 (
    call :print_warning "Installing Streamlit..."
    %PIP_CMD% install streamlit
)

call :print_success "Setup completed successfully!"
echo.
echo 🎉 Your AI Legal Assistant is ready!
echo ==================================
echo.
echo To start the application:
echo   streamlit run streamlit_app.py
echo.
echo Or run the CLI version:
echo   python agent/legal_assistant.py
echo.
echo 📋 What's installed:
echo   ✅ Python dependencies (LangChain, ChromaDB, etc.)
echo   ✅ Ollama (Local LLM server)
echo   ✅ AI Model (llama3.2:3b or mistral:7b)
echo   ✅ Vector database setup
echo.
echo 💡 Pro Tips:
echo   • The first run will take time to process all legal documents
echo   • Keep Ollama running in the background
echo   • Check logs/ directory for troubleshooting
echo   • If you encounter issues, run as Administrator
echo.
echo 🚀 Ready to help people understand Indian laws!
echo.
pause
