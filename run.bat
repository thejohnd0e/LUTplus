@echo off
echo LUTplus - Image Processing Tool
echo ==============================================

REM Check for arguments
set NETWORK_MODE=
set GPU_MODE=
for %%A in (%*) do (
    if "%%A"=="--network" set NETWORK_MODE=--network
    if "%%A"=="--gpu" set GPU_MODE=--gpu
)

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed! Please install Python 3.9 or higher.
    pause
    exit /b 1
)

REM Check Python version for compatibility
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Detected Python version: %PYTHON_VERSION%

for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set PYTHON_MAJOR=%%a
    set PYTHON_MINOR=%%b
)

REM Check if using Python 3.13 or higher (not compatible)
if %PYTHON_MAJOR% EQU 3 (
    if %PYTHON_MINOR% GEQ 13 (
        echo ERROR: Python %PYTHON_VERSION% is not compatible with LUTplus.
        echo NumPy 1.x packages required by LUTplus are not available for Python 3.13+.
        echo Please install Python 3.12.x instead.
        echo See README.md for more information on compatibility.
        pause
        exit /b 1
    )
)

REM Check if virtual environment exists
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo Failed to create virtual environment!
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo Failed to activate virtual environment!
    pause
    exit /b 1
)

REM Update pip
echo Updating pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo Failed to update pip!
    pause
    exit /b 1
)

REM Install requirements
echo Installing required packages...
echo Trying primary packages...

REM Try primary installation
python -m pip install gradio==4.19.2 numpy==1.26.4 opencv-python==4.9.0.80 Pillow==10.2.0 colour-science==0.4.6 --only-binary=:all: --no-cache-dir
if errorlevel 1 (
    echo Primary installation failed. Trying with more compatible versions...
    
    REM Try fallback installation
    python -m pip install gradio==4.19.2 numpy==1.24.3 opencv-python==4.8.1.78 Pillow==10.0.0 colour-science==0.4.2 --only-binary=:all: --no-cache-dir
    if errorlevel 1 (
        echo Fallback installation failed. Trying minimal compatible versions...
        
        REM Try minimal compatible installation
        python -m pip install gradio==3.50.2 numpy==1.21.6 opencv-python==4.7.0.72 Pillow==9.5.0 colour-science==0.4.1 --only-binary=:all: --no-cache-dir
        if errorlevel 1 (
            echo Failed to install packages after multiple attempts!
            echo Trying to install from requirements_compatible.txt file...
            
            REM Last attempt - try to install from the requirements file
            if exist requirements_compatible.txt (
                python -m pip install -r requirements_compatible.txt --only-binary=:all: --no-cache-dir
                if errorlevel 1 (
                    echo All installation attempts failed!
                    echo Please try running manual_install.bat or install packages manually.
                    pause
                    exit /b 1
                )
            ) else (
                echo All installation attempts failed!
                echo Please install packages manually.
                pause
                exit /b 1
            )
        )
    )
)

REM Launch the application
echo Starting LUTplus...
if defined NETWORK_MODE (
    echo Network mode enabled. Application will be accessible from your local network.
    python app.py %NETWORK_MODE% %GPU_MODE%
) else (
    python app.py %GPU_MODE%
)

REM Keep the window open if there's an error
if errorlevel 1 (
    echo Application failed to start!
    pause
) 